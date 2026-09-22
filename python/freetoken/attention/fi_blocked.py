"""Prefill over an fp8 KV cache, computed in bf16 one block of context at a time.

flashinfer's fa2 prefill reads fp8 KV through a slower tile path than 16-bit KV: on an RTX 4060
(sm89, head_dim 256, GQA 8) it reaches 21.5 TFLOPS against 28.1 for the same attention over
bf16 KV. The fp8 slab is kept (halving the context is not an option), but for a long cached
prefix the context is gathered and dequantized block by block into a bf16 scratch, each block
runs through the bf16 ragged kernel, and the partial results are merged by their log-sum-exp.
Blocks of past keys attend in full; the last block holds the remainder of the past plus the
chunk's own tokens and attends causally, which flashinfer aligns to the bottom-right corner.

Nothing here allocates per call. The scratch lives in the tail of the float workspace that
flashinfer's split-KV prefill never reaches for these shapes, the wrappers plan into disjoint
slices of the shared int workspace, and all of it exists before the first request, because the
attention backend is created after the KV budget is solved and any buffer it takes later comes
out of the headroom prefill activations need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

BLOCK = 16384  # past keys per block; 16k and 32k measured the same, the smaller one fits
QCAP = 2048  # query rows per request the scratch holds; longer chunks take the fp8 kernel
MAX_REQS = 4
HEAD_BYTES = 128 << 20  # float workspace kept for the wrappers' own split-KV temporaries
INT_SLICE = 1 << 20


@dataclass
class BlockedReq:
    q0: int  # query rows of this request within the batch
    q1: int
    k0: int  # first of its KV slots within the batch's index array
    prefix: int  # cached tokens before this chunk
    full: BatchPrefillWithRaggedKVCacheWrapper | None = None  # [q, BLOCK] non-causal
    tail: BatchPrefillWithRaggedKVCacheWrapper | None = None  # [q, rem + q] causal

    @property
    def n_full(self) -> int:
        return self.prefix // BLOCK

    @property
    def rem(self) -> int:
        return self.prefix % BLOCK


def _align(n: int) -> int:
    return (n + 255) & ~255


class BlockedFp8Prefill:
    def __init__(
        self,
        float_ws: torch.Tensor,
        int_ws: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

        self.hq, self.hkv, self.d, self.dtype = num_qo_heads, num_kv_heads, head_dim, dtype
        kv_rows = BLOCK + QCAP
        kv16 = kv_rows * num_kv_heads * head_dim * dtype.itemsize
        kv8 = kv_rows * num_kv_heads * head_dim
        out = QCAP * num_qo_heads * head_dim * dtype.itemsize
        lse = QCAP * num_qo_heads * 4
        need = HEAD_BYTES + 2 * _align(kv16) + 2 * _align(kv8) + _align(out) + _align(lse)
        if float_ws.numel() < need or int_ws.numel() < 2 * MAX_REQS * INT_SLICE:
            raise ValueError(
                f"workspace too small for blocked fp8 prefill: float {float_ws.numel()} < {need} "
                f"or int {int_ws.numel()} < {2 * MAX_REQS * INT_SLICE}"
            )

        off = HEAD_BYTES

        def take(nbytes: int, dt: torch.dtype, *shape: int) -> torch.Tensor:
            nonlocal off
            view = float_ws[off : off + nbytes].view(dt).view(*shape)
            off += _align(nbytes)
            return view

        self.k = take(kv16, dtype, kv_rows, num_kv_heads, head_dim)
        self.v = take(kv16, dtype, kv_rows, num_kv_heads, head_dim)
        self.k8 = take(kv8, torch.uint8, kv_rows, num_kv_heads, head_dim)
        self.v8 = take(kv8, torch.uint8, kv_rows, num_kv_heads, head_dim)
        self.o = take(out, dtype, QCAP, num_qo_heads, head_dim)
        self.s = take(lse, torch.float32, QCAP, num_qo_heads)

        head = float_ws[:HEAD_BYTES]
        self.pool: list[BatchPrefillWithRaggedKVCacheWrapper] = []
        for i in range(2 * MAX_REQS):
            w = BatchPrefillWithRaggedKVCacheWrapper(head, kv_layout="NHD", backend="fa2")
            # its own 8 MiB int buffer goes back to the allocator; a single-request plan needs KBs
            w.reset_workspace_buffer(head, int_ws[i * INT_SLICE : (i + 1) * INT_SLICE])
            self.pool.append(w)
        self._warm()

    def _plan(self, w, q: int, kv: int, causal: bool) -> None:
        w.plan(
            qo_indptr=torch.tensor([0, q], dtype=torch.int32),
            kv_indptr=torch.tensor([0, kv], dtype=torch.int32),
            num_qo_heads=self.hq, num_kv_heads=self.hkv, head_dim_qk=self.d,
            causal=causal, q_data_type=self.dtype, kv_data_type=self.dtype,
            non_blocking=True,
        )

    def _warm(self) -> None:
        """JIT-build both mask modes and the merge now, not on the first long prompt."""
        from flashinfer import merge_state_in_place

        q = torch.zeros(16, self.hq, self.d, dtype=self.dtype, device=self.k.device)
        for causal, w in ((False, self.pool[0]), (True, self.pool[1])):
            self._plan(w, 16, 16, causal)
            w.run(q, self.k[:16], self.v[:16], out=self.o[:16], lse=self.s[:16])
        merge_state_in_place(self.o[:8], self.s[:8], self.o[8:16], self.s[8:16])
        torch.cuda.current_stream().synchronize()

    def plan(self, reqs: list[BlockedReq]) -> None:
        """One plan per shape per request, reused by every layer of the batch."""
        for i, r in enumerate(reqs):
            q = r.q1 - r.q0
            r.full = self.pool[2 * i] if r.n_full else None
            r.tail = self.pool[2 * i + 1]
            if r.full is not None:
                self._plan(r.full, q, BLOCK, causal=False)
            self._plan(r.tail, q, r.rem + q, causal=True)

    def _gather(self, kflat: torch.Tensor, vflat: torch.Tensor, idx: torch.Tensor) -> int:
        n = idx.numel()
        torch.index_select(kflat, 0, idx, out=self.k8[:n])
        torch.index_select(vflat, 0, idx, out=self.v8[:n])
        self.k[:n].copy_(self.k8[:n].view(torch.float8_e4m3fn))
        self.v[:n].copy_(self.v8[:n].view(torch.float8_e4m3fn))
        return n

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        indices: torch.Tensor,
        reqs: list[BlockedReq],
    ) -> torch.Tensor:
        from flashinfer import merge_state_in_place

        kflat = k_cache.view(-1, self.hkv, self.d).view(torch.uint8)
        vflat = v_cache.view(-1, self.hkv, self.d).view(torch.uint8)
        o = torch.empty(q.shape[0], self.hq, self.d, dtype=q.dtype, device=q.device)
        lse = torch.empty(q.shape[0], self.hq, dtype=torch.float32, device=q.device)
        for r in reqs:
            nq = r.q1 - r.q0
            qr, orr, lr = q[r.q0 : r.q1], o[r.q0 : r.q1], lse[r.q0 : r.q1]
            spans = [(r.k0 + j * BLOCK, r.k0 + (j + 1) * BLOCK, r.full) for j in range(r.n_full)]
            spans.append((r.k0 + r.n_full * BLOCK, r.k0 + r.prefix + nq, r.tail))
            for j, (s, e, w) in enumerate(spans):
                n = self._gather(kflat, vflat, indices[s:e])
                if j == 0:
                    w.run(qr, self.k[:n], self.v[:n], out=orr, lse=lr)
                else:
                    w.run(qr, self.k[:n], self.v[:n], out=self.o[:nq], lse=self.s[:nq])
                    merge_state_in_place(orr, lr, self.o[:nq], self.s[:nq])
        return o


__all__ = ["BLOCK", "MAX_REQS", "QCAP", "BlockedFp8Prefill", "BlockedReq"]
