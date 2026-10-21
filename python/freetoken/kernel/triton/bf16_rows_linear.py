"""bf16 linear for decode-sized batches: one program per output row, every token row on its own.

cuBLAS answers a 2..3-row bf16 GEMM against a small weight (the routers, GDN ``in_proj_ba``)
with a 16x16 WMMA tile kernel that takes ~8.6 us on sm_120 whatever N is, against ~1.4-1.8 us
here; that is ~0.5 ms of every MTP verify step on Qwen3.5. Each token row is reduced over the
whole K in one fixed order, so a row's output does not depend on how many rows share the call.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_ROWS = 4
MAX_OUT = 1024
MAX_IN = 8192


@triton.jit
def _rows_linear_kernel(x_ptr, w_ptr, y_ptr, N, K, M: tl.constexpr, BLOCK_K: tl.constexpr):
    n = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    kmask = offs_k < K
    w = tl.load(w_ptr + n * K + offs_k, mask=kmask, other=0.0).to(tl.float32)
    for m in tl.static_range(M):
        x = tl.load(x_ptr + m * K + offs_k, mask=kmask, other=0.0).to(tl.float32)
        tl.store(y_ptr + m * N + n, tl.sum(w * x, axis=0).to(y_ptr.dtype.element_ty))


def rows_linear_ok(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> bool:
    return (
        b is None
        and x.is_cuda
        and x.dim() == 2
        and 1 <= x.shape[0] <= MAX_ROWS
        and x.dtype == w.dtype == torch.bfloat16
        and w.shape[0] <= MAX_OUT
        and w.shape[1] <= MAX_IN
        and x.is_contiguous()
        and w.is_contiguous()
    )


def rows_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    (m, k), n = x.shape, w.shape[0]
    y = torch.empty(m, n, device=x.device, dtype=x.dtype)
    _rows_linear_kernel[(n,)](x, w, y, n, k, M=m, BLOCK_K=triton.next_power_of_2(k), num_warps=4)
    return y
