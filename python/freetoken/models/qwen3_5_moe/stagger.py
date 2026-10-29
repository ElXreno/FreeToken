"""Staggered MTP verify (``FREETOKEN_STAGGER=1``): the two rows of a one-request verify on two streams, row 1 half a
layer behind row 0, so the CPU pool, which bounds a hybrid MoE layer, works on one row's experts while the GPU runs
the other row's dense part (norms, mixer, router, shared expert) and its own routing, miss copy and GEMMs.

Per layer k and row r: ``D`` the dense part, ``R`` the routing and capped-fetch ensure, ``G`` the miss copy and the
GPU experts' GEMMs, ``C`` the CPU pool's task (submit, wait, read back), ``A`` the sum with ``G`` and the merge with the
shared expert::

    row r stream:  D_r,k  R_r,k  G_r,k ............ A_r,k  D_r,k+1 ...
    CPU pool:                   C_0,k  C_1,k  C_0,k+1  C_1,k+1 ...

The pool runs one task at a time. With a handshake slot per row (each with its own IO rows and its own stream for the
submit, wait and read back) C_1,k is submitted as soon as R_1,k is done and the pool starts it when C_0,k finishes;
with one slot, one stream carries both rows' handshakes and C_1,k is submitted once C_0,k has read its result back.
Neither waits for the GEMMs. A row's ensure waits until the other row's last call no longer reads the cache's copy
staging (its miss copy, or the copy engine's snapshot of the victims); the slots that call routes to are its most
recent, so the ensure never evicts them. The expert GEMMs still run one at a time: a row's wait for the other row's
last ones. Row 1's dense part waits for row 0's, which wrote the KV row and the GDN stash it reads. Each row is a
one-row verify against the same KV and GDN state. Whatever moves the misses (``copy_missing``) runs inside G
unchanged.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import torch
from freetoken.attention.linear import verify_row_fla_metadata
from freetoken.core import Batch, get_global_ctx
from freetoken.env import ENV

if TYPE_CHECKING:
    from .model import Qwen3_5Model

ROWS = 2
_STREAMS: dict[tuple[torch.device, str], torch.cuda.Stream] = {}


def _stream(device: torch.device, name: str, priority: int = 0) -> torch.cuda.Stream:
    stream = _STREAMS.get((device, name))
    if stream is None:
        stream = _STREAMS[(device, name)] = torch.cuda.Stream(device=device, priority=priority)
    return stream


def can_stagger(batch: Batch, layers) -> bool:
    ctx = get_global_ctx()
    return bool(
        ENV.STAGGER
        and batch.phase == "verify"
        and batch.rows_per_req == ROWS
        and batch.verify_row is None
        and len(batch.padded_reqs) == 1
        and batch.mm_embeds is None
        and ctx.attn_backend.supports_verify_rows
        and all(layer.splits_at_experts and layer.mlp.experts.stages_hybrid for layer in layers)
    )


def row_batch(batch: Batch, row: int) -> Batch:
    """Row ``row`` of a one-request verify as its own batch, with its attention and GDN metadata."""
    rb = copy.copy(batch)
    rb.rows_per_req = 1
    rb.verify_row = row
    rows = slice(row, row + 1)
    rb.positions = batch.positions[rows]
    if batch.mrope_positions is not None:
        rb.mrope_positions = batch.mrope_positions[:, rows].contiguous()
    rb.out_loc = batch.out_loc[rows]
    rb.features = None
    get_global_ctx().attn_backend.prepare_verify_row(batch, rb)
    if batch.fla_metadata is not None:
        rb.fla_metadata = verify_row_fla_metadata(batch.fla_metadata, row)
    return rb


def staggered_forward(model: Qwen3_5Model, x: torch.Tensor) -> torch.Tensor:
    """``model``'s layers and final norm over the embedded rows ``x`` [2, hidden], staggered; returns [2, hidden]."""
    ctx = get_global_ctx()
    batch = ctx.batch
    layers = model.layers.op_list
    executor = layers[0].mlp.experts.offload_cache.cpu_executor
    rows = [row_batch(batch, r) for r in range(ROWS)]
    streams = (torch.cuda.current_stream(), _stream(x.device, "row"))
    # one handshake slot per row lets row 1's task be submitted while row 0's runs; with one, the tasks share a stream
    slotted = getattr(executor, "slots", 1) >= ROWS
    hosts = tuple(_stream(x.device, f"cpu{r if slotted else 0}", priority=-2) for r in range(ROWS))
    hidden = [x[r : r + 1] for r in range(ROWS)]
    residual: list[torch.Tensor | None] = [None] * ROWS
    dense0 = torch.cuda.Event()
    routed, staged, gemm, done = ([torch.cuda.Event() for _ in range(ROWS)] for _ in range(4))
    # tensors one stream allocates and another reads stay referenced until the join
    keep = []
    for s in (streams[1], *hosts):
        s.wait_stream(streams[0])
    try:
        for k, layer in enumerate(layers):
            experts = layer.mlp.experts
            for r, s in enumerate(streams):
                with torch.cuda.stream(s):
                    ctx._batch = rows[r]
                    if r == 1:
                        s.wait_event(dense0)
                    h, residual[r], dense = layer.dense(hidden[r], residual[r])
                    if r == 0:
                        dense0.record(s)
                    if k or r:
                        s.wait_event(staged[1 - r])
                    routes = experts.hybrid_route(h, dense[0])
                    routed[r].record(s)
                    gpu = experts.hybrid_gpu(h, routes, staged[r], gemm[1 - r] if k or r else None)
                    gemm[r].record(s)
                with torch.cuda.stream(hosts[r]):
                    hosts[r].wait_event(routed[r])
                    pending = executor.decode_submit(experts.layer_id, h, routes[0], routes[3], slot=r if slotted else 0)
                    cpu = executor.decode_sync(pending)
                    done[r].record(hosts[r])
                with torch.cuda.stream(s):
                    s.wait_event(done[r])
                    hidden[r] = layer.merge(gpu.add_(cpu), dense)
                    experts.hybrid_done()
                keep += [h, *dense, *routes, gpu, cpu, pending]
    finally:
        ctx._batch = batch
    joined = torch.cuda.Event()
    joined.record(streams[1])
    streams[0].wait_event(joined)
    out = [model.norm.forward(hidden[r], residual[r])[0] for r in range(ROWS)]
    return torch.cat(out)


__all__ = ["can_stagger", "row_batch", "staggered_forward"]
