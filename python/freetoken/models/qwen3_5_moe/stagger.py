"""Staggered MTP verify (``FREETOKEN_STAGGER=1``): the two rows of a one-request verify on two streams, row 1 half a
layer behind row 0, so one row's dense part (norms, mixer, router, shared expert; VRAM bound) runs while the other
row's routed experts wait on host memory and PCIe.

Per layer k, ``D`` the dense part and ``M`` the routed experts::

    D0_k ; [M0_k || D1_k] ; [M1_k || D0_k+1] ; ...

Each row is computed as a one-row verify against the same KV and GDN state: row 1's dense part waits for row 0's,
which wrote the KV row and the GDN stash it reads. The routed parts run one at a time, M0_k, M1_k, M0_k+1: the
expert cache's staging, the CPU pool's one task in flight and the slots a GEMM reads belong to one call at a time.
Whatever moves the misses (``copy_missing``) runs inside M unchanged.
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
_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _row_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _STREAMS.get(device)
    if stream is None:
        stream = _STREAMS[device] = torch.cuda.Stream(device=device)
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
        and all(layer.splits_at_experts for layer in layers)
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
    rows = [row_batch(batch, r) for r in range(ROWS)]
    main = torch.cuda.current_stream()
    side = _row_stream(x.device)
    hidden = [x[r : r + 1] for r in range(ROWS)]
    residual: list[torch.Tensor | None] = [None] * ROWS
    dense0, routed0, routed1 = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
    side.wait_stream(main)
    first = True
    try:
        for layer in model.layers.op_list:
            ctx._batch = rows[0]
            h, residual[0], pre = layer.dense(hidden[0], residual[0])
            dense0.record(main)
            if not first:
                main.wait_event(routed1)
            hidden[0] = layer.routed(h, pre)
            routed0.record(main)
            with torch.cuda.stream(side):
                ctx._batch = rows[1]
                side.wait_event(dense0)
                h, residual[1], pre = layer.dense(hidden[1], residual[1])
                side.wait_event(routed0)
                hidden[1] = layer.routed(h, pre)
                routed1.record(side)
            first = False
    finally:
        ctx._batch = batch
    main.wait_event(routed1)
    out = [model.norm.forward(hidden[r], residual[r])[0] for r in range(ROWS)]
    return torch.cat(out)


__all__ = ["can_stagger", "row_batch", "staggered_forward"]
