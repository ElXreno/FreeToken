"""Opt-in decode routing trace for offline expert-cache simulation.

``FREETOKEN_ROUTING_TRACE=<path>`` makes every decode step record each MoE layer's top-k expert ids and router
weights into a device ring that the decode CUDA graph writes on replay; the host drains the ring between steps
into an append-only file. Unset, nothing is allocated and the decode path is unchanged.

File: a header, then records, all little-endian.

* header: ``b"FTRT"``, u32 version, u32 num_layers, u32 num_experts, u32 top_k
* ``STEP`` (1): u16 rows, then int16 ids ``[num_layers, rows, top_k]`` (-1: the layer did not route this step)
  and float16 weights of the same shape
* ``PREFILL`` (2): a prefill chunk began; it overwrites every pool's slots below its stable index
* ``RESET`` (3): the cache was emptied
* ``START`` (4): u32 npools, per pool (u32 size, u32 stable_from); u32 nseeds, per seeded layer
  (u16 layer, u16 share, u32 first slot): the cache layout and the seeds serving starts from
* ``LOST`` (5): u32 steps the ring overwrote before a drain
* ``HIDDEN`` (6, version 2): u16 num_layers, u16 hidden_size, then bfloat16 ``[num_layers, hidden_size]``: the router
  input of every MoE layer for the first row of the ``STEP`` just before it (zeros where the layer did not route).
  ``FREETOKEN_ROUTING_TRACE_HIDDEN=1`` adds it for offline next-layer expert predictors; the header then says version 2.
"""

from __future__ import annotations

import os
import struct

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

MAGIC = b"FTRT"
VERSION = 1
VERSION_HIDDEN = 2
STEP, PREFILL, RESET, START, LOST, HIDDEN = 1, 2, 3, 4, 5, 6
_DRAIN_EVERY = 64
_HIDDEN_RING = 256


def trace_path() -> str | None:
    return os.environ.get("FREETOKEN_ROUTING_TRACE") or None


def trace_hidden() -> bool:
    return os.environ.get("FREETOKEN_ROUTING_TRACE_HIDDEN", "").strip().lower() in ("1", "true", "yes", "on")


class RoutingTrace:
    """Device ring of per-step routing plus its host-side drain into the trace file."""

    def __init__(self, path: str, *, num_layers: int, num_experts: int, top_k: int, max_rows: int,
                 device: torch.device, ring: int = 1024, hidden_size: int | None = None):
        self.path = path
        self.num_layers, self.num_experts, self.top_k = num_layers, num_experts, top_k
        self.max_rows, self.ring = max_rows, ring
        self.device = device
        self.ids = torch.full((ring, num_layers, max_rows, top_k), -1, dtype=torch.int16, device=device)
        self.weights = torch.zeros((ring, num_layers, max_rows, top_k), dtype=torch.float16, device=device)
        self.rows = torch.zeros((ring,), dtype=torch.int32, device=device)
        self.pos = torch.zeros((1,), dtype=torch.int64, device=device)
        self.hidden_size = hidden_size
        self.hidden = None
        if hidden_size is not None:
            self.hidden = torch.zeros((_HIDDEN_RING, num_layers, hidden_size), dtype=torch.bfloat16, device=device)
        self._done = 0
        self._ticks = 0
        self._file = None
        self._steps = 0

    def record(self, layer_id: int, topk_ids: torch.Tensor, topk_weights: torch.Tensor,
               hidden: torch.Tensor | None = None) -> None:
        """Called by every MoE layer's decode before it touches ``topk_ids`` or ``hidden``; graph-capturable (fixed shapes, device index)."""
        n = topk_ids.shape[0]
        assert n <= self.max_rows, (n, self.max_rows)
        if layer_id == 0:
            self.pos.add_(1)
        idx = torch.remainder(self.pos, self.ring)
        self.ids[:, layer_id, :n].index_copy_(0, idx, topk_ids.reshape(1, n, self.top_k).to(torch.int16))
        self.weights[:, layer_id, :n].index_copy_(0, idx, topk_weights.reshape(1, n, self.top_k).to(torch.float16))
        if layer_id == 0:
            self.rows.index_fill_(0, idx, n)
        if self.hidden is not None and hidden is not None:
            hidx = torch.remainder(self.pos, _HIDDEN_RING)
            self.hidden[:, layer_id].index_copy_(0, hidx, hidden[:1].reshape(1, self.hidden_size).to(torch.bfloat16))

    def _open(self):
        if self._file is None:
            self._file = open(self.path, "ab")  # noqa: SIM115 -- held for the process lifetime
            if self._file.tell() == 0:
                version = VERSION if self.hidden is None else VERSION_HIDDEN
                self._file.write(MAGIC + struct.pack("<4I", version, self.num_layers, self.num_experts, self.top_k))
        return self._file

    def start(self, pools: list[tuple[int, int]], seeds: list[tuple[int, int, int]]) -> None:
        """Serving starts from this cache layout: drop what capture and warmup wrote, then mark the layout."""
        self.ids.fill_(-1)
        self.rows.zero_()
        self.pos.zero_()
        if self.hidden is not None:
            self.hidden.zero_()
        self._done = 0
        f = self._open()
        f.write(struct.pack("<BI", START, len(pools)))
        for size, stable in pools:
            f.write(struct.pack("<2I", size, stable))
        f.write(struct.pack("<I", len(seeds)))
        for layer, share, first in seeds:
            f.write(struct.pack("<2HI", layer, share, first))
        f.flush()

    def marker(self, kind: int) -> None:
        """Flush the steps before an event so the event lands in order."""
        if self._file is None:
            return
        self.drain()
        self._file.write(struct.pack("<B", kind))
        self._file.flush()

    def tick(self, stream: torch.cuda.Stream | None) -> None:
        """One processed decode batch; drains every ``_DRAIN_EVERY`` on the engine stream, behind the replays in flight."""
        self._ticks += 1
        if self._ticks % _DRAIN_EVERY == 0:
            self.flush(stream)

    def flush(self, stream: torch.cuda.Stream | None) -> None:
        if self._file is None:
            return
        if stream is None:
            self.drain()
            return
        with torch.cuda.stream(stream):
            self.drain()

    def drain(self) -> None:
        if self._file is None:
            return
        pos = int(self.pos.item())
        new = pos - self._done
        if new <= 0:
            return
        f = self._file
        lost = max(0, new - self.ring)
        if lost:
            f.write(struct.pack("<BI", LOST, lost))
            logger.warning(f"routing trace: the ring overwrote {lost} decode steps before a drain")
            self._done += lost
        steps = range(self._done + 1, pos + 1)
        idx = torch.tensor([s % self.ring for s in steps], dtype=torch.long, device=self.device)
        ids = self.ids.index_select(0, idx).cpu()
        weights = self.weights.index_select(0, idx).cpu()
        rows = self.rows.index_select(0, idx).cpu().tolist()
        self.ids.index_fill_(0, idx, -1)
        hidden = None
        if self.hidden is not None:
            kept = [s for s in steps if pos - s < _HIDDEN_RING]
            hidx = torch.tensor([s % _HIDDEN_RING for s in kept], dtype=torch.long, device=self.device)
            hidden = dict(zip(kept, self.hidden.index_select(0, hidx).view(torch.int16).cpu().numpy(), strict=True))
            self.hidden.index_fill_(0, hidx, 0)
        for i, n in enumerate(rows):
            f.write(struct.pack("<BH", STEP, n))
            f.write(ids[i, :, :n].contiguous().numpy().tobytes())
            f.write(weights[i, :, :n].contiguous().numpy().tobytes())
            if hidden is not None and (h := hidden.get(steps[i])) is not None:
                f.write(struct.pack("<BHH", HIDDEN, self.num_layers, self.hidden_size))
                f.write(h.tobytes())
        f.flush()
        self._steps += len(rows)
        self._done = pos

    def close(self) -> None:
        if self._file is not None:
            self.drain()
            self._file.close()
            self._file = None
            logger.info(f"routing trace: {self._steps} decode steps in {self.path}")


__all__ = ["RoutingTrace", "trace_path", "trace_hidden", "MAGIC", "VERSION", "VERSION_HIDDEN", "STEP", "PREFILL", "RESET", "START", "LOST", "HIDDEN"]
