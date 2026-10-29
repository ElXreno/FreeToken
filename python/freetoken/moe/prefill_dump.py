"""Opt-in dump of every MoE layer's router input on prefill, the training data of next-layer expert predictors.

``FREETOKEN_PREFILL_ROUTER_DUMP=<dir>`` writes one ``chunk-NNNNNNNN.bin`` per prefill forward that reaches the last MoE
layer: magic ``FTPD``, u16 version, u16 num_layers, u32 rows, u16 hidden_size, then bfloat16 ``[num_layers, rows,
hidden_size]``. ``FREETOKEN_PREFILL_ROUTER_DUMP_EVERY=k`` keeps every k-th token of the forward (the phase rotates
between forwards). A chunk appears under its final name only once complete, so a reader may consume and delete chunks
while the engine runs. Prefill runs eagerly, so the copies are plain synchronous device-to-host reads.
"""

from __future__ import annotations

import os
import struct

import torch

MAGIC = b"FTPD"
VERSION = 1


def dump_dir() -> str | None:
    return os.environ.get("FREETOKEN_PREFILL_ROUTER_DUMP") or None


def dump_every() -> int:
    return max(1, int(os.environ.get("FREETOKEN_PREFILL_ROUTER_DUMP_EVERY", "1")))


class PrefillDump:
    def __init__(self, path: str, *, num_layers: int, hidden_size: int, every: int = 1):
        self.path = path
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.every = every
        self.forwards = 0
        self.chunks = 0
        self._rows: torch.Tensor | None = None
        self._layers: list[torch.Tensor | None] = [None] * num_layers
        os.makedirs(path, exist_ok=True)

    def record(self, layer_id: int, hidden: torch.Tensor) -> None:
        if layer_id >= self.num_layers:
            return
        if layer_id == 0:
            phase = self.forwards % self.every
            self.forwards += 1
            self._rows = torch.arange(phase, hidden.shape[0], self.every, device=hidden.device)
            self._layers = [None] * self.num_layers
        if self._rows is None or self._rows.numel() == 0:
            return
        self._layers[layer_id] = hidden.index_select(0, self._rows).to(torch.bfloat16).cpu()
        if layer_id == self.num_layers - 1:
            self._write()

    def _write(self) -> None:
        layers, self._layers, self._rows = self._layers, [None] * self.num_layers, None
        if any(t is None for t in layers):
            return
        data = torch.stack(layers)
        name = os.path.join(self.path, f"chunk-{self.chunks:08d}.bin")
        self.chunks += 1
        with open(name + ".tmp", "wb") as f:
            f.write(MAGIC + struct.pack("<HHIH", VERSION, self.num_layers, data.shape[1], self.hidden_size))
            f.write(data.view(torch.uint16).numpy().tobytes())
        os.replace(name + ".tmp", name)


def read_chunk(name: str) -> torch.Tensor:
    """``[num_layers, rows, hidden_size]`` bfloat16 of one dump chunk."""
    with open(name, "rb") as f:
        head = f.read(14)
        if head[:4] != MAGIC:
            raise ValueError(f"{name}: not a prefill router dump")
        version, layers, rows, hidden = struct.unpack("<HHIH", head[4:])
        if version != VERSION:
            raise ValueError(f"{name}: dump version {version}, expected {VERSION}")
        raw = torch.frombuffer(bytearray(f.read()), dtype=torch.uint16)
    return raw.view(torch.bfloat16).view(layers, rows, hidden)


__all__ = ["MAGIC", "VERSION", "PrefillDump", "dump_dir", "dump_every", "read_chunk"]
