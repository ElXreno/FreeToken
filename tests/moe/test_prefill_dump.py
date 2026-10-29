"""The prefill router-input dump writes one chunk per forward with every k-th token of each MoE layer."""

from __future__ import annotations

import torch
from freetoken.moe.prefill_dump import PrefillDump, read_chunk

L, H = 3, 8


def _forward(dump, rows, base):
    inputs = [torch.arange(rows * H, dtype=torch.float32).view(rows, H) + 1000 * layer + base for layer in range(L)]
    for layer, x in enumerate(inputs):
        dump.record(layer, x)
    dump.record(L, inputs[0])
    return inputs


def test_each_forward_lands_in_its_own_complete_chunk(tmp_path):
    dump = PrefillDump(str(tmp_path), num_layers=L, hidden_size=H, every=1)
    first = _forward(dump, 5, 0)
    second = _forward(dump, 3, 7)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["chunk-00000000.bin", "chunk-00000001.bin"]
    for name, inputs in (("chunk-00000000.bin", first), ("chunk-00000001.bin", second)):
        got = read_chunk(str(tmp_path / name))
        assert got.dtype == torch.bfloat16 and got.shape == (L, inputs[0].shape[0], H)
        assert torch.equal(got, torch.stack(inputs).to(torch.bfloat16))


def test_subsampling_rotates_its_phase_between_forwards(tmp_path):
    dump = PrefillDump(str(tmp_path), num_layers=L, hidden_size=H, every=3)
    first = _forward(dump, 7, 0)
    second = _forward(dump, 7, 0)
    a = read_chunk(str(tmp_path / "chunk-00000000.bin"))
    b = read_chunk(str(tmp_path / "chunk-00000001.bin"))
    assert torch.equal(a, torch.stack(first)[:, 0::3].to(torch.bfloat16))
    assert torch.equal(b, torch.stack(second)[:, 1::3].to(torch.bfloat16))


def test_a_forward_that_stops_before_the_last_layer_writes_nothing(tmp_path):
    dump = PrefillDump(str(tmp_path), num_layers=L, hidden_size=H, every=1)
    dump.record(0, torch.zeros(4, H))
    dump.record(1, torch.zeros(4, H))
    _forward(dump, 2, 0)
    assert [p.name for p in tmp_path.iterdir()] == ["chunk-00000000.bin"]
    assert read_chunk(str(tmp_path / "chunk-00000000.bin")).shape == (L, 2, H)
