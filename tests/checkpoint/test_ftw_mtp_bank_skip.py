"""An FTW converted with the draft head's expert bank loads without MTP: the trailing bank layer is left out."""

from __future__ import annotations

import pytest
import torch
from freetoken.checkpoint.ftw import FTWReader, FTWWriter, _split_bank_entries, layer_bank_entry_name

L, E = 3, 4


def _write(tmp_path, *, mtp_weights: bool):
    writer = FTWWriter(str(tmp_path))
    writer.add_tensor("model.norm.weight", torch.ones(8))
    if mtp_weights:
        writer.add_tensor("mtp.fc.weight", torch.ones(8, 8))
    writer.add_tensor("gate_up_alpha", torch.arange((L + 1) * E, dtype=torch.float32), kind="experts_bank")
    for layer in range(L + 1):
        writer.add_tensor(layer_bank_entry_name("gate_up_packed", layer), torch.full((E, 16), layer, dtype=torch.int32),
                          kind="experts_bank")
    writer.finalize({"quant_format": "nvfp4_b12x", "expert_bank_num_layers": L + 1})
    return FTWReader(str(tmp_path))


def test_the_draft_heads_bank_layer_is_dropped_when_mtp_is_off(tmp_path):
    reader = _write(tmp_path, mtp_weights=True)
    bank_entries, alphas, flat, per_layer = _split_bank_entries(reader, str(tmp_path), L)
    assert flat == []
    assert sorted(per_layer["gate_up_packed"]) == list(range(L))
    (alpha,) = alphas
    assert alpha["shape"] == [L * E] and alpha["nbytes"] == L * E * 4
    assert len(bank_entries) == 1 + L


def test_the_full_bank_loads_with_mtp_on(tmp_path):
    reader = _write(tmp_path, mtp_weights=True)
    _, alphas, _, per_layer = _split_bank_entries(reader, str(tmp_path), L + 1)
    assert sorted(per_layer["gate_up_packed"]) == list(range(L + 1))
    assert alphas[0]["shape"] == [(L + 1) * E]


def test_an_extra_layer_without_a_draft_head_still_fails(tmp_path):
    reader = _write(tmp_path, mtp_weights=False)
    with pytest.raises(RuntimeError, match="does not match its config"):
        _split_bank_entries(reader, str(tmp_path), L)
