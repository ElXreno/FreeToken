"""With a routing draft head the NVFP4 expert reader keeps the parallel/serial auto-pick of the plain path."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from freetoken.layers.quantization import QuantKind
from freetoken.models import nvfp4_banks
from freetoken.models import weight as model_weight
from freetoken.models.qwen3_5_moe import weight as qw


@pytest.fixture
def seen(monkeypatch):
    got = []
    monkeypatch.setattr(nvfp4_banks, "iter_nvfp4_expert_pieces", lambda *a, parallel, **k: got.append(parallel) or iter(()))
    monkeypatch.setattr(qw, "nvfp4_expert_spec", lambda *a: None)
    monkeypatch.setattr(model_weight, "experts_scattered", lambda path: True)
    return got


CONFIG = SimpleNamespace(num_mtp_moe_layers=1, mtp_bank_layer=2, num_layers=2, first_k_dense_replace=0)


@pytest.mark.parametrize(("asked", "used"), [(None, True), (False, False), (True, True)])
def test_auto_resolves_through_the_scatter_check(seen, asked, used):
    list(qw.iter_expert_pieces("/m", CONFIG, QuantKind.NVFP4, layers=[0, 1], parallel=asked))
    assert seen == [used]
