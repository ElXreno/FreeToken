"""Hybrid ensure with LFU-decay eviction: hot experts outlive one-off ones, this step's hits are never victims,
and slots kept out of the pool stay out. Two layers of four experts share one four-slot pool."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cache(monkeypatch, tau):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe import offload_kernels
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setattr(offload_kernels, "_HYBRID_LFU_TAU", float(tau))
    cache = OffloadMoeCache(num_layers=2, num_experts=4, cache_size=4, device=torch.device("cuda"),
                            decode_target="hybrid", hybrid_max_fetch=1)
    cache.hybrid_fetch_fraction = [1.0, 1.0]
    return cache


def _route(cache, layer, *ids):
    t = torch.tensor(ids, dtype=torch.int32, device="cuda")
    cache.ensure_experts_hybrid(layer, t, torch.ones(len(ids), device="cuda"))
    torch.cuda.synchronize()


def _slot(cache, layer, e):
    return int(cache.slot_for_id[layer, e])


@cuda
@pytest.mark.parametrize("tau", [0, 1000])
def test_a_hot_expert_outlives_newer_one_off_experts_only_under_lfu(monkeypatch, tau):
    cache = _cache(monkeypatch, tau)
    for _ in range(6):
        _route(cache, 0, 0)
    for e in (1, 2, 3):
        _route(cache, 0, e)
    _route(cache, 1, 0)
    assert (_slot(cache, 0, 0) >= 0) == (tau > 0)
    assert (_slot(cache, 0, 1) >= 0) == (tau == 0)
    assert _slot(cache, 0, 2) >= 0 and _slot(cache, 0, 3) >= 0 and _slot(cache, 1, 0) >= 0


@cuda
def test_this_steps_hit_is_never_the_victim_even_when_hotter_slots_exist(monkeypatch):
    cache = _cache(monkeypatch, 1e6)
    for layer, e in ((1, 0), (1, 1), (0, 1)):
        for _ in range(4):
            _route(cache, layer, e)
    _route(cache, 0, 0)
    cold = _slot(cache, 0, 0)
    _route(cache, 0, 0, 2)
    assert _slot(cache, 0, 0) == cold
    assert _slot(cache, 0, 2) >= 0 and _slot(cache, 0, 2) != cold


@cuda
def test_an_unavailable_slot_is_never_a_victim(monkeypatch):
    from freetoken.moe.offload_cache import UNAVAILABLE

    cache = _cache(monkeypatch, 1000)
    cache.usage[3] = UNAVAILABLE
    for layer in (0, 1, 0, 1):
        for e in range(4):
            _route(cache, layer, e)
            assert int(cache.id_of_slot[3]) == -1
