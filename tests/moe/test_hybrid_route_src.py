"""Hybrid ensure with route sources: a row whose source is another row routes to that row's experts, so its own
experts are never fetched, counted as misses or sent to the CPU; identity sources route every row as given."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cache(fraction):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    cache = OffloadMoeCache(num_layers=1, num_experts=8, cache_size=8, device=torch.device("cuda"),
                            decode_target="hybrid", hybrid_max_fetch=0)
    cache.hybrid_fetch_fraction = [fraction]
    return cache


def _ensure(cache, src):
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda")
    route_src = torch.tensor(src, dtype=torch.int32, device="cuda")
    routes = cache.ensure_experts_hybrid(0, ids, torch.ones(2, 2, device="cuda"), route_src=route_src)
    torch.cuda.synchronize()
    return routes


@cuda
def test_a_row_routed_to_another_rows_source_fetches_only_the_source_experts():
    cache = _cache(1.0)
    routes = _ensure(cache, [0, 0])
    assert int(cache.num_missing_full) == 2
    assert all(int(cache.slot_for_id[0, e]) >= 0 for e in (0, 1))
    assert all(int(cache.slot_for_id[0, e]) == -1 for e in (2, 3))
    assert torch.equal(routes.gpu_slots[1], routes.gpu_slots[0])


@cuda
def test_a_row_routed_to_another_row_sends_the_source_experts_to_the_cpu():
    cache = _cache(0.0)
    routes = _ensure(cache, [0, 0])
    assert routes.cpu_ids.tolist() == [[0, 1], [0, 1]]


@cuda
def test_identity_sources_route_every_row_as_given():
    cache = _cache(1.0)
    _ensure(cache, [0, 1])
    assert int(cache.num_missing_full) == 4
    assert all(int(cache.slot_for_id[0, e]) >= 0 for e in range(4))
