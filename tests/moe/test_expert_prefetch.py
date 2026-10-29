"""Next-layer expert prefetch: the best-scored cold experts land in LRU slots, the tables point at them, and the rows match the banks."""

from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, D, SLOTS = 3, 16, 32, 24


def _setup():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.expert_prefetch import ExpertPrefetch
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    torch.manual_seed(0)
    sources = {
        "gate_up": [torch.randn(E, 64, 32, dtype=torch.bfloat16).pin_memory() for _ in range(L)],
        "down": [torch.randn(E, 32, 32, dtype=torch.bfloat16).pin_memory() for _ in range(L)],
    }
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=SLOTS, device=torch.device("cuda"))
    cache.set_bank_sources(sources)
    # expert e of the next layer scores e: the predictor ranks 15, 14, 13, ...
    rank = torch.arange(E, dtype=torch.bfloat16, device="cuda")[:, None].expand(E, D).contiguous() / D
    predictors = {0: rank, 1: rank}
    return cache, ExpertPrefetch(cache, predictors, budget=3, topn=6)


def _resident(cache, layer, e):
    return int(cache.slot_for_id[layer, e])


def _rows_match(cache, layer, e):
    s = _resident(cache, layer, e)
    return s >= 0 and all(torch.equal(view[s].cpu(), src[layer][e]) for (src, _), view in zip(cache.banks, cache.pool_bank_views(layer)))


@cuda
def test_the_best_cold_experts_are_fetched_and_resident_ones_skipped():
    cache, pf = _setup()
    cache.ensure_experts(0, torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    cache.copy_missing()
    cache.ensure_experts(1, torch.tensor([15, 13], dtype=torch.int32, device="cuda"))
    cache.copy_missing()
    cache.ensure_experts(0, torch.tensor([0, 1], dtype=torch.int32, device="cuda"))
    cache.copy_missing()
    pf.issue(0, torch.ones(1, D, dtype=torch.bfloat16, device="cuda"))
    pf.wait(1)
    torch.cuda.synchronize()
    assert int(pf.num[1]) == 3
    assert sorted(pf.src[1].tolist()) == [11, 12, 14]
    for e in (11, 12, 14, 15, 13):
        assert _rows_match(cache, 1, e), e
    for e in (0, 1):
        assert _rows_match(cache, 0, e), e
    slots = [_resident(cache, layer, e) for layer, e in ((0, 0), (0, 1), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15))]
    assert len(set(slots)) == len(slots)
    ids = torch.tensor([14, 12], dtype=torch.int32, device="cuda")
    cache.ensure_experts(1, ids)
    assert int(cache.num_indices) == 0


@cuda
def test_a_fresh_cache_gives_every_prefetched_expert_its_own_slot():
    cache, pf = _setup()
    pf.issue(0, torch.ones(1, D, dtype=torch.bfloat16, device="cuda"))
    pf.wait(1)
    torch.cuda.synchronize()
    assert sorted(pf.src[1].tolist()) == [13, 14, 15]
    assert len(set(pf.evict[1].tolist())) == 3
    for e in (13, 14, 15):
        assert _rows_match(cache, 1, e), e


@cuda
def test_prefetch_captures_into_a_graph_and_follows_the_live_input():
    cache, pf = _setup()
    # a decode step has run before: the prefetch stamps lie above the never-used slots
    cache.ensure_experts(2, torch.tensor([5], dtype=torch.int32, device="cuda"))
    cache.copy_missing()
    hidden = torch.ones(1, D, dtype=torch.bfloat16, device="cuda")
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        pf.issue(0, hidden)
        pf.wait(1)
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pf.issue(0, hidden)
        pf.wait(1)
    hidden.fill_(-1.0)
    graph.replay()
    torch.cuda.synchronize()
    # a negative input reverses the ranking: experts 0, 1, 2 are now the best cold ones
    assert sorted(pf.src[1].tolist()) == [0, 1, 2]
    for e in (0, 1, 2, 13, 14, 15):
        assert _rows_match(cache, 1, e), e


def test_routers_are_found_through_the_model_layer_list():
    from types import SimpleNamespace

    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.base import OPList
    from freetoken.layers.linear import LinearReplicated
    from freetoken.moe.expert_prefetch import next_layer_routers

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    decs = []
    for layer_id in range(3):
        mlp = SimpleNamespace(experts=SimpleNamespace(layer_id=layer_id), gate=LinearReplicated(D, E, has_bias=False))
        decs.append(SimpleNamespace(mlp=mlp))
    model = SimpleNamespace(model=SimpleNamespace(layers=OPList(decs)))
    routers = next_layer_routers(model)
    assert sorted(routers) == [0, 1, 2]
    assert all(w.shape == (E, D) for w in routers.values())


@cuda
def test_early_mode_installs_low_priority_slots_and_waits_before_the_demand_copy(monkeypatch):
    monkeypatch.setenv("FREETOKEN_EXPERT_PREFETCH_MODE", "early")
    cache, pf = _setup()
    assert pf.early
    cache.ensure_experts(2, torch.tensor([5], dtype=torch.int32, device="cuda"))
    cache.copy_missing()
    pf.after_demand(0, torch.ones(1, D, dtype=torch.bfloat16, device="cuda"))
    assert pf.pending[1]
    pf.before_demand(1)
    assert not pf.pending[1]
    torch.cuda.synchronize()
    id_of_slot, usage = cache.pool_state(1)
    for e in (13, 14, 15):
        s = _resident(cache, 1, e)
        assert _rows_match(cache, 1, e), e
        assert int(usage[s]) == 0, (e, int(usage[s]))


@cuda
def test_a_single_row_scores_by_its_logits_and_rows_by_their_summed_softmax():
    cache, pf = _setup()
    one = torch.randn(1, D, dtype=torch.bfloat16, device="cuda")
    pf.score(0, one)
    torch.testing.assert_close(pf.scores[0], torch.nn.functional.linear(one, pf.predictors[0])[0], rtol=0, atol=0)
    two = torch.randn(2, D, dtype=torch.bfloat16, device="cuda")
    pf.score(0, two)
    want = torch.softmax(torch.nn.functional.linear(two, pf.predictors[0]).float(), dim=-1).sum(0)
    torch.testing.assert_close(pf.scores[0], want)
