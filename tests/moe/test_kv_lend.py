"""KV lending: borrowed MoE slots live in KV chunks no page below the boundary touches, and the page allocator only hands out pages below it."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_plan_backs_each_borrowed_slot_with_high_pages_of_one_slab():
    from freetoken.moe.kv_lend import plan_lend

    granule, page_bytes, pages, slabs = 4096, 256, 100, 6
    slab_bytes = (pages + 1) * page_bytes
    rows, base = [3000, 1024, 16], 5
    plan = plan_lend(
        slab_bytes=slab_bytes, num_slabs=slabs, page_bytes=page_bytes, usable_pages=pages,
        granule=granule, base_slots=base, row_bytes=rows, max_slots=1000,
    )
    assert plan.slots > 4
    assert list(plan.thresholds) == sorted(plan.thresholds, reverse=True)
    used = [c for ext in plan.ext_chunks for c in ext]
    assert len(used) == len(set(used)), "a KV chunk backs two places"
    for k, r in enumerate(rows):
        # every byte of a base slot stays in the bank's own memory
        assert plan.own_chunks[k] * granule >= base * r
        for j, t in enumerate(plan.thresholds):
            slot = base + j
            for v in range(slot * r // granule, ((slot + 1) * r - 1) // granule + 1):
                if v < plan.own_chunks[k]:
                    continue
                lo = plan.ext_chunks[k][v - plan.own_chunks[k]] * granule
                hi = lo + granule - 1
                assert lo // slab_bytes == hi // slab_bytes, "a chunk spans two slabs"
                first, last = (lo % slab_bytes) // page_bytes, (hi % slab_bytes) // page_bytes
                assert t <= first and last < pages, (k, j, t, first, last)


def _manager(thresholds, num_pages, margin):
    from freetoken.moe.kv_lend import KvLend, LendPlan
    from freetoken.scheduler.cache import CacheManager

    calls = []
    stub = SimpleNamespace(device=torch.device("cpu"), set_live_slots=lambda live, segment: calls.append(live))
    plan = LendPlan(granule=0, base_slots=0, own_chunks=(), ext_chunks=(), thresholds=tuple(thresholds))
    lend = KvLend(stub, plan, num_pages, margin)
    cm = CacheManager(num_pages, 1, torch.zeros(2, num_pages, dtype=torch.int32), "radix", kv_lend=lend)
    return cm, lend, calls


def test_allocator_stays_below_the_boundary_and_evicts_the_tree_first():
    num_pages = 64
    cm, lend, calls = _manager([60 - 4 * (j // 2) for j in range(32)], num_pages, margin=4)
    assert (lend.slots, lend.boundary) == (30, 4)
    assert cm.available_size == num_pages and cm.lent_pages == 60
    cm.check_integrity()

    held = cm._allocate(20)
    assert int(held.max()) < lend.boundary == 24 and lend.slots == 20
    tree = cm._allocate(8)
    assert int(tree.max()) < lend.boundary == 32
    cm.prefix_cache.insert_prefix(torch.arange(8, dtype=torch.int32), tree)

    more = cm._allocate(10)
    assert lend.boundary == 32, "the evictable tree goes before borrowed slots"
    assert int(more.max()) < 32 and cm.prefix_cache.size_info.total_size < 8

    fragment = more[-1:]
    cm._free(held)
    cm._free(more[:-1])
    cm._free(cm._evict(cm.prefix_cache.size_info.evictable_size))
    cm.maybe_lend()
    assert lend.boundary == 32 or int(fragment) < lend.boundary - lend.margin, "lent past a held page"
    cm._free(fragment)
    cm.maybe_lend()
    assert (lend.slots, lend.boundary) == (30, 4)
    assert sorted(cm.free_slots.tolist()) == [0, 1, 2, 3]
    assert calls == [30, 20, 16, 30]
    cm.check_integrity()


def _tiered_manager(tmp_path, num_pages=64):
    from freetoken.kvcache.host_tier import HostTier
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    from freetoken.moe.kv_lend import KvLend, LendPlan
    from freetoken.scheduler.cache import CacheManager

    stub = SimpleNamespace(device=torch.device("cpu"), set_live_slots=lambda live, segment: None)
    plan = LendPlan(granule=0, base_slots=0, own_chunks=(), ext_chunks=(), thresholds=tuple(60 - 4 * (j // 2) for j in range(32)))
    lend = KvLend(stub, plan, num_pages, 4)
    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0,), num_key_heads=2, num_value_heads=4,
        key_head_dim=16, value_head_dim=16, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=7, dtype=torch.bfloat16, device=torch.device("cpu"), tp_size=1)
    kv = SimpleNamespace(_kv_buffer=torch.zeros((2, 1, num_pages + 1, 1, 2, 8), dtype=torch.uint8), device=torch.device("cpu"), dtype=torch.uint8)
    tier = HostTier(str(tmp_path), 64 << 20, kv, pool, {"model": "fake"})
    cm = CacheManager(num_pages, 1, torch.zeros(2, num_pages, dtype=torch.int32), "hybrid_radix", linear_state_pool=pool, host_tier=tier, kv_lend=lend)

    def commit(first, kv_slots):
        _, node, _ = cm.prefix_cache.insert(torch.arange(first, first + len(kv_slots), dtype=torch.int32), kv_slots)
        node.host_kv = tier.put_kv(node.value)
        node.host_snap = tier.put_snap(1)
        return node

    return cm, lend, commit


def test_idle_trim_drops_a_cold_branch_off_the_top_pages(tmp_path):
    cm, lend, commit = _tiered_manager(tmp_path)
    pages = cm._allocate(48)
    assert lend.boundary > 48
    cold = commit(100, pages[8:])
    # the newest path is two nodes stamped by one insert, like a finished request's chunk commits
    hot = [commit(200, pages[:4]), commit(200, pages[:8])]
    assert hot[0].timestamp == hot[1].timestamp
    cm.maybe_lend()
    assert lend.boundary > 48, "a held page sits near the top"
    cm.maybe_lend(idle=True)
    assert not cold.resident and cold.host_kv is not None and all(n.resident for n in hot)
    assert lend.boundary == 12 and int(pages[:8].max()) < lend.boundary - lend.margin
    cm.check_integrity()


def test_page_pressure_evicts_the_lowest_stale_branch_first(tmp_path):
    cm, lend, commit = _tiered_manager(tmp_path)
    pages = cm._allocate(48)
    high = commit(100, pages[24:])
    low = commit(300, pages[8:24])
    hot = [commit(200, pages[:4]), commit(200, pages[:8])]
    boundary = lend.boundary
    taken = cm._allocate(len(cm.free_slots) + 8)
    assert not low.resident and high.resident and all(n.resident for n in hot), "LRU would take the older, higher branch"
    cm._allocate(26)
    assert not high.resident and not hot[0].resident and hot[1].resident, "the newest path goes last, lowest pages first"
    assert lend.boundary == boundary and int(taken.max()) < 24, "the allocation took the free pages under the boundary"


def _lend_setup(monkeypatch, target, I=512, linear_pool=None):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.moe.kv_lend import attach_kv_lend
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    monkeypatch.setenv("FREETOKEN_KV_LEND_MARGIN", "1024")
    torch.manual_seed(0)
    L, E, H = 3, 4, 512
    sources = {
        "gate_up": [torch.randn(E, 2 * I, H, dtype=torch.bfloat16).pin_memory() for _ in range(L)],
        "down": [torch.randn(E, H, I, dtype=torch.bfloat16).pin_memory() for _ in range(L)],
    }
    cache = OffloadMoeCache(num_layers=L, num_experts=E, cache_size=2 * E, device=torch.device("cuda"), decode_target=target)
    cache.set_bank_sources(sources)
    kv = MHAKVCache(num_kv_heads=2, num_layers=2, head_dim=256, num_pages=16384, page_size=1, dtype=torch.bfloat16, device=torch.device("cuda"))
    lend = attach_kv_lend(cache, kv, usable_pages=16383, page_size=1, linear_pool=linear_pool)
    assert lend is not None
    return cache, kv, lend


def _route(cache, layer, experts):
    ids = torch.tensor(experts, dtype=torch.int32, device="cuda")
    if cache.decode_target == "hybrid":
        cache.hybrid_max_fetch = len(experts)
        cache.ensure_experts_hybrid(layer, ids, torch.ones(len(experts), device="cuda"))
    else:
        cache.ensure_experts(layer, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    return ids.tolist()


def _rows_match(cache, layer, expert):
    s = int(cache.slot_for_id[layer, expert])
    return s >= 0 and all(torch.equal(view[s].cpu(), src[layer][expert]) for (src, _), view in zip(cache.banks, cache.pool_bank_views(layer)))


@cuda
@pytest.mark.parametrize("target", ["gpu", "hybrid"])
def test_taking_kv_pages_back_never_mixes_kv_and_expert_bytes(monkeypatch, target):
    from freetoken.moe.offload_cache import UNAVAILABLE

    cache, kv, lend = _lend_setup(monkeypatch, target)
    base, experts = cache.lend_base, [(layer, e) for layer in range(cache.num_layers) for e in range(cache.num_experts)]
    live0, b0 = lend.slots, lend.boundary
    assert live0 >= len(experts) and cache.cache_size == base + lend.capacity
    assert torch.all(cache.usage[base + live0 :] == UNAVAILABLE)

    # base slots busy, so the experts land in borrowed slots
    cache.usage[:base] = 1 << 40
    cache.step.fill_(1 << 40)
    for layer in range(cache.num_layers):
        slots = _route(cache, layer, list(range(cache.num_experts)))
        assert all(base <= s < base + live0 for s in slots), slots
    assert all(_rows_match(cache, layer, e) for layer, e in experts)
    kv._kv_buffer[:, :, :b0] = 3.0
    torch.cuda.synchronize()
    assert all(_rows_match(cache, layer, e) for layer, e in experts)

    b1 = lend.grow_kv(lend.boundary_for(10) - b0 - lend.margin)
    live1 = lend.slots
    assert live1 == 10 and b1 > b0
    gone = [(layer, e) for layer, e in experts if int(cache.slot_for_id[layer, e]) < 0]
    assert len(gone) == len(experts) - min(len(experts), live1)
    assert torch.all(cache.id_of_slot[base + live1 :] == -1)
    assert torch.all(cache.usage[base + live1 :] == UNAVAILABLE)
    kv._kv_buffer[:, :, b0:b1] = 5.0
    torch.cuda.synchronize()
    assert all(_rows_match(cache, layer, e) for layer, e in experts if (layer, e) not in gone)

    for layer in range(cache.num_layers):
        slots = _route(cache, layer, list(range(cache.num_experts)))
        assert all(s < base + live1 for s in slots), slots
    assert all(_rows_match(cache, layer, e) for layer, e in experts)
    assert torch.all(kv._kv_buffer[:, :, :b0] == 3.0) and torch.all(kv._kv_buffer[:, :, b0:b1] == 5.0)

    assert lend.shrink_kv(-1) == b0 and lend.slots == live0
    assert torch.all(cache.usage[base + live1 : base + live0] == 0)
    cache.reset()
    assert torch.all(cache.usage[base + live0 :] == UNAVAILABLE) and torch.all(cache.usage[: base + live0] == 0)


@cuda
def test_idle_gdn_state_slots_back_experts_until_a_request_takes_them(monkeypatch):
    from freetoken.kvcache.linear_state_pool import LinearStatePool
    from freetoken.models.config import LinearGatedDeltaGroupConfig
    from freetoken.moe.offload_cache import UNAVAILABLE

    group = LinearGatedDeltaGroupConfig(
        name="linear", layer_ids=(0, 1), num_key_heads=4, num_value_heads=32,
        key_head_dim=128, value_head_dim=128, conv_kernel_dim=4, output_gate="silu",
    )
    pool = LinearStatePool(group=group, num_slots=7, dtype=torch.bfloat16, device=torch.device("cuda"), tp_size=1)
    # 768 KiB and 384 KiB rows: the state segment starts past a gap of never-usable slots
    cache, _, _ = _lend_setup(monkeypatch, "hybrid", I=384, linear_pool=pool)
    gdn = pool.lend
    lo, slots, live = cache.lend_segments[1]
    kv_end = cache.lend_segments[0][0] + cache.lend_segments[0][1]
    assert lo > kv_end and live == slots == gdn.capacity and gdn.boundary == 1 and pool.num_free_slots == 6
    assert torch.all(cache.usage[kv_end:lo] == UNAVAILABLE)
    experts = [(layer, e) for layer in range(cache.num_layers) for e in range(cache.num_experts)]

    busy = cache.usage[:lo] != UNAVAILABLE
    cache.usage[:lo][busy] = 1 << 40
    cache.step.fill_(1 << 40)
    for layer in range(cache.num_layers):
        assert all(lo <= s < lo + live for s in _route(cache, layer, list(range(cache.num_experts))))
    assert all(_rows_match(cache, layer, e) for layer, e in experts)

    assert pool.alloc(3) == [1, 2, 3] and gdn.boundary == 4
    live1 = cache.lend_segments[1][2]
    assert 0 < live1 < len(experts) and torch.all(cache.usage[lo + live1 : lo + slots] == UNAVAILABLE)
    gone = [(layer, e) for layer, e in experts if int(cache.slot_for_id[layer, e]) < 0]
    assert len(gone) == len(experts) - live1
    pool.recurrent_states[:, 1:4] = 7.0
    torch.cuda.synchronize()
    assert all(_rows_match(cache, layer, e) for layer, e in experts if (layer, e) not in gone)

    for layer in range(cache.num_layers):
        slots_now = _route(cache, layer, list(range(cache.num_experts)))
        assert all(s < cache.lend_base + cache.lend_segments[0][2] or lo <= s < lo + live1 for s in slots_now), slots_now
    assert all(_rows_match(cache, layer, e) for layer, e in experts)
    assert torch.all(pool.recurrent_states[:, 1:4] == 7.0)

    pool.free([3, 1, 2])
    assert gdn.boundary == 1 and pool.num_free_slots == 6 and cache.lend_segments[1][2] == slots
    assert torch.all(cache.usage[lo + live1 : lo + slots] == 0)
