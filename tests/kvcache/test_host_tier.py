"""Host tier of the hybrid radix cache on CPU tensors: arena allocator, KV/snapshot round
trips, tree export/import, VRAM de-residency + promotion, arena eviction, adoption."""
from __future__ import annotations

import os
import tempfile

import pytest
import torch

from freetoken.kvcache.host_tier import ALIGN, HostArena, HostTier
from freetoken.kvcache.tiered_hybrid_cache import TieredHybridRadixCache

DEVICE = torch.device("cpu")


class FakeKVPool:
    def __init__(self, layers=2, pages=1024, page_size=1, heads=2, head_dim=8, dtype=torch.uint8):
        self._kv_buffer = torch.zeros((2, layers, pages, page_size, heads, head_dim), dtype=dtype)
        self.device = DEVICE
        self.dtype = dtype

    def raw(self) -> torch.Tensor:
        two, layers, pages, page_size, heads, _ = self._kv_buffer.shape
        return self._kv_buffer.view(torch.uint8).view(two, layers, pages * page_size, heads, -1)

    def rows(self, slots: torch.Tensor) -> torch.Tensor:
        return self.raw()[:, :, slots.long()].clone()

    def randomize(self, slots: torch.Tensor) -> None:
        raw = self.raw()
        two, layers, _, heads, width = raw.shape
        raw[:, :, slots.long()] = torch.randint(0, 255, (two, layers, len(slots), heads, width), dtype=torch.uint8)


class FakeLinearPool:
    def __init__(self, n_layers=3, slots=8):
        self.conv_states = torch.zeros((n_layers, slots, 6, 3), dtype=torch.bfloat16)
        self.recurrent_states = torch.zeros((n_layers, slots, 2, 4, 4), dtype=torch.float32)
        self.slot_states: dict[str, torch.Tensor] = {}

    def randomize(self, slot: int) -> None:
        self.conv_states[:, slot] = torch.randn_like(self.conv_states[:, slot])
        self.recurrent_states[:, slot] = torch.randn_like(self.recurrent_states[:, slot])


def make_tier(directory: str, capacity: int = 64 << 20, key: dict | None = None, kv=None, lin=None):
    kv = kv or FakeKVPool()
    lin = lin or FakeLinearPool()
    return HostTier(directory, capacity, kv, lin, key or {"model": "fake"}), kv, lin


class Alloc:
    """Hands out fresh token slots like CacheManager._allocate_tokens would."""

    def __init__(self, start: int):
        self.next = start

    def __call__(self, n: int) -> torch.Tensor:
        out = torch.arange(self.next, self.next + n, dtype=torch.int32)
        self.next += n
        return out


def ids(n: int, base: int = 1000) -> torch.Tensor:
    return torch.arange(base, base + n, dtype=torch.int32)


def commit(cache: TieredHybridRadixCache, tier: HostTier, tokens: torch.Tensor, kv_slots: torch.Tensor, slot: int):
    prefix_len, node, dups = cache.insert(tokens, kv_slots)
    if node.host_kv is None and node.resident:
        node.host_kv = tier.put_kv(node.value)
    if node.host_snap is None:
        node.host_snap = tier.put_snap(slot)
    return prefix_len, node, dups


def test_arena_alloc_free_coalesce():
    with tempfile.TemporaryDirectory() as d:
        arena = HostArena(os.path.join(d, "a.bin"), 16 * ALIGN)
        a, b, c = arena.alloc(ALIGN), arena.alloc(2 * ALIGN), arena.alloc(ALIGN)
        assert (a.offset, b.offset, c.offset) == (0, ALIGN, 3 * ALIGN)
        assert arena.used == 4 * ALIGN
        arena.free(b)
        arena.free(a)
        assert arena.free_extents[0] == [0, 3 * ALIGN]
        arena.free(c)
        assert arena.free_extents == [[0, 16 * ALIGN]]
        assert arena.used == 0
        assert arena.alloc(17 * ALIGN) is None
        small = arena.alloc(10)
        assert small.nbytes == 10 and arena.used == ALIGN
        arena.close()


def test_kv_roundtrip_and_split():
    with tempfile.TemporaryDirectory() as d:
        tier, kv, _ = make_tier(d)
        src = torch.arange(0, 64, dtype=torch.int32)
        kv.randomize(src)
        want = kv.rows(src)
        ref = tier.put_kv(src)
        assert ref is not None and ref.nbytes == 64 * tier.row_bytes
        kv.randomize(src)
        dst = torch.arange(100, 164, dtype=torch.int32)
        tier.get_kv(ref, dst)
        assert torch.equal(kv.rows(dst), want)
        head, tail = tier.split_kv(ref, 16)
        tier.get_kv(tail, dst[:48])
        assert torch.equal(kv.rows(dst[:48]), want[:, :, 16:])
        tier.get_kv(head, dst[:16])
        assert torch.equal(kv.rows(dst[:16]), want[:, :, :16])
        tier.close()


def _assert_extents_sane(arena: HostArena, live) -> None:
    ext = sorted(arena.free_extents)
    for (a, sa), (b, _) in zip(ext, ext[1:]):
        assert a + sa <= b, f"free extents overlap: {ext}"
    for off, size in ext:
        for ref in live:
            assert off + size <= ref.offset or off >= ref.offset + ref.nbytes, f"free extent {off}+{size} overlaps live {ref}"
    assert sum(size for _, size in ext) + arena.used == arena.capacity


def test_split_halves_free_exactly_the_extent_they_came_from():
    with tempfile.TemporaryDirectory() as d:
        kv = FakeKVPool(layers=10, pages=64, heads=2, head_dim=256)
        tier, _, _ = make_tier(d, capacity=64 * ALIGN, kv=kv)
        assert tier.row_bytes == 10240
        node = tier.put_kv(torch.arange(0, 3, dtype=torch.int32))
        neighbour = tier.put_kv(torch.arange(3, 4, dtype=torch.int32))
        head, tail = tier.split_kv(node, 1)
        head_a, head_b = tier.split_kv(head, 0)
        for ref in (tail, head_b, head_a):
            tier.free(ref)
        _assert_extents_sane(tier.arena, [neighbour])
        tier.free(neighbour)
        assert tier.arena.free_extents == [[0, tier.arena.capacity]] and tier.arena.used == 0
        tier.close()


def test_kv_roundtrip_spans_several_staging_chunks():
    for dtype in (torch.uint8, torch.float8_e4m3fn, torch.bfloat16):
        with tempfile.TemporaryDirectory() as d:
            tier, kv, _ = make_tier(d, kv=FakeKVPool(pages=8192, dtype=dtype))
            n = 2 * tier.stage_rows + 300
            src = torch.randperm(8192, dtype=torch.int32)[:n]
            kv.randomize(src)
            want = kv.rows(src)
            ref = tier.put_kv(src)
            assert ref is not None and ref.nbytes == n * tier.row_bytes
            kv.randomize(src)
            dst = torch.randperm(8192, dtype=torch.int32)[:n]
            tier.get_kv(ref, dst)
            assert torch.equal(kv.rows(dst), want), dtype
            tier.close()


def test_snapshot_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        tier, _, lin = make_tier(d)
        lin.randomize(2)
        conv, rec = lin.conv_states[:, 2].clone(), lin.recurrent_states[:, 2].clone()
        ref = tier.put_snap(2)
        assert ref is not None and ref.nbytes == tier.snap_bytes
        tier.get_snap(ref, 5)
        assert torch.equal(lin.conv_states[:, 5], conv)
        assert torch.equal(lin.recurrent_states[:, 5], rec)
        tier.close()


def test_meta_roundtrip_and_key_mismatch():
    with tempfile.TemporaryDirectory() as d:
        tier, kv, lin = make_tier(d)
        ref = tier.put_kv(torch.arange(0, 8, dtype=torch.int32))
        nodes = [{"parent": -1, "tokens": "AAAA", "kv": list(ref), "snap": None, "ts": 1}]
        tier.save_meta(nodes)
        free_before = [list(e) for e in tier.arena.free_extents]
        tier.close()
        again, _, _ = make_tier(d, kv=kv, lin=lin)
        assert again.take_loaded_nodes() == nodes
        assert again.take_loaded_nodes() == []
        assert again.arena.free_extents == free_before
        again.close()
        other, _, _ = make_tier(d, key={"model": "other"}, kv=kv, lin=lin)
        assert other.take_loaded_nodes() == []
        other.close()


def test_tree_evict_promote_and_reload():
    with tempfile.TemporaryDirectory() as d:
        tier, kv, lin = make_tier(d)
        cache = TieredHybridRadixCache(DEVICE, 1, tier)
        tokens = ids(128)
        slots = torch.arange(0, 128, dtype=torch.int32)
        kv.randomize(slots)
        lin.randomize(1)
        want = kv.rows(slots)
        prefix_len, node, dups = commit(cache, tier, tokens, slots, 1)
        assert prefix_len == 0 and dups == [] and node.resident and node.host_kv is not None
        assert cache.full_evictable_size == 128

        m = cache.match_prefix(ids(200))
        assert (m.cached_len, m.promote_tokens, m.snap) == (128, 0, node.host_snap)

        ev = cache.evict_full(64)
        assert ev.kv_indices.numel() == 128 and not node.resident and cache.full_evictable_size == 0
        m = cache.match_prefix(ids(200))
        assert (m.cached_len, m.promote_tokens) == (128, 128)
        cache.inc_lock(m.node)
        kv.randomize(slots)
        alloc = Alloc(500)
        assert cache.promote(m.node, alloc) == 128
        assert node.resident and cache.full_protected == 128
        restored = torch.cat([n.value for n in cache._path(m.node)])
        assert torch.equal(kv.rows(restored), want)
        cache.dec_lock(m.node)
        cache.check_integrity()

        tier.save_meta(cache.export_nodes())
        tier.close()

        tier2, _, _ = make_tier(d, kv=kv, lin=lin)
        cache2 = TieredHybridRadixCache(DEVICE, 1, tier2)
        assert cache2.import_nodes(tier2.take_loaded_nodes()) == 1
        m2 = cache2.match_prefix(ids(300))
        assert (m2.cached_len, m2.promote_tokens) == (128, 128)
        cache2.inc_lock(m2.node)
        cache2.promote(m2.node, Alloc(700))
        restored2 = torch.cat([n.value for n in cache2._path(m2.node)])
        assert torch.equal(kv.rows(restored2), want)
        lin.randomize(3)
        tier2.get_snap(m2.snap, 3)
        assert torch.equal(lin.conv_states[:, 3], lin.conv_states[:, 1])
        tier2.close()


def test_insert_adopts_host_only_prefix_and_reports_dups():
    with tempfile.TemporaryDirectory() as d:
        tier, kv, lin = make_tier(d)
        cache = TieredHybridRadixCache(DEVICE, 1, tier)
        commit(cache, tier, ids(64), torch.arange(0, 64, dtype=torch.int32), 1)
        cache.evict_full(64)
        new_slots = torch.arange(200, 328, dtype=torch.int32)
        prefix_len, node, dups = commit(cache, tier, ids(128), new_slots, 2)
        assert prefix_len == 64 and dups == []
        first = cache._path(node)[0]
        assert first.resident and torch.equal(first.value, new_slots[:64])
        assert node.length == 64 and cache.full_evictable_size == 128
        prefix_len, node2, dups = commit(cache, tier, ids(96), torch.arange(400, 496, dtype=torch.int32), 3)
        assert prefix_len == 96 and dups == [(0, 64), (64, 96)]
        assert node2.length == 32 and node2.host_snap is not None
        cache.check_integrity()
        tier.close()


def test_arena_eviction_drops_lru_leaf():
    with tempfile.TemporaryDirectory() as d:
        kv, lin = FakeKVPool(pages=1024), FakeLinearPool()
        probe, _, _ = make_tier(d, kv=kv, lin=lin)
        snap_bytes, row_bytes = probe.snap_bytes, probe.row_bytes
        probe.close()

        def ru(n):
            return (n + ALIGN - 1) // ALIGN * ALIGN

        capacity = 3 * (ru(snap_bytes) + ru(64 * row_bytes))
        tier, _, _ = make_tier(d, capacity=capacity, kv=kv, lin=lin)
        cache = TieredHybridRadixCache(DEVICE, 1, tier)
        nodes = []
        for i in range(5):
            base = 10_000 * (i + 1)
            _, node, _ = commit(cache, tier, ids(64, base), torch.arange(i * 64, i * 64 + 64, dtype=torch.int32), 1)
            nodes.append(node)
        assert cache.host_evictions >= 1
        assert nodes[0]._parent is None
        assert nodes[-1].host_kv is not None and nodes[-1].host_snap is not None
        assert cache.match_prefix(ids(64, 10_000)).cached_len == 0
        assert cache.match_prefix(ids(64, 50_000)).cached_len == 64
        freed = cache.take_freed_kv()
        assert freed.numel() >= 64
        cache.check_integrity()
        tier.close()


def _slow_landing(monkeypatch):
    import time

    import freetoken.kvcache.host_tier as ht

    land = ht.HostArena._land

    def slow(done, staging, dst):
        time.sleep(0.2)
        land(done, staging, dst)

    monkeypatch.setattr(ht.HostArena, "_land", staticmethod(slow))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_device_write_lands_before_a_read_of_its_range(monkeypatch):
    _slow_landing(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        arena = HostArena(os.path.join(d, "a.bin"), 64 * ALIGN)
        ref = arena.alloc(4 * ALIGN)
        src = torch.randint(0, 255, (4 * ALIGN,), dtype=torch.uint8, device="cuda")
        arena.write(ref, src)
        out = torch.empty(4 * ALIGN, dtype=torch.uint8)
        arena.read(ref, out)
        assert torch.equal(out, src.cpu())
        arena.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_freed_extent_is_reused_only_after_its_pending_write_landed(monkeypatch):
    _slow_landing(monkeypatch)
    with tempfile.TemporaryDirectory() as d:
        arena = HostArena(os.path.join(d, "a.bin"), 64 * ALIGN)
        ref = arena.alloc(4 * ALIGN)
        arena.write(ref, torch.full((4 * ALIGN,), 3, dtype=torch.uint8, device="cuda"))
        arena.free(ref)
        again = arena.alloc(4 * ALIGN)
        assert again.offset == ref.offset
        arena.write(again, torch.full((4 * ALIGN,), 7, dtype=torch.uint8))
        arena.flush()
        assert torch.equal(arena.view(again), torch.full((4 * ALIGN,), 7, dtype=torch.uint8))
        arena.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_device_commits_round_trip_through_the_tier():
    with tempfile.TemporaryDirectory() as d:
        kv = FakeKVPool(pages=8192)
        kv._kv_buffer = kv._kv_buffer.cuda()
        kv.device = torch.device("cuda")
        lin = FakeLinearPool()
        lin.conv_states, lin.recurrent_states = lin.conv_states.cuda(), lin.recurrent_states.cuda()
        tier = HostTier(d, 64 << 20, kv, lin, {"model": "fake"})
        raw = kv.raw()
        src = torch.arange(0, 2500, dtype=torch.int32, device="cuda")

        def scramble():
            raw[:, :, src.long()] = torch.randint(0, 255, raw[:, :, src.long()].shape, dtype=torch.uint8, device="cuda")
            lin.recurrent_states[:, 2] = torch.randn_like(lin.recurrent_states[:, 2])

        scramble()
        want_kv, want_rec = kv.rows(src), lin.recurrent_states[:, 2].clone()
        kref, sref = tier.put_kv(src), tier.put_snap(2)
        scramble()
        dst = torch.arange(3000, 3000 + 2500, dtype=torch.int32, device="cuda")
        tier.get_kv(kref, dst)
        tier.get_snap(sref, 5)
        assert torch.equal(kv.rows(dst), want_kv)
        assert torch.equal(lin.recurrent_states[:, 5], want_rec)
        tier.close()


def test_validity_key_follows_the_ablation_that_shaped_the_cached_states():
    from types import SimpleNamespace

    from freetoken.kvcache.host_tier import build_validity_key

    with tempfile.TemporaryDirectory() as d:
        table = os.path.join(d, "direction.safetensors")
        with open(table, "wb") as f:
            f.write(b"direction table v1")

        def key(**ablation):
            cfg = SimpleNamespace(model_path=d, page_size=1, kv_cache_scales=None, ablate_direction=None,
                                  ablate_layer=0, ablate_alpha=1.0)
            vars(cfg).update(ablation)
            return build_validity_key(cfg, FakeKVPool(), None)

        base = key(ablate_direction=table, ablate_layer=20, ablate_alpha=0.6)
        assert base == key(ablate_direction=table, ablate_layer=20, ablate_alpha=0.6)
        assert base != key()
        assert base != key(ablate_direction=table, ablate_layer=20, ablate_alpha=0.5)
        assert base != key(ablate_direction=table, ablate_layer=19, ablate_alpha=0.6)
        with open(table, "wb") as f:
            f.write(b"direction table v2")
        assert base != key(ablate_direction=table, ablate_layer=20, ablate_alpha=0.6)
