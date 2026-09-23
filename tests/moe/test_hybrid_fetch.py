"""Hybrid decode's bandwidth-matched fetch split.

Covers the two halves of --moe-hybrid-max-fetch auto: the profile reader that turns
`ft bench bw` kernel bandwidths into a fetch fraction, and the ensure kernel's
per-step integer split (GPU kernel vs CPU reference mirror, and the balance rule).
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch

from freetoken.moe.bench_profile import default_profile_path, load_backend_recommendation, load_hybrid_fetch_fraction
from freetoken.moe.offload_cache import OffloadMoeCache

Q = 1 << 16


def _balanced_fetch(num_missing: int, frac_q16: int) -> int:
    """Reference split: F ~ frac * misses, rounded to whichever integer neighbor
    minimizes the slower overlapped side (fetch ~ F*(1-frac), CPU ~ (M-F)*frac)."""
    lo = (num_missing * frac_q16) >> 16
    cost = lambda f: max(f * (Q - frac_q16), (num_missing - f) * frac_q16)  # noqa: E731
    return min(num_missing, lo if cost(lo) <= cost(lo + 1) else lo + 1)


def test_balanced_fetch_tracks_fraction():
    # The split follows fetched : cpu = pcie : (cpu - pcie) up to integer rounding, and
    # never over/under-shoots by more than one expert.
    for frac in (0.1, 0.415, 0.454, 0.7, 1.0):
        q = round(frac * Q)
        for m in range(0, 65):
            f = _balanced_fetch(m, q)
            assert 0 <= f <= m
            assert abs(f - frac * m) <= 1.0
    # ceil would over-fetch here (the regression this rule fixed): 41.5% of 3 misses is
    # 1.24 -> fetching 2 makes the PCIe side ~1.6x slower than balance; keep it at 1.
    assert _balanced_fetch(3, round(0.415 * Q)) == 1
    assert _balanced_fetch(4, round(0.415 * Q)) == 2


def test_load_hybrid_fetch_fraction(tmp_path):
    prof = {
        "gpu": {"name": "FAKE GPU"},
        "dtype_kernels": {
            "bf16": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0},
            # overlapped (contended) pair wins over the standalone numbers when present
            "nvfp4_x": {"cpu_moe_gbs": 100.0, "pcie_gather_gbs": 40.0,
                        "cpu_moe_overlap_gbs": 90.0, "pcie_gather_overlap_gbs": 30.0},
        },
        "workloads": {
            "m": {"kernels": {"ds_fp4": {"cpu_moe_gbs": 80.0, "pcie_gather_gbs": 50.0}}}
        },
    }
    path = tmp_path / "benchbw.json"
    path.write_text(json.dumps(prof))
    # standalone fallback: full-contention assumption -> pcie / cpu
    assert load_hybrid_fetch_fraction("bf16", path=str(path)) == pytest.approx(0.4)
    # overlapped pair preferred: pcie_ov / (pcie_ov + cpu_ov)
    assert load_hybrid_fetch_fraction("nvfp4_x", path=str(path)) == pytest.approx(0.25)
    # per-model fallback when there is no per-dtype entry for the format
    assert load_hybrid_fetch_fraction("ds_fp4", path=str(path)) == pytest.approx(0.625)
    assert load_hybrid_fetch_fraction("nvfp4", path=str(path)) is None
    # a profile from different hardware is ignored
    assert load_hybrid_fetch_fraction("bf16", gpu_name="OTHER", path=str(path)) is None


def test_profile_lookup_prefers_the_gpu_uuid_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.delenv("FREETOKEN_BENCHBW_PATH", raising=False)
    uuid = "GPU-2f3a9b1c-0000-1111-2222-333344445555"

    def write(path, name, verdict):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"gpu": {"name": name}, "dtypes": {"bf16": verdict}}, f)

    # legacy single file only: used when the name matches, ignored otherwise
    write(default_profile_path(), "FAKE GPU", "hybrid")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "hybrid"
    assert load_backend_recommendation("bf16", gpu_name="OTHER", gpu_uuid=uuid) is None
    # this card's own file wins over the legacy one
    write(default_profile_path(uuid), "FAKE GPU", "offload")
    assert load_backend_recommendation("bf16", gpu_name="FAKE GPU", gpu_uuid=uuid) == "offload"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fraction_gpu_matches_cpu_reference():
    torch.manual_seed(0)
    num_experts, cache_size, top_k, frac = 32, 40, 8, 0.415

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=frac,
        )

    gpu, ref = make(), make()
    frac_q16 = round(frac * Q)
    for step in range(64):
        ids = torch.randperm(num_experts)[:top_k].to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()  # a CPU ids tensor drives the reference path
        gpu.ensure_experts_hybrid(0, g)
        ref.ensure_experts_hybrid(0, c)
        missing = int(gpu.num_missing_full.item())
        fetched = int(gpu.num_indices.item())
        assert missing == int(ref.num_missing_full.item())
        assert fetched == int(ref.num_indices.item()) == _balanced_fetch(missing, frac_q16)
        # slot rewrites (hit/fetched -> slot, overflow -> -1) and LRU state stay identical
        assert torch.equal(g.cpu(), c)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        assert torch.equal(gpu.id_of_slot.cpu(), ref.id_of_slot.cpu())
        assert (g >= 0).sum().item() == len(set(ids.tolist())) - (missing - fetched)


def _resolve(monkeypatch, flag, profile):
    import freetoken.engine.engine as eng
    import freetoken.moe.bench_profile as bp

    monkeypatch.setattr(eng, "_profile_gpu", lambda index=None: ("FAKE GPU", None))
    monkeypatch.setattr(bp, "load_hybrid_fetch_fraction", lambda *a, **k: profile)
    config = SimpleNamespace(moe_hybrid_max_fetch=-1, moe_hybrid_fetch_fraction=flag)
    cache = SimpleNamespace(
        quant_format="nvfp4", num_experts=256, hybrid_max_fetch=1, hybrid_fetch_fraction=0.0
    )
    eng.Engine._resolve_hybrid_fetch(SimpleNamespace(device=torch.device("cpu")), config, cache)
    return cache.hybrid_max_fetch, cache.hybrid_fetch_fraction


def test_fetch_fraction_flag_overrides_the_profile(monkeypatch):
    assert _resolve(monkeypatch, -1.0, 0.28) == (256, 0.28)
    assert _resolve(monkeypatch, 0.1, 0.28) == (256, 0.1)
    assert _resolve(monkeypatch, 0.1, None) == (256, 0.1)
    assert _resolve(monkeypatch, -1.0, None) == (1, 0.0)
    # zero means no fetches at all, not the fraction-off fixed cap of num_experts
    assert _resolve(monkeypatch, 0.0, 0.28) == (0, 0.0)


def test_tiny_fraction_does_not_turn_into_the_fixed_cap():
    from freetoken.moe.offload_kernels import ensure_experts_hybrid

    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cpu"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=32,
    )
    ensure_experts_hybrid(cache, 0, torch.arange(8, dtype=torch.int32), 32, 1e-7)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize(("rows", "num_experts"), [(2, 32), (12, 256)])
def test_hybrid_cpu_routes_gpu_match_cpu_reference(rows, num_experts):
    from freetoken.moe.offload_kernels import ensure_experts_hybrid

    torch.manual_seed(0)
    cache_size, top_k = num_experts + 8, 8

    def make():
        return OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts,
        )

    gpu, ref = make(), make()
    for step in range(64):
        ids = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(rows)]).to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()
        g_cpu, c_cpu = torch.empty_like(g), torch.empty_like(c)
        ensure_experts_hybrid(gpu, step % 2, g, num_experts, 0.3, cpu_ids=g_cpu)
        ensure_experts_hybrid(ref, step % 2, c, num_experts, 0.3, cpu_ids=c_cpu)
        assert int(gpu.num_indices.item()) == int(ref.num_indices.item())
        assert torch.equal(g.cpu(), c)
        assert torch.equal(g_cpu.cpu(), c_cpu)
        assert torch.equal(gpu.slot_for_id.cpu(), ref.slot_for_id.cpu())
        missing = int(ref.num_missing_full.item())
        uniq = len(set(ids.view(-1).tolist()))
        on_cpu = missing - int(ref.num_indices.item())
        assert len(set(c_cpu[c_cpu >= 0].tolist())) == on_cpu
        assert len(set(c[c >= 0].tolist())) == uniq - on_cpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fused_hybrid_bookkeeping_matches_the_reference(monkeypatch):
    import freetoken.moe.offload_kernels as ok

    torch.manual_seed(0)
    num_experts, cache_size, top_k, rows = 32, 40, 8, 2

    def make():
        cache = OffloadMoeCache(
            num_layers=2, num_experts=num_experts, cache_size=cache_size,
            device=torch.device("cuda"), quant_format="bf16", decode_target="hybrid",
            hybrid_max_fetch=num_experts, hybrid_fetch_fraction=0.3,
        )
        cache.collect_stats = cache.collect_decode_freq = True
        return cache

    monkeypatch.setattr(ok, "HYBRID_FUSED", True)
    fused, ref = make(), make()
    ref.collect_decode_freq = False
    freq = torch.zeros_like(ref.decode_freq, device="cpu")
    for step in range(64):
        layer = step % 2
        ids = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(rows)]).to(torch.int32)
        g, c = ids.clone().cuda(), ids.clone()
        g_cpu = torch.full_like(g, 7)
        fused.ensure_experts_hybrid(layer, g, cpu_ids=g_cpu)
        fused.record_decode_stats_hybrid(layer)
        ref.ensure_experts_hybrid(layer, c)
        ref.record_decode_stats_hybrid(layer)
        freq[layer] += torch.bincount(ids.view(-1).long(), minlength=num_experts).to(freq.dtype)
        assert torch.equal(g.cpu(), c)
        assert torch.equal(g_cpu.cpu(), torch.where(c >= 0, -1, ids))
    assert torch.equal(fused.hybrid_stats.cpu(), ref.hybrid_stats.cpu())
    assert torch.equal(fused.decode_freq.cpu(), freq)
    assert int(fused.stat_calls.item()) == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_hybrid_fixed_cap_unchanged():
    # fraction 0 (no profile / explicit --moe-hybrid-max-fetch) keeps the fixed cap.
    cache = OffloadMoeCache(
        num_layers=1, num_experts=32, cache_size=40, device=torch.device("cuda"),
        quant_format="bf16", decode_target="hybrid", hybrid_max_fetch=1,
    )
    ids = torch.arange(8, dtype=torch.int32).cuda()
    cache.ensure_experts_hybrid(0, ids)
    assert int(cache.num_missing_full.item()) == 8
    assert int(cache.num_indices.item()) == 1
