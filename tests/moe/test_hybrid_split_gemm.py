"""Hybrid split GEMM: hit routes and fetched routes partition the GPU routes, and their two launches sum to one."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _cache(evict, slots_total=12):
    cache = SimpleNamespace(
        evict_slots=torch.tensor(evict + [0] * (8 - len(evict)), dtype=torch.int32, device="cuda"),
        num_indices=torch.tensor(len(evict), dtype=torch.int64, device="cuda"),
        copied=[],
    )
    cache.pool_bank_views = lambda layer: ()
    cache.alphas_for_slots = lambda layer, slots: None
    cache.copy_missing = lambda: cache.copied.append(True)
    stream = torch.cuda.Stream()
    cache.split_gemm_stream = lambda: stream
    return cache


@cuda
def test_routes_split_by_this_steps_fetched_slots():
    from freetoken.moe.offload_kernels import split_fetched_routes

    slots = torch.tensor([[3, 7, 0, 5], [7, 9, 2, 0]], dtype=torch.int32, device="cuda")
    w = torch.tensor([[0.4, 0.3, 0.0, 0.3], [0.5, 0.2, 0.2, 0.1]], device="cuda")
    hs, hw, ns, nw = split_fetched_routes(_cache([7, 2]), slots, w)
    assert hs.tolist() == [[3, 0, 0, 5], [0, 9, 0, 0]]
    assert ns.tolist() == [[0, 7, 0, 0], [7, 0, 2, 0]]
    assert torch.equal(hw + nw, w)
    assert torch.equal(hw * nw, torch.zeros_like(w))


@cuda
def test_the_two_launches_sum_to_the_single_one(monkeypatch):
    from freetoken.layers import moe as moe_mod

    table = torch.randn(12, 8, device="cuda")

    def gemm(self, cache, x, w, slots, **_):
        return x + (w.unsqueeze(-1) * table[slots.long()]).sum(1)

    monkeypatch.setattr(moe_mod.OffloadMoELayer, "_expert_gemm", gemm)
    layer = SimpleNamespace(layer_id=0, _expert_gemm=lambda *a, **k: gemm(layer, *a, **k))
    slots = torch.tensor([[3, 7, 0, 5], [7, 9, 2, 0]], dtype=torch.int32, device="cuda")
    w = torch.rand(2, 4, device="cuda")
    x = torch.randn(2, 8, device="cuda")
    cache = _cache([7, 2])
    out = moe_mod.OffloadMoELayer._split_hybrid_gemm(layer, cache, x, w, slots, None)
    torch.cuda.synchronize()
    want = 2 * x + (w.unsqueeze(-1) * table[slots.long()]).sum(1)
    assert cache.copied == [True]
    torch.testing.assert_close(out, want)
