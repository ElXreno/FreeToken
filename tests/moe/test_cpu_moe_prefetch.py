"""The CPU MoE pool's cache prefetch of hinted next-layer experts never changes a result and skips stale hints."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

L, E, H, I, TOP_K = 3, 6, 512, 256, 4


def _cache():
    g = torch.Generator().manual_seed(0)

    def rb(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=g)

    layers = [{"gate_up": rb(E, H // 16, 4 * I * 4).view(torch.int32), "gate_up_scale": rb(E, H // 16, 2 * I).clamp_max(0x40),
               "down": rb(E, I // 16, 2 * H * 4).view(torch.int32), "down_scale": rb(E, I // 16, H).clamp_max(0x40)}
              for _ in range(L)]
    return SimpleNamespace(quant_format="nvfp4_b12x", bank_sources={k: [l[k] for l in layers] for k in layers[0]},
                           gate_up_alpha=torch.full((L * E,), 1e36), down_alpha=torch.full((L * E,), 1e36),
                           num_layers=L, num_experts=E, decode_target="cpu", cpu_executor=None)


def _executor(monkeypatch, cache, prefetch, device):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    if prefetch:
        monkeypatch.setenv("FREETOKEN_CPU_MOE_PREFETCH", "1")
    else:
        monkeypatch.delenv("FREETOKEN_CPU_MOE_PREFETCH", raising=False)
    return CpuMoeExecutor(cache, top_k=TOP_K, activation="silu", apply_router_weight_on_input=False, num_threads=4,
                          max_tokens=1, device=device)


def _hint(ex, layer, stamp, ids):
    ex.prefetch_hint[layer, 1] = len(ids)
    ex.prefetch_hint[layer, 2:2 + len(ids)] = torch.tensor(ids, dtype=torch.int32)
    ex.prefetch_hint[layer, 0] = stamp


def _run(ex, layer, x, ids, w):
    ex._io["x"][0].copy_(x)
    ex._io["ids"][0].copy_(ids)
    ex._io["w"][0].copy_(w)
    ex._ext.run(layer, 1)
    return ex._io["y"][0].clone()


def test_a_prefetched_layer_computes_the_same_bits(monkeypatch):
    cache = _cache()
    plain = _executor(monkeypatch, cache, False, torch.device("cpu"))
    pf = _executor(monkeypatch, cache, True, torch.device("cpu"))
    assert plain.prefetch_hint is None and pf.prefetch_hint is not None
    x = torch.randn(H, dtype=torch.bfloat16)
    ids = torch.tensor([3, -1, 0, 5], dtype=torch.int32)
    w = torch.rand(TOP_K)
    ref = _run(plain, 1, x, ids, w)
    assert torch.isfinite(ref.float()).all()
    _hint(pf, 1, 7, [3, 0, 5])
    pf._ext.prefetch_now(1)
    assert torch.equal(_run(pf, 1, x, ids, w), ref)
    per_expert = sum(t[0].numel() * t[0].element_size() for t in cache.bank_sources.values()) // E
    assert pf._ext.prefetch_stats() == (1, 3 * per_expert, 0)


def test_stale_hints_and_foreign_ids_are_skipped(monkeypatch):
    cache = _cache()
    pf = _executor(monkeypatch, cache, True, torch.device("cpu"))
    per_expert = sum(t[0].numel() * t[0].element_size() for t in cache.bank_sources.values()) // E
    _hint(pf, 2, 1, [-1, 99, 4])
    pf._ext.prefetch_now(2)
    pf._ext.prefetch_now(2)
    assert pf._ext.prefetch_stats() == (1, per_expert, 0)
    _hint(pf, 2, 2, [-1, 99])
    pf._ext.prefetch_now(2)
    assert pf._ext.prefetch_stats()[0] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_coordinator_prefetches_between_layers_without_changing_outputs(monkeypatch):
    cache = _cache()
    dev = torch.device("cuda")
    plain = _executor(monkeypatch, cache, False, dev)
    pf = _executor(monkeypatch, cache, True, dev)
    if not pf._flag_sync:
        pytest.skip("no flag handshake on this host")
    torch.manual_seed(1)
    for step in range(6):
        for layer in range(L):
            hidden = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
            ids = torch.stack([torch.randperm(E)[:TOP_K]]).to(device=dev, dtype=torch.int32)
            w = torch.rand(1, TOP_K, device=dev)
            if layer + 1 < L:
                _hint(pf, layer + 1, step * L + layer, torch.randperm(E)[:3].tolist())
            want = plain.decode(layer, hidden, w, ids.clone())
            got = pf.decode(layer, hidden, w, ids.clone())
            torch.cuda.synchronize()
            assert torch.equal(got, want), (step, layer)
    tasks, nbytes, aborted = pf._ext.prefetch_stats()
    assert tasks > 0 and nbytes > 0 and aborted <= tasks
    assert not pf._ext.flag_pending()
    pf.raise_if_unhealthy()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_gpu_hint_names_the_best_cold_experts_and_follows_the_graph():
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.cpu_prefetch_hint import write_cpu_prefetch_hint

    width = 34
    hint = alloc_pinned_tensor(L, width, dtype=torch.int32)
    hint.fill_(-1)
    slot_for_id = torch.full((L, E), -1, dtype=torch.int32, device="cuda")
    slot_for_id[1, 5] = 0
    cache = SimpleNamespace(num_layers=L, num_experts=E, slot_for_id=slot_for_id,
                            step=torch.tensor([41], dtype=torch.int64, device="cuda"),
                            cpu_executor=SimpleNamespace(prefetch_hint=hint))
    score = torch.arange(E, dtype=torch.float32, device="cuda")
    write_cpu_prefetch_hint(cache, 1, score, max_n=3)
    torch.cuda.synchronize()
    assert hint[1, :5].tolist() == [41, 3, 4, 3, 2]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write_cpu_prefetch_hint(cache, 1, score, max_n=3)
    score.copy_(-score)
    cache.step.fill_(42)
    graph.replay()
    torch.cuda.synchronize()
    assert hint[1, :5].tolist() == [42, 3, 0, 1, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_the_hint_takes_the_best_cold_experts_of_the_top_n_walk_with_ties():
    """Against the walk it replaces: visit the top ``topn`` experts by score (ties to the lower id), keep the cold ones."""
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.cpu_prefetch_hint import write_cpu_prefetch_hint

    g = torch.Generator().manual_seed(7)
    experts = 64
    for case in range(200):
        score = (torch.randint(0, 12, (experts,), generator=g).float() / 4).to(torch.bfloat16 if case % 2 else torch.float32)
        slot_for_id = torch.where(torch.rand(1, experts, generator=g) < 0.5, 0, -1).to(torch.int32)
        topn = int(torch.randint(1, 20, (), generator=g))
        max_n = int(torch.randint(1, 7, (), generator=g))
        order = sorted(range(experts), key=lambda e: (-float(score[e]), e))[:topn]
        want = [e for e in order if slot_for_id[0, e] == -1][:max_n]
        hint = alloc_pinned_tensor(1, 34, dtype=torch.int32)
        hint.fill_(-1)
        cache = SimpleNamespace(num_layers=1, num_experts=experts, slot_for_id=slot_for_id.cuda(),
                                step=torch.tensor([5], dtype=torch.int64, device="cuda"),
                                cpu_executor=SimpleNamespace(prefetch_hint=hint))
        write_cpu_prefetch_hint(cache, 0, score.cuda(), max_n=max_n, topn=topn)
        torch.cuda.synchronize()
        assert hint[0, 1].item() == len(want) and hint[0, 2:2 + len(want)].tolist() == want, (case, topn, max_n)
