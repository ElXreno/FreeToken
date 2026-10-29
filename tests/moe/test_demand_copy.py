"""Copy-engine demand copy: the copy thread's rows land byte-exact and stamp their slots, a stalled copy thread
costs a timeout and an SM pull but never a wrong read or a reused slot, and a hybrid decode graph replayed with it
computes what it computes with the SM copy."""

from __future__ import annotations


import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, E, H, I, TOP_K, BS, SLOTS = 4, 16, 1024, 1024, 4, 2, 24


def _cache(demand: bool, seed: int = 0, monkeypatch=None):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.cpu_executor import CpuMoeExecutor
    from freetoken.moe.demand_copy import DemandCopy
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    g = torch.Generator().manual_seed(seed)
    sources = {
        "gate_up": [(torch.randn(E, 2 * I, H, generator=g) * 0.05).bfloat16().pin_memory() for _ in range(L)],
        "down": [(torch.randn(E, H, I, generator=g) * 0.05).bfloat16().pin_memory() for _ in range(L)],
    }
    cache = OffloadMoeCache(
        num_layers=L, num_experts=E, cache_size=SLOTS, device=torch.device("cuda"), decode_target="hybrid",
        hybrid_fetch_fraction=[0.5] * L,
    )
    cache.set_bank_sources(sources)
    ex = CpuMoeExecutor(cache, top_k=TOP_K, activation="silu", apply_router_weight_on_input=False, num_threads=4,
                        max_tokens=BS, device=torch.device("cuda"))
    if not ex._flag_sync:
        pytest.skip("the copy-engine demand copy needs the flag coordinator (stream memops)")
    cache.set_cpu_executor(ex)
    if demand:
        cache.demand_copy = DemandCopy(cache)
    return cache, ex


def _rows_match(cache, layer, e):
    s = int(cache.slot_for_id[layer, e])
    return s >= 0 and all(torch.equal(view[s].cpu(), src[layer][e]) for (src, _), view in zip(cache.banks, cache.pool_bank_views(layer)))


def _fetch(cache, layer, ids):
    """One layer's ensure with every miss fetched; the fetched (slot, expert) pairs."""
    cache.hybrid_fetch_fraction = [1.0] * L
    ids = torch.tensor([ids], dtype=torch.int32, device="cuda")
    cache.ensure_experts_hybrid(layer, ids, torch.full(ids.shape, 0.25, device="cuda"), clamp=True)
    n = int(cache.num_indices)
    return list(zip(cache.evict_slots[:n].tolist(), cache.src_indices[:n].tolist(), strict=True))


def test_rows_land_byte_exact_and_stamp_their_slots():
    cache, _ = _cache(True)
    dc = cache.demand_copy
    fetched = _fetch(cache, 1, [3, 7, 9, 12])
    assert len(fetched) == 4
    dc.publish(1)
    dc.wait(1)
    torch.cuda.synchronize()
    for slot, e in fetched:
        assert _rows_match(cache, 1, e), e
        assert int(dc.ready[slot]) == int(dc.gen[slot]) == 1
    rows, experts, nbytes, _, waited, _, _, timeouts, unpublished = dc.stats()
    assert (rows, experts, waited, timeouts, unpublished) == (1, 4, 1, 0, 0)
    assert nbytes == 4 * sum(src[1][0].numel() * src[1][0].element_size() for src, _ in cache.banks)
    assert int(dc.fallback[dc.lanes]) == 0


def test_two_lanes_of_a_layer_copy_from_their_own_streams():
    cache, _ = _cache(True)
    dc = cache.demand_copy
    rows = [torch.cuda.Stream(), torch.cuda.Stream()]
    fetched = []
    for lane, ids in enumerate(([2, 4, 6], [3, 5, 7, 9])):
        rows[lane].wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(rows[lane]):
            fetched.append(_fetch(cache, 3, ids))
            dc.publish(3, lane)
        torch.cuda.current_stream().wait_stream(rows[lane])
    for lane in (1, 0):
        with torch.cuda.stream(rows[lane]):
            dc.wait(3, lane)
    torch.cuda.synchronize()
    for slot, e in fetched[0] + fetched[1]:
        assert _rows_match(cache, 3, e) and int(dc.ready[slot]) == int(dc.gen[slot])
    assert dc.stats()[0] == 2 and dc.stats()[4] == 2 and dc.fallback[6:8].tolist() == [[0], [0]]


def test_a_layer_with_an_unacked_row_is_pulled_and_stamped_on_the_gpu():
    cache, _ = _cache(True)
    dc = cache.demand_copy
    c = 2 * dc.lanes
    dc.seq[c] += 1
    fetched = _fetch(cache, 2, [1, 2, 5])
    dc.publish(2)
    dc.wait(2)
    torch.cuda.synchronize()
    assert int(dc.state[c]) == 0 and int(dc.fallback[c]) == 3
    for slot, e in fetched:
        assert _rows_match(cache, 2, e) and int(dc.ready[slot]) == int(dc.gen[slot])
    assert dc.stats()[:2] == (0, 0) and dc.stats()[-1] == 1


def test_a_timed_out_row_is_pulled_and_its_slots_stay_pending_until_the_late_copy():
    cache, ex = _cache(True)
    dc = cache.demand_copy
    ex._ext.ce_detach()
    dc.timeout_ns = 200_000
    fetched = _fetch(cache, 1, [4, 6])
    c = dc.lanes
    dc.publish(1)
    dc.wait(1)
    torch.cuda.synchronize()
    assert int(dc.state[c]) == 1 and int(dc.fallback[c]) == 2 and dc.stats()[7] == 1
    pending = [slot for slot, _ in fetched]
    for slot, e in fetched:
        assert _rows_match(cache, 1, e) and int(dc.ready[slot]) != int(dc.gen[slot])
    # the pending experts go to the CPU and their slots are no victims
    ids = torch.tensor([[4, 6, 10, 11]], dtype=torch.int32, device="cuda")
    _, gpu_w, cpu_ids = cache.ensure_experts_hybrid(1, ids, torch.full(ids.shape, 0.25, device="cuda"), clamp=True)
    torch.cuda.synchronize()
    assert cpu_ids[0, :2].tolist() == [4, 6] and gpu_w[0, :2].tolist() == [0.0, 0.0]
    assert not set(cache.evict_slots[: int(cache.num_indices)].tolist()) & set(pending)
    # layer 1 has its row in flight: the next one is pulled on the GPU
    dc.publish(1)
    dc.wait(1)
    torch.cuda.synchronize()
    assert int(dc.state[c]) == 0
    dc._setup()
    dc.drain()
    for slot, e in fetched:
        assert _rows_match(cache, 1, e) and int(dc.ready[slot]) == int(dc.gen[slot])
    assert int(dc.ack[c]) == int(dc.seq[c])


def _gpu_experts(cache, layer, x, slots, w):
    gate_up, down = cache.pool_bank_views(layer)
    s = slots.long()
    h = torch.einsum("bkoh,bh->bko", gate_up[s].float(), x.float())
    g = (F.silu(h[..., :I]) * h[..., I:]).bfloat16().float()
    y = torch.einsum("bkhi,bki->bkh", down[s].float(), g)
    return (y * w[..., None]).sum(1).to(x.dtype)


def _reference(cache, layer, x, ids, w):
    gu = cache.bank_sources["gate_up"][layer].float()
    dn = cache.bank_sources["down"][layer].float()
    out = torch.zeros(BS, H)
    for t in range(BS):
        for k in range(TOP_K):
            e = int(ids[t, k])
            gate, up = gu[e, :I] @ x[t].float(), gu[e, I:] @ x[t].float()
            out[t] += float(w[t, k]) * (dn[e] @ (F.silu(gate) * up).bfloat16().float())
    return out


def _decode(cache, ex, xs, ids, ws, outs):
    """One hybrid decode step over every layer, as ``_split_hybrid_gemm`` runs it in copy mode."""
    from freetoken.moe.offload_kernels import split_fetched_routes

    dc = cache.demand_copy
    for layer in range(L):
        gpu_slots, gpu_w, cpu_ids = cache.ensure_experts_hybrid(layer, ids[layer], ws[layer], clamp=True)
        pending = ex.decode_submit(layer, xs[layer], ws[layer], cpu_ids)
        hit_s, hit_w, new_s, new_w = split_fetched_routes(cache, gpu_slots, gpu_w, 0)
        if dc is None:
            cache.copy_missing()
            hit = _gpu_experts(cache, layer, xs[layer], hit_s, hit_w)
        else:
            dc.publish(layer)
            hit = _gpu_experts(cache, layer, xs[layer], hit_s, hit_w)
            dc.wait(layer)
        gpu = hit + _gpu_experts(cache, layer, xs[layer], new_s, new_w)
        outs[layer].copy_(ex.decode_sync(pending, add_to=gpu))


def _steps(seed, count):
    g = torch.Generator().manual_seed(seed)
    return [([torch.randn(BS, H, generator=g).bfloat16() for _ in range(L)],
             [torch.stack([torch.randperm(12, generator=g)[:TOP_K] for _ in range(BS)]).int() for _ in range(L)],
             [torch.rand(BS, TOP_K, generator=g) for _ in range(L)]) for _ in range(count)]


def _replays(demand, steps, after_replay=None):
    cache, ex = _cache(demand)
    xs = [torch.zeros(BS, H, dtype=torch.bfloat16, device="cuda") for _ in range(L)]
    ids = [torch.zeros(BS, TOP_K, dtype=torch.int32, device="cuda") for _ in range(L)]
    ws = [torch.zeros(BS, TOP_K, dtype=torch.float32, device="cuda") for _ in range(L)]
    outs = [torch.zeros(BS, H, dtype=torch.bfloat16, device="cuda") for _ in range(L)]

    def load(step):
        x, i, w = step
        for layer in range(L):
            xs[layer].copy_(x[layer])
            ids[layer].copy_(i[layer])
            ws[layer].copy_(w[layer])

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        load(steps[0])
        _decode(cache, ex, xs, ids, ws, outs)
        graph = torch.cuda.CUDAGraph()
        load(steps[0])
        with torch.cuda.graph(graph, stream=stream):
            _decode(cache, ex, xs, ids, ws, outs)
    torch.cuda.synchronize()
    cache.reset()
    results = []
    for n, step in enumerate(steps):
        load(step)
        graph.replay()
        if after_replay is not None:
            after_replay(n)
        torch.cuda.synchronize()
        results.append([o.float().cpu() for o in outs])
    return cache, results


def _check(cache, steps, *runs):
    for n, step in enumerate(steps):
        x, i, w = step
        for layer in range(L):
            ref = _reference(cache, layer, x[layer], i[layer], w[layer])
            scale = ref.abs().max()
            for out in runs:
                assert (out[n][layer] - ref).abs().max() / scale < 2e-2, (n, layer)


def test_a_decode_graph_replays_the_same_with_the_copy_engine():
    steps = _steps(7, 40)
    cache, sm = _replays(False, steps)
    ce_cache, ce = _replays(True, steps)
    _check(cache, steps, sm, ce)
    rows, experts, _, _, waited, _, _, timeouts, unpublished = ce_cache.demand_copy.stats()
    assert rows > 0 and experts > 0 and waited == rows and timeouts == unpublished == 0
