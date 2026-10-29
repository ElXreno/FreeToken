"""The decode routing trace: ring writes per layer, ordered drains with markers, graph-replay safety."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch
from freetoken.moe import routing_trace as rt

L, E, K = 3, 16, 2


def _parse(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == rt.MAGIC
    version, layers, experts, top_k = struct.unpack_from("<4I", data, 4)
    assert (version, layers, experts, top_k) == (rt.VERSION, L, E, K)
    off, out = 20, []
    while off < len(data):
        kind = data[off]
        off += 1
        if kind == rt.STEP:
            (n,) = struct.unpack_from("<H", data, off)
            off += 2
            cnt = L * n * K
            ids = np.frombuffer(data, np.int16, cnt, off).reshape(L, n, K)
            off += 2 * cnt
            w = np.frombuffer(data, np.float16, cnt, off).reshape(L, n, K)
            off += 2 * cnt
            out.append(("step", ids.tolist(), w.tolist()))
        elif kind == rt.START:
            (npools,) = struct.unpack_from("<I", data, off)
            off += 4
            pools = [struct.unpack_from("<2I", data, off + 8 * i) for i in range(npools)]
            off += 8 * npools
            (nseeds,) = struct.unpack_from("<I", data, off)
            off += 4
            seeds = [struct.unpack_from("<2HI", data, off + 8 * i) for i in range(nseeds)]
            off += 8 * nseeds
            out.append(("start", pools, seeds))
        elif kind == rt.LOST:
            (n,) = struct.unpack_from("<I", data, off)
            off += 4
            out.append(("lost", n))
        else:
            out.append(({rt.PREFILL: "prefill", rt.RESET: "reset"}[kind],))
    return out


def _step(trace, base, rows=1, skip=()):
    for layer in range(L):
        if layer in skip:
            continue
        ids = (torch.arange(rows * K).view(rows, K) + base + layer) % E
        trace.record(layer, ids.to(torch.int32), torch.full((rows, K), 0.5 + layer))


def _trace(tmp_path, ring=8, device="cpu"):
    return rt.RoutingTrace(str(tmp_path / "t.bin"), num_layers=L, num_experts=E, top_k=K, max_rows=4,
                           device=torch.device(device), ring=ring)


def test_steps_drain_in_order_with_their_markers(tmp_path):
    t = _trace(tmp_path)
    _step(t, 9)  # capture warm-up: start() drops it
    t.start([(40, 32)], [(0, 4, 32), (1, 4, 36)])
    _step(t, 0)
    _step(t, 1, rows=2, skip=(2,))
    t.marker(rt.PREFILL)
    _step(t, 2)
    t.marker(rt.RESET)
    t.close()
    recs = _parse(tmp_path / "t.bin")
    assert [r[0] for r in recs] == ["start", "step", "step", "prefill", "step", "reset"]
    assert recs[0][1:] == ([(40, 32)], [(0, 4, 32), (1, 4, 36)])
    assert recs[1][1] == [[[0, 1]], [[1, 2]], [[2, 3]]]
    assert recs[1][2] == [[[0.5, 0.5]], [[1.5, 1.5]], [[2.5, 2.5]]]
    assert recs[2][1] == [[[1, 2], [3, 4]], [[2, 3], [4, 5]], [[-1, -1], [-1, -1]]]
    assert recs[4][1] == [[[2, 3]], [[3, 4]], [[4, 5]]]


def test_a_ring_overrun_is_reported_not_misread(tmp_path):
    t = _trace(tmp_path, ring=4)
    t.start([(40, 32)], [])
    for s in range(6):
        _step(t, s)
    t.close()
    recs = _parse(tmp_path / "t.bin")
    assert [r[0] for r in recs] == ["start", "lost", "step", "step", "step", "step"]
    assert recs[1][1] == 2
    assert [r[1][0][0][0] for r in recs[2:]] == [2, 3, 4, 5]


def test_drained_rows_are_cleared_for_layers_that_skip_a_step(tmp_path):
    t = _trace(tmp_path, ring=2)
    t.start([(40, 32)], [])
    _step(t, 0)
    t.drain()
    _step(t, 1)
    _step(t, 2, skip=(1,))
    t.close()
    recs = _parse(tmp_path / "t.bin")
    assert recs[3][1][1] == [[-1, -1]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_replays_record_their_own_routing(tmp_path):
    t = _trace(tmp_path, ring=16, device="cuda")
    ids = torch.zeros((1, K), dtype=torch.int32, device="cuda")
    w = torch.ones((1, K), device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for layer in range(L):
            t.record(layer, ids, w)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for layer in range(L):
            t.record(layer, ids + layer, w)
    t.start([(40, 32)], [])
    for step in range(3):
        ids.fill_(step)
        g.replay()
    t.close()
    recs = _parse(tmp_path / "t.bin")
    assert [r[1] for r in recs[1:]] == [[[[s + layer] * K] for layer in range(L)] for s in range(3)]


def test_the_offload_layer_records_raw_ids_before_ensure_rewrites_them(tmp_path, monkeypatch):
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.layers.quantization import NoQuantConfig
    from freetoken.moe.offload_cache import OffloadMoeCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    layer = OffloadMoELayer(0, 4, 2, 8, 16, quant_config=NoQuantConfig(), prefix="model.layers.0.mlp.experts")
    cache = OffloadMoeCache(num_layers=1, num_experts=4, cache_size=6, device=torch.device("cpu"))
    cache.set_bank_sources({"gate_up": [torch.randn(4, 32, 8)], "down": [torch.randn(4, 8, 16)]})
    layer.offload_cache = cache
    cache.routing_trace = rt.RoutingTrace(str(tmp_path / "t.bin"), num_layers=1, num_experts=4, top_k=2, max_rows=2,
                                          device=torch.device("cpu"))
    cache.routing_trace.start([(6, 0)], [])
    topk = (torch.tensor([[0.75, 0.25]]), torch.tensor([[2, 1]], dtype=torch.int32))
    monkeypatch.setattr("freetoken.layers.moe.fused_topk", lambda **_: topk)
    monkeypatch.setattr(cache, "ensure_experts", lambda layer_id, ids: ids.copy_(torch.tensor([[5, 0]])))
    monkeypatch.setattr(cache, "copy_missing", lambda: None)
    monkeypatch.setattr("freetoken.moe.fused.fused_experts_decode_impl", lambda hidden_states, *a, **k: hidden_states)
    layer.decode_forward(torch.randn(1, 8), torch.randn(1, 4))
    cache.routing_trace.close()
    with open(tmp_path / "t.bin", "rb") as f:
        data = f.read()
    step = data.index(bytes([rt.STEP]), 20 + 1 + 4 + 8 + 4)
    assert struct.unpack_from("<H2h2e", data, step + 1) == (1, 2, 1, 0.75, 0.25)
