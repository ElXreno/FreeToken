"""The opt-in router-input dump of the routing trace: one bf16 row per MoE layer after each step."""

from __future__ import annotations

import struct

import numpy as np
import pytest
import torch
from freetoken.moe import routing_trace as rt

L, E, K, D = 3, 16, 2, 8


def _bf16(values: torch.Tensor) -> np.ndarray:
    return values.to(torch.bfloat16).view(torch.int16).numpy()


def _parse(path):
    with open(path, "rb") as f:
        data = f.read()
    version, layers, _, top_k = struct.unpack_from("<4I", data, 4)
    off, out = 20, []
    while off < len(data):
        kind = data[off]
        off += 1
        if kind == rt.STEP:
            (n,) = struct.unpack_from("<H", data, off)
            off += 2 + 4 * layers * n * top_k
            out.append(("step",))
        elif kind == rt.HIDDEN:
            nl, d = struct.unpack_from("<HH", data, off)
            off += 4
            out.append(("hidden", np.frombuffer(data, np.int16, nl * d, off).reshape(nl, d).copy()))
            off += 2 * nl * d
        elif kind == rt.START:
            (npools,) = struct.unpack_from("<I", data, off)
            off += 4 + 8 * npools
            (nseeds,) = struct.unpack_from("<I", data, off)
            off += 4 + 8 * nseeds
            out.append(("start",))
        elif kind == rt.LOST:
            off += 4
            out.append(("lost",))
        else:
            out.append(("marker",))
    return version, out


def _trace(tmp_path, ring=8, hidden=True):
    return rt.RoutingTrace(str(tmp_path / "t.bin"), num_layers=L, num_experts=E, top_k=K, max_rows=4,
                           device=torch.device("cpu"), ring=ring, hidden_size=D if hidden else None)


def _hidden(step: int, layer: int, rows: int = 1) -> torch.Tensor:
    return torch.arange(rows * D, dtype=torch.float32).view(rows, D) / 8 + step * 10 + layer


def _step(trace, step, rows=1, skip=()):
    for layer in range(L):
        if layer in skip:
            continue
        ids = torch.zeros((rows, K), dtype=torch.int32)
        trace.record(layer, ids, torch.ones((rows, K)), _hidden(step, layer, rows))


def test_each_step_is_followed_by_the_first_rows_router_inputs(tmp_path):
    t = _trace(tmp_path)
    _step(t, 9)
    t.start([(40, 32)], [])
    _step(t, 0, rows=2)
    _step(t, 1, skip=(1,))
    t.close()
    version, recs = _parse(tmp_path / "t.bin")
    assert version == rt.VERSION_HIDDEN
    assert [r[0] for r in recs] == ["start", "step", "hidden", "step", "hidden"]
    np.testing.assert_array_equal(recs[2][1], np.stack([_bf16(_hidden(0, layer)[0]) for layer in range(L)]))
    want = np.stack([_bf16(_hidden(1, 0)[0]), np.zeros(D, np.int16), _bf16(_hidden(1, 2)[0])])
    np.testing.assert_array_equal(recs[4][1], want)


def test_without_the_option_the_file_stays_version_one(tmp_path):
    t = _trace(tmp_path, hidden=False)
    t.start([(40, 32)], [])
    _step(t, 0)
    t.close()
    version, recs = _parse(tmp_path / "t.bin")
    assert version == rt.VERSION
    assert [r[0] for r in recs] == ["start", "step"]


def test_steps_older_than_the_hidden_ring_keep_their_routing_but_drop_the_dump(tmp_path, monkeypatch):
    monkeypatch.setattr(rt, "_HIDDEN_RING", 2)
    t = _trace(tmp_path, ring=8)
    t.start([(40, 32)], [])
    for s in range(4):
        _step(t, s)
    t.close()
    _, recs = _parse(tmp_path / "t.bin")
    assert [r[0] for r in recs] == ["start", "step", "step", "step", "hidden", "step", "hidden"]
    np.testing.assert_array_equal(recs[4][1][0], _bf16(_hidden(2, 0)[0]))
    np.testing.assert_array_equal(recs[6][1][0], _bf16(_hidden(3, 0)[0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_replays_dump_their_own_router_inputs(tmp_path):
    t = rt.RoutingTrace(str(tmp_path / "t.bin"), num_layers=L, num_experts=E, top_k=K, max_rows=4,
                        device=torch.device("cuda"), ring=16, hidden_size=D)
    ids = torch.zeros((1, K), dtype=torch.int32, device="cuda")
    w = torch.ones((1, K), device="cuda")
    h = torch.zeros((1, D), dtype=torch.bfloat16, device="cuda")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for layer in range(L):
            t.record(layer, ids, w, h)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for layer in range(L):
            t.record(layer, ids, w, h + layer)
    t.start([(40, 32)], [])
    for step in range(3):
        h.fill_(step * 10)
        g.replay()
    t.close()
    _, recs = _parse(tmp_path / "t.bin")
    dumps = [r[1] for r in recs if r[0] == "hidden"]
    assert len(dumps) == 3
    for step, dump in enumerate(dumps):
        for layer in range(L):
            np.testing.assert_array_equal(dump[layer], _bf16(torch.full((D,), float(step * 10 + layer))))


def test_the_offload_layer_dumps_the_router_input_it_was_given(tmp_path, monkeypatch):
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
                                          device=torch.device("cpu"), hidden_size=8)
    cache.routing_trace.start([(6, 0)], [])
    topk = (torch.tensor([[0.75, 0.25]]), torch.tensor([[2, 1]], dtype=torch.int32))
    monkeypatch.setattr("freetoken.layers.moe.fused_topk", lambda **_: topk)
    monkeypatch.setattr(cache, "ensure_experts", lambda layer_id, ids: ids.copy_(torch.tensor([[5, 0]])))
    monkeypatch.setattr(cache, "copy_missing", lambda: None)
    monkeypatch.setattr("freetoken.moe.fused.fused_experts_decode_impl", lambda hidden_states, *a, **k: hidden_states.zero_())
    hidden = torch.randn(1, 8)
    want = _bf16(hidden[0].clone())
    layer.decode_forward(hidden, torch.randn(1, 4))
    cache.routing_trace.close()
    with open(tmp_path / "t.bin", "rb") as f:
        data = f.read()
    at = data.index(bytes([rt.HIDDEN]) + struct.pack("<HH", 1, 8))
    np.testing.assert_array_equal(np.frombuffer(data, np.int16, 8, at + 5), want)
