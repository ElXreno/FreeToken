"""The CPU MoE executor reads b12x (flashinfer SM12x W4A16) expert banks in place."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

FP8 = torch.float8_e4m3fn
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float64)


def _prepare_module(monkeypatch):
    fp = pytest.importorskip("flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_prepare")
    # the pack runs on the CPU here: scales go in unswizzled and no launch workspace is needed
    monkeypatch.setattr(fp, "unswizzle_expert_scales", lambda s, rows, cols: s)
    monkeypatch.setattr(fp, "_make_workspace", lambda device, max_blocks_per_sm: None)
    return fp


def _dequant(packed, scale, glob):
    codes = torch.stack([(packed & 0xF).long(), (packed >> 4).long()], -1).flatten(-2)
    return E2M1[codes] * scale.double().repeat_interleave(16, -1) * glob.double()[..., None]


def _b12x_cache(fp, L, E, H, I, seed):
    """b12x banks packed the way B12xNvfp4MoEKernel.pack does, plus the dense weights they encode."""
    torch.manual_seed(seed)
    gu_banks, gus_banks, dn_banks, dns_banks, gu_alpha, dn_alpha, dense = [], [], [], [], [], [], []
    for _ in range(L):
        gu = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)
        gus = (0.5 + torch.rand(E, 2 * I, H // 16)).to(FP8)
        gug = (0.002 + 0.002 * torch.rand(E, 2)).repeat_interleave(I, 1)
        dn = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)
        dns = (0.5 + torch.rand(E, H, I // 16)).to(FP8)
        dng = 0.002 + 0.002 * torch.rand(E)
        g_max = gug.max(1, keepdim=True).values
        folded = (gus.to(torch.float16).float() * (gug / g_max).unsqueeze(-1)).to(torch.float16)
        prep = fp._prepare_w4a16_packed_weights(
            torch.cat([gu[:, I:], gu[:, :I]], 1).contiguous(), torch.cat([folded[:, I:], folded[:, :I]], 1).to(FP8),
            g_max.squeeze(1), dn, dns, dng, activation="silu", params_dtype=torch.bfloat16, source_format="modelopt",
        )
        gu_banks.append(prep.w13.view(E, H // 16, 4 * I))
        gus_banks.append(prep.w13_scale.view(E, H // 16, 2 * I))
        dn_banks.append(prep.w2.view(E, I // 16, 2 * H))
        dns_banks.append(prep.w2_scale.view(E, I // 16, H))
        gu_alpha.append(prep.w13_global_scale.float())
        dn_alpha.append(prep.w2_global_scale.float())
        dense.append((_dequant(gu, folded.to(FP8), g_max.expand(E, 2 * I)), _dequant(dn, dns, dng[:, None].expand(E, H))))
    cache = SimpleNamespace(
        quant_format="nvfp4_b12x",
        bank_sources={"gate_up": gu_banks, "gate_up_scale": gus_banks, "down": dn_banks, "down_scale": dns_banks},
        gate_up_alpha=torch.cat(gu_alpha), down_alpha=torch.cat(dn_alpha),
        num_layers=L, num_experts=E, decode_target="cpu", cpu_executor=None,
    )
    return cache, dense


def _reference(dense, layer, x, ids, w, I):
    gu_w, dn_w = dense[layer]
    out = torch.zeros(x.shape[0], dn_w.shape[1], dtype=torch.float64)
    for t in range(x.shape[0]):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            gu = gu_w[e] @ x[t].double()
            inter = torch.nn.functional.silu(gu[:I]) * gu[I:]
            inter = inter.to(torch.bfloat16).double()
            out[t] += float(w[t, k]) * (dn_w[e] @ inter)
    return out


@pytest.mark.parametrize("bs", [1, 3])
@pytest.mark.parametrize("path", ["default", "scalar"])
def test_cpu_b12x_matches_the_dense_weights_it_packs(monkeypatch, bs, path):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    fp = _prepare_module(monkeypatch)
    if path == "scalar":
        monkeypatch.setenv("FREETOKEN_CPU_MOE_NO_VNNI", "1")
    L, E, H, I, top_k = 2, 6, 512, 256, 3
    cache, dense = _b12x_cache(fp, L, E, H, I, seed=bs)
    ex = CpuMoeExecutor(cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False,
                        num_threads=2, max_tokens=bs, device=torch.device("cpu"))
    assert "b12x" in ex.isa
    x = torch.randn(bs, H, dtype=torch.bfloat16)
    ids = torch.stack([torch.randperm(E)[:top_k] for _ in range(bs)]).to(torch.int32)
    w = torch.rand(bs, top_k)
    layer = 1
    ex._io["x"][:bs].copy_(x)
    ex._io["ids"][:bs].copy_(ids)
    ex._io["w"][:bs].copy_(w)
    ex._ext.run(layer, bs)
    got = ex._io["y"][:bs].double()
    ref = _reference(dense, layer, x, ids, w, I)
    rel = ((got - ref).abs().max() / ref.abs().max()).item()
    assert rel < (5e-3 if path == "scalar" else 3e-2), f"{ex.isa} bs={bs} rel err {rel}"


def test_cpu_b12x_simd_activation_quant_is_the_scalar_one_bitwise():
    """The AVX2 activation quant writes the scalar quant's bytes and scales: ties round away from zero, and zero,
    denormal, NaN and infinite inputs land where the scalar one puts them."""
    from freetoken.kernel import _cpu_moe

    torch.manual_seed(7)
    K = 3072
    x = torch.randn(K) * torch.logspace(-3, 3, K)[torch.randperm(K)]
    # a block whose largest value is 127 quantizes with scale 1, so each k + 0.5 is a tie
    halves = torch.arange(-127, 127).float() + 0.5
    halves = halves[: len(halves) // 15 * 15].view(-1, 15)
    tie = torch.cat([torch.full((halves.shape[0], 1), 127.0), halves], 1).flatten()
    special = torch.tensor([0.0] * 16 + [-0.0] * 16 + [1e-39, -1e-39] * 8 + [float("nan")] + [1.0] * 15
                           + [float("inf")] + [2.0] * 15 + [float("-inf"), float("nan")] * 8)
    xs = torch.cat([x, tie, special]).to(torch.bfloat16)
    ref = _cpu_moe.b12x_quant_act(xs, False)
    got = _cpu_moe.b12x_quant_act(xs, True)
    assert torch.equal(got[0], ref[0])
    assert torch.equal(got[1].view(torch.int32), ref[1].view(torch.int32))
    order = [i for j in range(4) for i in (2 * j, 2 * j + 1, 2 * j + 8, 2 * j + 9)]
    blocks = ref[0].view(torch.int8).view(-1, 16)
    q = torch.empty_like(blocks)
    q[:, order] = blocks
    first = K // 16
    q_tie = q[first:first + halves.shape[0]].double()
    v = tie.view(-1, 16).double()
    assert torch.equal(q_tie, torch.sign(v) * torch.floor(v.abs() + 0.5))
    assert not torch.equal(q_tie, torch.round(v))


@pytest.mark.parametrize(
    "env",
    [
        {"FLOW": "1", "FLOW_P2": "2"},
        {"FLOW": "2", "FLOW_P2": "3", "SPLIT_PREP": "1"},
        {"STATIC": "1", "SPLIT_PREP": "1"},
    ],
)
def test_cpu_b12x_work_splits_write_the_barrier_passes_bytes(monkeypatch, env):
    """The flow queue (gate_up and down items without barriers), the input prep on the workers and the static split
    write the default passes' bytes, a token without routes included."""
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    fp = _prepare_module(monkeypatch)
    L, E, H, I, top_k, bs = 2, 6, 512, 256, 3, 3
    cache, _ = _b12x_cache(fp, L, E, H, I, seed=11)

    def make():
        return CpuMoeExecutor(cache, top_k=top_k, activation="silu", apply_router_weight_on_input=False, num_threads=3,
                              max_tokens=bs, device=torch.device("cpu"))

    base = make()
    for k, v in env.items():
        monkeypatch.setenv(f"FREETOKEN_CPU_MOE_{k}", v)
    split = make()
    x = torch.randn(bs, H, dtype=torch.bfloat16)
    ids = torch.tensor([[0, -1, 4], [4, 2, -1], [-1, -1, -1]], dtype=torch.int32)
    w = torch.rand(bs, top_k)
    outs = []
    for ex in (base, split):
        ex._io["x"][:bs].copy_(x)
        ex._io["ids"][:bs].copy_(ids)
        ex._io["w"][:bs].copy_(w)
        ex._io["y"].fill_(1.0)
        ex._ext.profile_start(1)
        ex._ext.run(1, bs)
        outs.append((ex._io["y"][:bs].clone(), ex._ext.profile_take()))
    assert torch.equal(outs[0][0].view(torch.int16), outs[1][0].view(torch.int16))
    assert not outs[1][0][2].any()
    task, nids, thread, n = split._ext.profile_layout()
    per = outs[1][1][0, len(task) + nids:].view(n, len(thread))
    th = {name: per[:, i] for i, name in enumerate(thread)}
    if "FLOW" in env:
        assert (th["pass1_end"] == th["barrier2"]).all()
        assert int(th["pass1_items"].sum()) == 4 * (I // 64 // int(env["FLOW"]))
        assert int(th["pass2_items"].sum()) == 4 * -(-(H // 64) // int(env["FLOW_P2"]))
    if "STATIC" in env:
        assert th["pass1_items"].tolist() == [4 * len(range(t, I // 64, n)) for t in range(n)]
