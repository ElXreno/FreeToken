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
