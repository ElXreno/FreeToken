"""The GDN prefill chunk op that drops its inputs early matches the plain op bit for bit and peaks lower."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

HK, HV, D, SLOTS = 16, 32, 128, 4


def _inputs(total: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, total, HK, D, device="cuda", generator=gen).bfloat16()
    k = torch.randn(1, total, HK, D, device="cuda", generator=gen).bfloat16()
    v = torch.randn(1, total, HV, D, device="cuda", generator=gen).bfloat16()
    g = -torch.rand(1, total, HV, device="cuda", generator=gen) * 0.1
    beta = torch.rand(1, total, HV, device="cuda", generator=gen)
    state = torch.randn(SLOTS, HV, D, D, device="cuda", generator=gen) * 0.1
    return q, k, v, g, beta, state


@pytest.mark.parametrize("lens", [(4096,), (1500, 2596), (700, 64, 3332)])
def test_owned_prefill_matches_the_plain_op_and_peaks_lower(lens):
    from freetoken.models.qwen3_5_moe.gdn_kernels import gdn_prefill_chunk_fla, gdn_prefill_chunk_fla_owned

    total = sum(lens)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda", dtype=torch.int64)
    idx = torch.arange(1, len(lens) + 1, device="cuda", dtype=torch.int32)
    q, k, v, g, beta, state = _inputs(total, seed=len(lens))
    ref_state, own_state = state.clone(), state.clone()
    kw = dict(indices=idx, cu_seqlens=cu, scale=D ** -0.5, return_h=True)

    held = [q.clone(), k.clone(), v.clone()]
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    ref_o, ref_h = gdn_prefill_chunk_fla(*held, g, beta, state_source=ref_state, **kw)
    torch.cuda.synchronize()
    ref_peak = torch.cuda.max_memory_allocated() - base
    del held

    owned = [q.clone(), k.clone(), v.clone()]
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    own_o, own_h = gdn_prefill_chunk_fla_owned(owned, g, beta, state_source=own_state, **kw)
    torch.cuda.synchronize()
    own_peak = torch.cuda.max_memory_allocated() - base

    assert owned == []
    assert torch.equal(own_o, ref_o)
    assert torch.equal(own_h, ref_h)
    assert torch.equal(own_state, ref_state)
    # the raw q/k and v alone are this much; the early drops must win at least that back
    released = (q.numel() + k.numel() + v.numel()) * q.element_size()
    assert own_peak <= ref_peak - released, (own_peak / 2**20, ref_peak / 2**20)
