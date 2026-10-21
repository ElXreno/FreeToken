"""Gated shared-expert epilogue: ``routed + sigmoid(h . w) * shared`` in two launches matches
the unfused bf16 ops to bf16 rounding."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid


@pytest.mark.parametrize("tokens", [1, 2, 4, 300])
def test_matches_the_unfused_ops(tokens: int):
    torch.manual_seed(tokens)
    h = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1, 2048, device="cuda", dtype=torch.bfloat16) / 40
    shared = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
    routed = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)

    ref = routed.float() + torch.sigmoid(F.linear(h.float(), w.float())) * shared.float()
    out = shared_gate_mul_add(routed, shared, shared_gate_sigmoid(h, w.view(-1)))
    assert out.dtype == torch.bfloat16
    assert ((out.float() - ref).abs() / ref.abs().clamp(min=1e-2)).max().item() < 1e-2
