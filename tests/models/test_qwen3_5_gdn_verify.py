"""MTP verify through a GDN layer: the one-launch path (two-token conv + two-token recurrent
update) must leave the same bits in the output and in both state slots as the two ordered
single-token passes it replaces, and touch no other slot."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kernel.backend import is_sgl_kernel_installed
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet

if is_sgl_kernel_installed():  # pragma: no cover
    pytest.skip("the one-launch verify conv is Triton-only", allow_module_level=True)

DEV = "cuda"
HK, DK, HV, DV, WIDTH, SLOTS = 16, 128, 32, 128, 4, 9


def _layer() -> Qwen3_5GatedDeltaNet:
    layer = Qwen3_5GatedDeltaNet.__new__(Qwen3_5GatedDeltaNet)
    layer.num_k_heads, layer.head_k_dim = HK, DK
    layer.num_v_heads, layer.head_v_dim = HV, DV
    layer.key_dim, layer.value_dim = HK * DK, HV * DV
    conv_dim = 2 * HK * DK + HV * DV
    layer.conv1d = SimpleNamespace(weight=torch.randn(conv_dim, 1, WIDTH, device=DEV, dtype=torch.bfloat16) / 2)
    layer.A_log = torch.randn(HV, device=DEV) / 4
    layer.dt_bias = torch.randn(HV, device=DEV) / 4
    return layer


def _pool(conv_dim: int):
    conv = torch.randn(1, SLOTS, conv_dim, WIDTH - 1, device=DEV, dtype=torch.bfloat16)
    rec = torch.randn(1, SLOTS, HV, DK, DV, device=DEV) / 8
    return SimpleNamespace(conv_states=conv, recurrent_states=rec)


@pytest.mark.parametrize("requests", [1, 2])
def test_one_launch_verify_matches_two_passes(requests: int):
    torch.manual_seed(requests)
    layer = _layer()
    conv_dim = layer.conv1d.weight.shape[0]
    conv_in = torch.randn(2 * requests, conv_dim, device=DEV, dtype=torch.bfloat16)
    a = torch.randn(2 * requests, HV, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(2 * requests, HV, device=DEV, dtype=torch.bfloat16)
    slot_in = torch.tensor([1, 4][:requests], dtype=torch.int32, device=DEV)
    slot_out = torch.tensor([6, 2][:requests], dtype=torch.int32, device=DEV)
    fla = SimpleNamespace(cache_indices=slot_in, verify_out_indices=slot_out)

    ref_pool, new_pool = _pool(conv_dim), None
    new_pool = SimpleNamespace(
        conv_states=ref_pool.conv_states.clone(), recurrent_states=ref_pool.recurrent_states.clone()
    )
    ref = layer._decode_verify_two_pass(conv_in, a, b, ref_pool, 0, fla, torch.bfloat16)
    out = layer._decode_verify(conv_in, a, b, new_pool, 0, fla, torch.bfloat16)

    assert torch.equal(out, ref)
    assert torch.equal(new_pool.conv_states, ref_pool.conv_states)
    assert torch.equal(new_pool.recurrent_states, ref_pool.recurrent_states)
