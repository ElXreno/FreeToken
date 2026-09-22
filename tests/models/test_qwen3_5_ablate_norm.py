"""The fused add + cut + (1 + w) RMSNorm launch matches ``add_cut_`` followed by the norm's ``gemma_rmsnorm``.

The norm half is computed as the Triton ``gemma_rmsnorm`` does; only the cut's dot product
may reduce in another order (the norm's warp count instead of the cut kernel's 4), so the
two may differ by one bf16 ulp in a few elements and never more.
"""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.layers import GemmaPlusOneRMSNormFused
from freetoken.models.qwen3_5_moe.ablate import DirectionAblation

H = 2048


def _ablation(tmp_path) -> DirectionAblation:
    from safetensors.torch import save_file

    path = tmp_path / "dir.safetensors"
    save_file({"direction": torch.randn(4, H)}, str(path))
    return DirectionAblation(str(path), layer=2, alpha=0.6, hidden_size=H)


def _ulps(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


@pytest.mark.parametrize("rows", [1, 2, 4, 1500])
def test_fused_add_cut_norm_matches_the_two_launches(tmp_path, rows: int):
    torch.manual_seed(rows)
    ablation = _ablation(tmp_path)
    norm = GemmaPlusOneRMSNormFused(H, eps=1e-6)
    norm.weight = torch.randn(H, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)
    hidden = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)

    ref_res, ref_hid = residual.clone(), hidden.clone()
    ablation.add_cut_(ref_res, ref_hid)
    norm.gemma_rmsnorm(ref_res, norm.weight, norm.eps, out=ref_hid)

    ablation.add_cut_norm_(residual, hidden, norm)
    for got, ref in ((residual, ref_res), (hidden, ref_hid)):
        diff = _ulps(got, ref)
        assert diff.max().item() <= 1
        assert (diff > 0).float().mean().item() < 0.01
