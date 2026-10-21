"""bf16 decode-batch linear: matches the fp32 reference, and a token row's output is the same
bits whether it runs alone or next to other rows."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.kernel.triton.bf16_rows_linear import rows_linear, rows_linear_ok

DEV = "cuda"
BF16 = torch.bfloat16


@pytest.mark.parametrize("N", [1, 64, 256, 1000])
@pytest.mark.parametrize("K", [2048, 6112])
@pytest.mark.parametrize("M", [1, 2, 3, 4])
def test_matches_fp32_reference(M: int, N: int, K: int):
    torch.manual_seed(M * 7 + N)
    x = torch.randn(M, K, device=DEV, dtype=BF16)
    w = torch.randn(N, K, device=DEV, dtype=BF16) / 40
    y = rows_linear(x, w)
    ref = x.float() @ w.float().t()
    rel = ((y.float() - ref).abs().max() / ref.abs().max()).item()
    assert rel < 1e-2, rel


@pytest.mark.parametrize("M", [2, 3, 4])
def test_row_output_does_not_depend_on_the_batch(M: int):
    x = torch.randn(M, 2048, device=DEV, dtype=BF16)
    w = torch.randn(256, 2048, device=DEV, dtype=BF16) / 40
    batched = rows_linear(x, w)
    for i in range(M):
        assert torch.equal(rows_linear(x[i : i + 1], w), batched[i : i + 1])


def test_dispatch_bounds():
    w = torch.randn(64, 2048, device=DEV, dtype=BF16)
    x = torch.randn(2, 2048, device=DEV, dtype=BF16)
    assert rows_linear_ok(x, w, None)
    assert not rows_linear_ok(torch.randn(5, 2048, device=DEV, dtype=BF16), w, None)
    assert not rows_linear_ok(x, w, torch.zeros(64, device=DEV, dtype=BF16))
    assert not rows_linear_ok(x.half(), w.half(), None)
    assert not rows_linear_ok(x, torch.randn(2048, 2048, device=DEV, dtype=BF16), None)
    assert not rows_linear_ok(torch.randn(2, 4096, device=DEV, dtype=BF16)[:, ::2], w, None)


def test_bf16_linear_layer_takes_the_rows_kernel_for_decode_batches():
    from types import SimpleNamespace

    from freetoken.layers.quantization.linear.unquantized import TorchLinearKernel

    layer = SimpleNamespace(weight=torch.randn(256, 2048, device=DEV, dtype=BF16) / 40, bias=None)
    x = torch.randn(2, 2048, device=DEV, dtype=BF16)
    kernel = TorchLinearKernel.__new__(TorchLinearKernel)
    assert torch.equal(kernel.apply(layer, x), rows_linear(x, layer.weight))
    big = torch.randn(64, 2048, device=DEV, dtype=BF16)
    assert torch.equal(kernel.apply(layer, big), torch.nn.functional.linear(big, layer.weight))
