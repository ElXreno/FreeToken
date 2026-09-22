"""The ablation operator: what a cut removes, what it leaves, and which tables it refuses."""

from __future__ import annotations

import pytest
import torch
from freetoken.models.qwen3_5_moe.ablate import DirectionAblation
from safetensors.torch import save_file

HIDDEN = 16
LAYERS = 4


@pytest.fixture
def table(tmp_path):
    g = torch.Generator().manual_seed(0)
    directions = torch.randn(LAYERS, HIDDEN, generator=g)
    path = tmp_path / "direction.safetensors"
    save_file({"direction": directions}, str(path))
    return str(path), directions


def test_full_cut_leaves_nothing_along_the_direction(table):
    path, directions = table
    a = DirectionAblation(path, layer=2, alpha=1.0, hidden_size=HIDDEN)
    r = directions[2] / directions[2].norm()
    x = torch.randn(5, HIDDEN, dtype=torch.float64).float()
    assert torch.allclose(a.cut(x) @ r, torch.zeros(5), atol=1e-5)


def test_partial_cut_scales_the_component_and_keeps_the_rest(table):
    path, directions = table
    a = DirectionAblation(path, layer=1, alpha=0.6, hidden_size=HIDDEN)
    r = directions[1] / directions[1].norm()
    x = torch.randn(3, HIDDEN)
    out = a.cut(x)
    assert torch.allclose(out @ r, 0.4 * (x @ r), atol=1e-5)
    orth = x - (x @ r).unsqueeze(-1) * r
    assert torch.allclose(out - (out @ r).unsqueeze(-1) * r, orth, atol=1e-5)


def test_two_cuts_compound_below_alpha_one(table):
    # the decoder cuts the block boundary twice; the tuned strength depends on that compounding
    path, directions = table
    a = DirectionAblation(path, layer=0, alpha=0.6, hidden_size=HIDDEN)
    r = directions[0] / directions[0].norm()
    x = torch.randn(2, HIDDEN)
    assert torch.allclose(a.cut(a.cut(x)) @ r, 0.16 * (x @ r), atol=1e-5)


def test_in_place_cut_matches_the_allocating_one(table):
    # the decoder uses the in-place form to keep prefill inside VRAM; it has to be the same math
    path, _ = table
    a = DirectionAblation(path, layer=3, alpha=0.6, hidden_size=HIDDEN)
    x = torch.randn(6, HIDDEN)
    want = a.cut(x)
    got = a.cut_(x.clone())
    assert torch.allclose(got, want, atol=1e-6)


def test_cut_sum_matches_cutting_the_materialized_sum(table):
    path, _ = table
    a = DirectionAblation(path, layer=1, alpha=0.6, hidden_size=HIDDEN)
    lhs, rhs = torch.randn(4, HIDDEN), torch.randn(4, HIDDEN)
    want = a.cut(lhs + rhs)
    got = a.cut_sum_(lhs.clone(), rhs)
    assert torch.allclose(got + rhs, want, atol=1e-6)


def test_cut_sum_leaves_the_other_side_untouched(table):
    path, _ = table
    a = DirectionAblation(path, layer=1, alpha=1.0, hidden_size=HIDDEN)
    lhs, rhs = torch.randn(4, HIDDEN), torch.randn(4, HIDDEN)
    keep = rhs.clone()
    a.cut_sum_(lhs, rhs)
    assert torch.equal(rhs, keep)


def test_add_cut_accumulates_into_the_first_argument(table):
    path, _ = table
    a = DirectionAblation(path, layer=2, alpha=0.6, hidden_size=HIDDEN)
    lhs, rhs = torch.randn(3, HIDDEN), torch.randn(3, HIDDEN)
    want = a.cut(lhs + rhs)
    buf = lhs.clone()
    got = a.add_cut_(buf, rhs)
    assert got is buf
    assert torch.allclose(got, want, atol=1e-6)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("rows", [1, 2, 2048])
def test_kernel_matches_the_torch_arithmetic_that_was_measured(tmp_path, rows):
    # the in-place kernel replaced the out-of-place torch ops the refusal rate was measured
    # with; all three modes have to land on those within bf16 rounding
    hidden = 2048
    g = torch.Generator().manual_seed(1)
    path = tmp_path / "direction.safetensors"
    save_file({"direction": torch.randn(LAYERS, hidden, generator=g)}, str(path))
    a = DirectionAblation(str(path), layer=1, alpha=0.6, hidden_size=hidden)
    x = torch.randn(rows, hidden, generator=g).to("cuda", torch.bfloat16)
    y = torch.randn(rows, hidden, generator=g).to("cuda", torch.bfloat16)

    got = a.cut_(x.clone())
    torch.testing.assert_close(got, a.cut(x), atol=2e-2, rtol=2e-2)

    got = a.add_cut_(x.clone(), y)
    torch.testing.assert_close(got, a.cut(x + y), atol=2e-2, rtol=2e-2)

    keep = y.clone()
    got = a.cut_sum_(x.clone(), y)
    torch.testing.assert_close(got + y, a.cut(x + y), atol=3e-2, rtol=3e-2)
    assert torch.equal(y, keep)


@cuda
def test_kernel_leaves_the_right_share_of_the_component(tmp_path):
    hidden = 2048
    path = tmp_path / "direction.safetensors"
    save_file({"direction": torch.randn(LAYERS, hidden)}, str(path))
    a = DirectionAblation(str(path), layer=0, alpha=0.6, hidden_size=hidden)
    r = a._vector(torch.empty(0, device="cuda", dtype=torch.bfloat16)).float()
    x = torch.randn(64, hidden, device="cuda").to(torch.bfloat16)
    before = x.float() @ r
    after = a.cut_(x.clone()).float() @ r
    torch.testing.assert_close(after, 0.4 * before, atol=5e-2, rtol=5e-2)


@cuda
def test_norm_into_a_buffer_matches_the_plain_norm():
    from freetoken.layers import GemmaRMSNorm

    norm = GemmaRMSNorm(2048, eps=1e-6)
    norm.weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(7, 2048, device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(x)
    assert norm.forward_into(x, out) is out
    assert torch.equal(out, norm.forward(x))


def test_cut_follows_the_input_dtype(table):
    path, _ = table
    a = DirectionAblation(path, layer=0, alpha=1.0, hidden_size=HIDDEN)
    assert a.cut(torch.randn(2, HIDDEN, dtype=torch.bfloat16)).dtype == torch.bfloat16


def test_wrong_hidden_size_is_refused(table):
    path, _ = table
    with pytest.raises(ValueError, match="want \\[layers, 32\\]"):
        DirectionAblation(path, layer=0, alpha=1.0, hidden_size=32)


def test_layer_outside_the_table_is_refused(table):
    path, _ = table
    with pytest.raises(ValueError, match="outside the 4 rows"):
        DirectionAblation(path, layer=LAYERS, alpha=1.0, hidden_size=HIDDEN)


def test_missing_tensor_is_refused(tmp_path):
    path = tmp_path / "other.safetensors"
    save_file({"weights": torch.zeros(LAYERS, HIDDEN)}, str(path))
    with pytest.raises(ValueError, match="no 'direction' tensor"):
        DirectionAblation(str(path), layer=0, alpha=1.0, hidden_size=HIDDEN)


def test_disabled_config_builds_nothing():
    class Cfg:
        ablate_direction = None

    assert DirectionAblation.from_config(Cfg()) is None
