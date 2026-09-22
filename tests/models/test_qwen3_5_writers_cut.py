"""``--ablate-mode writers`` in the decoder equals running the layers with orthogonalized writers."""

from __future__ import annotations

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken.layers import GemmaPlusOneRMSNormFused
from freetoken.models.qwen3_5_moe.ablate import DirectionAblation
from freetoken.models.qwen3_5_moe.model import Qwen3_5DecoderLayer
from safetensors.torch import save_file

H, FF, LAYERS, ALPHA = 512, 96, 2, 1.25


class _Linear:
    def __init__(self, w: torch.Tensor):
        self.w = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() @ self.w.T).to(x.dtype)


class _MLP:
    def __init__(self, up: torch.Tensor, down: torch.Tensor):
        self.up, self.down = up, down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.nn.functional.silu(x.float() @ self.up.T) @ self.down.T).to(x.dtype)


def _layer(layer_id, mix, up, down, norms, writer_cut):
    layer = Qwen3_5DecoderLayer.__new__(Qwen3_5DecoderLayer)
    layer._layer_id = layer_id
    layer._ablate = None
    layer._writer_cut = writer_cut
    layer._is_linear = True
    layer.linear_attn = _Linear(mix)
    layer.mlp = _MLP(up, down)
    layer.input_layernorm, layer.post_attention_layernorm = norms
    return layer


def test_writers_cut_matches_orthogonalized_weights(tmp_path):
    g = torch.Generator().manual_seed(3)
    directions = torch.randn(LAYERS, H, generator=g)
    path = tmp_path / "dir.safetensors"
    save_file({"direction": directions}, str(path))
    cuts = DirectionAblation.per_layer(str(path), alpha=ALPHA, hidden_size=H, num_layers=LAYERS)

    cut_layers, edited_layers, plain_layers = [], [], []
    for i in range(LAYERS):
        r = (directions[i] / directions[i].norm()).cuda()
        mix = torch.randn(H, H, generator=g).cuda() / H**0.5
        up = torch.randn(FF, H, generator=g).cuda() / H**0.5
        down = torch.randn(H, FF, generator=g).cuda() / FF**0.5
        norms = []
        for _ in range(2):
            n = GemmaPlusOneRMSNormFused(H, eps=1e-6)
            n.weight = (0.1 * torch.randn(H, generator=g)).to("cuda", torch.bfloat16)
            norms.append(n)
        orth = lambda w, r=r: w - ALPHA * torch.outer(r, r @ w)
        cut_layers.append(_layer(i, mix, up, down, norms, cuts[i]))
        edited_layers.append(_layer(i, orth(mix), up, orth(down), norms, None))
        plain_layers.append(_layer(i, mix, up, down, norms, None))

    x0 = torch.randn(7, H, generator=g).to("cuda", torch.bfloat16)
    outs = []
    for layers in (cut_layers, edited_layers, plain_layers):
        hidden, residual = x0.clone(), None
        for layer in layers:
            hidden, residual = layer.forward(hidden, residual)
        outs.append((hidden + residual).float())
    torch.testing.assert_close(outs[0], outs[1], atol=6e-2, rtol=3e-2)
    assert (outs[0] - outs[2]).abs().max().item() > 0.5
