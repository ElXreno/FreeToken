"""A staggered MTP verify (row 1 half a layer behind row 0 on its own stream) equals the lockstep layers.

The mixer of row 1 reads what row 0's wrote, as attention reads row 0's KV and GDN its stash, and every routed call
goes through one scratch buffer, as the expert cache's staging and the CPU pool's one task do; row 0 and the routed
calls sleep on the GPU first, so a row that does not wait for them reads stale or clobbered values.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA required", allow_module_level=True)

from freetoken import core
from freetoken.core import Batch, Context, Req, SamplingParams
from freetoken.env import ENV
from freetoken.layers import GemmaPlusOneRMSNormFused
from freetoken.models.qwen3_5_moe.model import Qwen3_5DecoderLayer, Qwen3_5Model
from freetoken.models.qwen3_5_moe.moe import Qwen3_5MoE
from freetoken.models.qwen3_5_moe.stagger import can_stagger, staggered_forward

H, E, LAYERS = 256, 8, 4
SLEEP = 1_000_000


class _Linear:
    def __init__(self, w: torch.Tensor):
        self.weight = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() @ self.weight.float().T).to(x.dtype)


class _Mixer(_Linear):
    """Each row adds the projections of the rows before it, which it reads back from ``seen``."""

    def __init__(self, w: torch.Tensor):
        super().__init__(w)
        self.seen = torch.zeros(2, H, device="cuda")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row = core.get_global_ctx().batch.verify_row
        y = x.float() @ self.weight.float().T
        if row is None:
            return torch.stack([y[0], y[1] + y[0]]).to(x.dtype)
        if row == 0:
            torch.cuda._sleep(SLEEP)
        self.seen[row].copy_(y[0])
        return (y if row == 0 else y + self.seen[0]).to(x.dtype)


class _Experts:
    """Writes its result into a scratch shared by every layer and row, then reads it back."""

    scratch = None

    def __init__(self, w: torch.Tensor):
        self.w = w

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        n = hidden_states.shape[0]
        s = _Experts.scratch[:n]
        s.copy_((hidden_states.float() @ self.w.T) * router_logits.float().softmax(-1)[:, :1])
        torch.cuda._sleep(SLEEP)
        return s.to(hidden_states.dtype)


def _model(g: torch.Generator) -> Qwen3_5Model:
    def mat(rows, cols):
        return (torch.randn(rows, cols, generator=g) / cols**0.5).to("cuda", torch.bfloat16)

    def norm():
        n = GemmaPlusOneRMSNormFused(H, eps=1e-6)
        n.weight = (0.1 * torch.randn(H, generator=g)).to("cuda", torch.bfloat16)
        return n

    layers = []
    for i in range(LAYERS):
        mlp = Qwen3_5MoE.__new__(Qwen3_5MoE)
        mlp.gate = _Linear(mat(E, H))
        mlp.shared_expert = _Linear(mat(H, H))
        mlp.shared_expert_gate = _Linear(mat(1, H))
        mlp.experts = _Experts(mat(H, H).float())
        layer = Qwen3_5DecoderLayer.__new__(Qwen3_5DecoderLayer)
        layer._layer_id, layer._ablate, layer._writer_cut, layer._is_linear = i, None, None, True
        layer.linear_attn, layer.mlp = _Mixer(mat(H, H)), mlp
        layer.input_layernorm, layer.post_attention_layernorm = norm(), norm()
        layers.append(layer)
    model = Qwen3_5Model.__new__(Qwen3_5Model)
    model.layers = SimpleNamespace(op_list=layers)
    model.norm = norm()
    return model


def test_staggered_rows_match_lockstep(monkeypatch):
    g = torch.Generator().manual_seed(7)
    _Experts.scratch = torch.zeros(2, H, device="cuda")
    model = _model(g)
    ctx = Context(page_size=1)
    ctx.attn_backend = SimpleNamespace(supports_verify_rows=True, prepare_verify_row=lambda batch, rb: None)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    monkeypatch.setattr(ENV.STAGGER, "value", True)
    req = Req(input_ids=torch.zeros(9, dtype=torch.int32), table_idx=1, cached_len=7, output_len=4, uid=0,
              sampling_params=SamplingParams(), cache_handle=None)
    batch = Batch(reqs=[req], phase="verify", rows_per_req=2)
    batch.padded_reqs = batch.reqs
    batch.positions = torch.tensor([7, 8], dtype=torch.int32, device="cuda")
    batch.out_loc = torch.tensor([7, 8], dtype=torch.int32, device="cuda")
    x = torch.randn(2, H, generator=g).to("cuda", torch.bfloat16)

    with ctx.forward_batch(batch):
        assert can_stagger(batch, model.layers.op_list)
        hidden, residual = x.clone(), None
        for layer in model.layers.op_list:
            hidden, residual = layer.forward(hidden, residual)
        want = model.norm.forward(hidden, residual)[0].float()
        got = staggered_forward(model, x.clone()).float()
        assert ctx.batch is batch
    torch.testing.assert_close(got, want, atol=2e-2, rtol=2e-2)
