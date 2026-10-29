from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.kernel.triton.moe_shared_gate import shared_gate_mul_add, shared_gate_sigmoid
from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearReplicated,
    LinearRowParallel,
    OffloadMoELayer,
    make_moe_layer,
    silu_and_mul,
)

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _SharedExpert(BaseOP):
    """Always-present shared SwiGLU expert of width ``shared_expert_intermediate_size``."""

    def __init__(
        self, config: ModelConfig, hidden_size: int, intermediate_size: int, *, prefix: str = ""
    ):
        self.gate_up_proj = LinearColParallelMerged(
            hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearRowParallel(
            intermediate_size, hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class Qwen3_5DenseMLP(_SharedExpert):
    """Dense (non-MoE) SwiGLU MLP for dense Qwen3.x checkpoints (e.g. 27B): ``gate_up_proj``
    (fused gate|up) + ``down_proj`` at full ``intermediate_size``. Same structure as the shared
    expert, so it reuses ``_SharedExpert`` directly and keeps the state-dict keys flat
    (``...layers.N.mlp.{gate_up_proj,down_proj}``)."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        super().__init__(config, config.hidden_size, config.intermediate_size, prefix=prefix)


_SIDE_STREAM_MAX_TOKENS = 16
_SIDE_STREAMS: dict[torch.device, torch.cuda.Stream] = {}


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _SIDE_STREAMS.get(device)
    if stream is None:
        stream = _SIDE_STREAMS[device] = torch.cuda.Stream(device=device)
    return stream


class Qwen3_5MoE(BaseOP):
    """Routed MoE (256 experts, top-8) plus a gated shared expert:

        out = routed(x) + sigmoid(shared_expert_gate(x)) * shared_expert(x)

    Router softmaxes over all experts, takes top-k, and renormalizes (HF semantics).
    """

    def __init__(self, config: ModelConfig, layer_id: int | None = None, *, prefix: str = ""):
        self.experts = make_moe_layer(
            config,
            layer_id=layer_id,
            renormalize=config.norm_topk_prob,
            quant_config=config.quant,
            prefix=f"{prefix}.experts",
        )
        # routers stay bf16 whatever the checkpoint quantizes
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            config, config.hidden_size, config.shared_expert_intermediate_size,
            prefix=f"{prefix}.shared_expert",
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        if (
            hidden_states.is_cuda
            and num_tokens <= _SIDE_STREAM_MAX_TOKENS
            and isinstance(self.experts, OffloadMoELayer)
        ):
            # a decode batch leaves most SMs idle: the shared expert runs beside the routed
            # chain (router, cache fill, fused MoE), which never writes ``hidden_states``
            main = torch.cuda.current_stream()
            side = _side_stream(hidden_states.device)
            side.wait_stream(main)
            with torch.cuda.stream(side):
                shared = self.shared_expert.forward(hidden_states)
                gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
            router_logits = self.gate.forward(hidden_states)
            routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
            main.wait_stream(side)
            return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)
        # Compute the router + shared expert BEFORE the routed experts: the fused MoE
        # kernel may write into ``hidden_states`` in place, which would corrupt the
        # shared expert's input (HF also evaluates the shared expert first).
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate).view(num_tokens, hidden_dim)

    def dense(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The block's part before the routed experts, in order on the current stream: (router logits, shared expert, its gate)."""
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        gate = shared_gate_sigmoid(hidden_states, self.shared_expert_gate.weight.view(-1))
        return router_logits, shared, gate

    def routed(self, hidden_states: torch.Tensor, dense: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """The routed experts over ``dense``'s router logits, merged with its shared expert: ``forward`` = ``routed(x, dense(x))``."""
        router_logits, shared, gate = dense
        routed = self.experts.forward(hidden_states=hidden_states, router_logits=router_logits)
        return shared_gate_mul_add(routed, shared, gate)


__all__ = ["Qwen3_5MoE", "Qwen3_5DenseMLP"]
