from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaPlusOneRMSNormFused,
    OPList,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from freetoken.layers.linear_attn import GatedDeltaNet
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin
from freetoken.utils import nvtx_annotate

from .ablate import DirectionAblation
from .attention import Qwen3_5Attention
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE
from .mtp import Qwen3_5MTPHead

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        prefix: str = "",
        ablate: DirectionAblation | None = None,
        writer_cut: DirectionAblation | None = None,
    ):
        self._layer_id = layer_id
        self._ablate = ablate
        self._writer_cut = writer_cut
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                output_gate=g.output_gate,
                quant_config=config.quant,
                prefix=f"{prefix}.linear_attn",
                # llama.cpp stores every qwen35 ssm_out with tiled value-head columns
                out_heads_tiled=config.quant is not None and config.quant.dialect == "gguf",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = GemmaPlusOneRMSNormFused(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaPlusOneRMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        if self._ablate is not None:
            return self._forward_ablated(hidden, residual)
        if self._writer_cut is not None:
            return self._forward_writers_cut(hidden, residual)
        # Residual-stream form: fuse each residual-add into the next RMSNorm so add + norm are one kernel per sublayer.
        hidden, residual = self.input_layernorm.forward(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual

    def _forward_ablated(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        """Cut the direction at the block input, after the mixer and after the MLP.

        The fused add+norm leaves no gap between the add and the norm, so the add is unrolled,
        but the buffers stay the fused path's: the stream accumulates in ``residual`` and the
        norm lands in ``hidden``. The caller still holds the old ``residual`` while the block
        runs, so accumulating anywhere else keeps a second chunk-sized stream alive, and with
        the norm's own output that is the 16 MiB a full prefill chunk did not have. The block
        input is also the previous block's output, so that point is cut twice; with alpha below
        1 the two compound, and that compounded strength is the one that was tuned.
        """
        a = self._ablate
        if residual is None:
            # as unablated: the embedding buffer becomes the stream, the norm gets a new one
            residual = a.cut_(hidden)
            hidden, _ = self.input_layernorm.forward(residual)
        else:
            a.add_cut_norm_(residual, hidden, self.input_layernorm)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        a.add_cut_norm_(residual, hidden, self.post_attention_layernorm)
        hidden = self.mlp.forward(hidden)
        return a.cut_sum_(hidden, residual), residual

    def _forward_writers_cut(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        """The fused path with this layer's row cut from the mixer's and the MLP's output before
        either reaches the stream; the block input is the previous layer's output, already cut."""
        hidden, residual = self.input_layernorm.forward(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        self._writer_cut.cut_(hidden)
        hidden, residual = self.post_attention_layernorm.forward(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return self._writer_cut.cut_(hidden), residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            quant_config=config.quant,
            prefix=f"{prefix}.embed_tokens",
        )
        ablate = DirectionAblation.from_config(config)
        writer_cuts = DirectionAblation.writers_from_config(config)
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(
                    config,
                    layer_id,
                    prefix=f"{prefix}.layers.{layer_id}",
                    ablate=ablate,
                    writer_cut=writer_cuts[layer_id] if writer_cuts else None,
                )
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaPlusOneRMSNormFused(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward(x, residual)
        return x


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        if config.mtp_draft:
            self.mtp = Qwen3_5MTPHead(config)
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """The MoE releases share the dense code path: the decoder picks the routed or dense MLP from config.num_experts."""


class Qwen3_5ForConditionalGeneration(QwenVLVisionMixin, Qwen3_5ForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, "Qwen3.5 consumes no DeepStack features"
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """The MoE releases with the vision tower; see Qwen3_5MoeForCausalLM."""


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
]
