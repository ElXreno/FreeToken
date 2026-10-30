"""Qwen3.5-family MTP draft: the checkpoint's own ``mtp.*`` next-token layer, built from the target's layer classes."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING

import torch

from freetoken.layers import (
    GemmaPlusOneRMSNorm,
    GemmaPlusOneRMSNormFused,
    LinearColParallelMerged,
    LinearReplicated,
    OPList,
    make_moe_layer,
    silu_and_mul,
)
from freetoken.layers.base import BaseOP
from freetoken.layers.quantization import finalize_quant
from freetoken.layers.rotary import get_rope
from freetoken.models.config import FullAttentionGroupConfig
from freetoken.utils import nvtx_annotate, torch_dtype

from .ablate import DirectionAblation
from .model import Qwen3_5DecoderLayer
from .weight import iter_mtp_expert_pieces, iter_mtp_weights

if TYPE_CHECKING:
    from freetoken.layers.embedding import VocabParallelEmbedding
    from freetoken.models.config import ModelConfig


def draft_config(config: ModelConfig) -> ModelConfig:
    """One full-attention layer with the target's geometry."""
    groups = [g for g in config.attention_groups if isinstance(g, FullAttentionGroupConfig)]
    group = groups[0] if groups else config.default_full_attention_group()
    return dataclasses.replace(
        config,
        num_layers=1,
        attention_groups=(dataclasses.replace(group, layer_ids=(0,)),),
    )


class Qwen3_5MTPDraft(BaseOP):
    """``h' = layer(fc(cat(norm_e(embed(token)), norm_h(h))))``: the next token's hidden from a token and the hidden before it."""

    # the draft MTPProposer picks from the target's backends
    attention_backend = None
    draft_config = staticmethod(draft_config)

    @staticmethod
    def relay_width(config: ModelConfig) -> int:
        return config.hidden_size

    @staticmethod
    def check_source(path: str, target_path: str) -> None:
        from freetoken.models.gguf.reader import gguf_architecture, gguf_tensor_index, is_gguf_path, load_gguf_metadata
        from freetoken.models.loader import safetensors_weight_map

        if is_gguf_path(path):
            if not is_gguf_path(target_path):
                raise ValueError(f"the Qwen3.5 MTP draft loads from safetensors mtp.* tensors, not from the GGUF {path}")
            from .gguf import nextn_block_index

            blk = nextn_block_index(path)
            index = gguf_tensor_index(path)
            if f"blk.{blk}.nextn.eh_proj.weight" not in index:
                raise ValueError(f"{path}: blk.{blk} carries no nextn tensors")
            if os.path.abspath(path) != os.path.abspath(target_path):
                # a separate draft file must be an mtp-*.gguf sidecar: only the MTP block, same
                # arch; anything else is another trunk read under this target's shapes
                blocks = {name.split(".")[1] for name in index if name.startswith("blk.")}
                if blocks != {str(blk)} or gguf_architecture(path) != gguf_architecture(target_path):
                    raise ValueError(
                        f"the MTP draft of a GGUF target is the trunk's own nextn block or an "
                        f"mtp-*.gguf sidecar of the same model, not {path}"
                    )
                # the same arch string spans sizes (qwen35 covers the 9B and the 27B); a wrong-size
                # sidecar would otherwise die on a bare shape assert deep in load_state_dict
                side_hidden = load_gguf_metadata(path).get(f"{gguf_architecture(path)}.embedding_length")
                target_hidden = load_gguf_metadata(target_path).get(f"{gguf_architecture(target_path)}.embedding_length")
                if side_hidden is not None and target_hidden is not None and int(side_hidden) != int(target_hidden):
                    raise ValueError(
                        f"{path}: an mtp-*.gguf sidecar of a model with hidden size {int(side_hidden)}, but the "
                        f"target {target_path} has hidden size {int(target_hidden)}; the sidecar must be the "
                        f"MTP layer of the same model as the target"
                    )
        elif is_gguf_path(target_path):
            # a folder draft would build under the trunk's GGUF quant names
            raise ValueError(f"a GGUF target's MTP draft is the trunk's own nextn block, not {path}")
        elif os.path.isdir(path) and not any(name.startswith("mtp.") for name in safetensors_weight_map(path)):
            raise ValueError(f"{path} has no mtp.* tensors")

    def __init__(self, config: ModelConfig, target_model) -> None:
        self.config = draft_config(config)
        hidden = config.hidden_size
        self.fc = LinearReplicated(2 * hidden, hidden, has_bias=False, quant_config=config.quant, prefix="mtp.fc")
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(hidden, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(hidden, eps=config.rms_norm_eps)
        self.layers = OPList([Qwen3_5DecoderLayer(self.config, 0, prefix="mtp.layers.0")])
        self.norm = GemmaPlusOneRMSNormFused(hidden, eps=config.rms_norm_eps)
        # the target's own module; an underscore name keeps it out of this module's state dict
        self._embed_tokens = target_model.model.embed_tokens

    @classmethod
    def load(cls, config: ModelConfig, target_model, model_path: str, device: torch.device, dtype: torch.dtype) -> Qwen3_5MTPDraft:
        from freetoken.models.gguf.reader import is_gguf_path

        # a GGUF trunk keeps the draft in its own ggml types, mapped by the target config's own to_gguf;
        # an mtp-*.gguf sidecar carries a different mix, so its table names the draft's types
        gguf = is_gguf_path(model_path)
        if gguf:
            from .gguf import nextn_quant

            config = dataclasses.replace(config, quant=nextn_quant(model_path))
        with torch.device("meta"), torch_dtype(dtype):
            draft = cls(config, target_model)
        expected = draft.state_dict()
        loaded = {}
        if gguf:
            from .gguf import iter_nextn_weights

            weights = iter_nextn_weights(model_path, draft.config)
        else:
            weights = iter_mtp_weights(model_path, device)
        for name, tensor in weights:
            if name not in expected:
                raise ValueError(f"{model_path}: unexpected MTP tensor {name!r}")
            loaded[name] = tensor.to(device=device, dtype=expected[name].dtype)
        if missing := sorted(set(expected) - set(loaded)):
            raise ValueError(f"{model_path}: MTP tensors missing: {missing}")
        draft.load_state_dict(loaded)
        if draft.config.moe_enabled:
            from freetoken.moe.expert_banks import attach_resident_banks, build_expert_banks

            # packed by the layer's own expert method, as the target's resident layers are
            moe = draft.layers.op_list[0].mlp.experts
            if gguf:
                from .gguf import nextn_expert_banks

                banks = nextn_expert_banks(model_path, device)
            else:
                pieces = iter_mtp_expert_pieces(model_path)
                banks = build_expert_banks(moe.quant_method, 1, pieces, device=device, resident=True)
            attach_resident_banks([moe], banks)
        # not the root: it reaches the target's embedding, which the target finalized already
        finalize_quant(draft.fc)
        finalize_quant(draft.layers)
        return draft

    @property
    def embed_tokens(self) -> VocabParallelEmbedding:
        return self._embed_tokens

    def forward_hidden(self, embeds: torch.Tensor, prev_hidden: torch.Tensor) -> torch.Tensor:
        """Draft hidden of each row from the embedding of its token and the hidden of the position before it."""
        x = self.fc.forward(torch.cat([self.pre_fc_norm_embedding.forward(embeds), self.pre_fc_norm_hidden.forward(prev_hidden)], dim=-1))
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward(x, residual)
        return x

    def head_input(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden


class _HeadExpertQuant:
    """Quant config view binding the head's experts to the model's routed-expert scheme.

    ModelOpt excluded the head, so the checkpoint calls its experts bf16; the loader quantizes
    them to NVFP4 to share the offload cache's banks, and the layer has to bind the method that
    reads those banks. Only the scheme lookup is redirected -- the layer keeps its own prefix.
    """

    def __init__(self, quant, model_experts_prefix: str) -> None:
        self._quant = quant
        self._prefix = model_experts_prefix

    def get_quant_method(self, layer, prefix: str):
        return self._quant.get_quant_method(layer, self._prefix)


class _MTPMlp(BaseOP):
    """The head's MoE block: ``mtp_top_k`` routed experts plus the shared one.

    The checkpoint ships the head unquantized (``exclude_modules: ["mtp*"]``, 1536 MiB for
    256 experts), so the loader quantizes its experts to NVFP4 and they ride the offload
    cache's banks as one extra bank layer. ``mtp_top_k == 0`` drops them and keeps only the
    shared expert, which costs 0.14 acceptance; ``gate`` is loaded either way so the state
    dict stays complete.
    """

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.experts = (
            make_moe_layer(
                config,
                layer_id=config.mtp_bank_layer,
                top_k=config.mtp_top_k,
                renormalize=config.norm_topk_prob,
                quant_config=_HeadExpertQuant(config.quant, "model.layers.0.mlp.experts"),
                prefix=f"{prefix}.experts",
            )
            if config.num_mtp_moe_layers
            else None
        )
        self.shared_expert = _MTPSharedExpert(
            config, config.shared_expert_intermediate_size, prefix=f"{prefix}.shared_expert"
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # router and shared branch first: the fused expert kernel may write into ``x``
        router_logits = None if self.experts is None else self.gate.forward(x)
        shared = self.shared_expert.forward(x) * torch.sigmoid(self.shared_expert_gate.forward(x))
        if self.experts is None:
            return shared
        return self.experts.forward(hidden_states=x, router_logits=router_logits) + shared


class _MTPSharedExpert(BaseOP):
    def __init__(self, config: ModelConfig, intermediate_size: int, *, prefix: str = ""):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size, [intermediate_size, intermediate_size], has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = LinearReplicated(
            intermediate_size, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class _MTPAttention(BaseOP):
    """Same projections as ``Qwen3_5Attention``; the ring lives in the head, not a KV pool."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        head_dim = config.head_dim
        self.num_q = config.num_qo_heads
        self.num_kv = config.num_kv_heads
        self.head_dim = head_dim
        self.qo_attn_dim = self.num_q * head_dim
        self.kv_attn_dim = self.num_kv * head_dim
        self._qkv_split = [self.qo_attn_dim * 2, self.kv_attn_dim, self.kv_attn_dim]
        self.qkv_proj = LinearColParallelMerged(
            config.hidden_size, self._qkv_split, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.qkv_proj",
        )
        self.q_norm = GemmaPlusOneRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaPlusOneRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.o_proj = LinearReplicated(
            self.qo_attn_dim, config.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.o_proj",
        )
        self._rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=config.rotary_config.rotary_dim,
            max_position=config.rotary_config.max_position,
            base=config.rotary_config.base,
            rope_scaling=(
                tuple(config.rotary_config.scaling.items()) if config.rotary_config.scaling else None
            ),
        )

    def project(self, x: torch.Tensor, positions: torch.Tensor):
        """``(q, k, v, gate)`` for the ring: q/k post qk-norm and rope, all flattened per token."""
        qkv = self.qkv_proj.forward(x)
        qg, k, v = torch.split(qkv, self._qkv_split, dim=-1)
        qg = qg.view(-1, self.num_q, self.head_dim * 2)
        q = qg[..., : self.head_dim].contiguous()
        gate = qg[..., self.head_dim :].reshape(-1, self.qo_attn_dim)
        k = k.view(-1, self.num_kv, self.head_dim).contiguous()
        q = self.q_norm.forward(q).reshape(-1, self.qo_attn_dim)
        k = self.k_norm.forward(k).reshape(-1, self.kv_attn_dim)
        q, k = self._rotary.forward(positions, q, k)
        return q, k, v.contiguous(), gate


class Qwen3_5MTPLayer(BaseOP):
    """One gated-attention decoder layer, attending over a bounded ring instead of a KV cache."""

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        self.self_attn = _MTPAttention(config, prefix=f"{prefix}.self_attn")
        self.input_layernorm = GemmaPlusOneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaPlusOneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = _MTPMlp(config, prefix=f"{prefix}.mlp")


class Qwen3_5MTPHead(BaseOP):
    """Multi-token-prediction draft head: ``fc(cat(norm(embed(x_next)), norm(h)))`` through one
    decoder layer, then the base model's lm_head.

    EAGLE alignment: slot ``i`` takes token ``x_{i+1}`` with hidden ``h_i`` at position ``p_i``
    and predicts ``x_{i+2}``. Its layer attends over a per-slot ring of the last ``mtp_window``
    tokens, so the head needs no KV-cache layer and leaves the paged cache, the prefix cache and
    the arena format untouched.
    """

    def __init__(self, config: ModelConfig, *, prefix: str = "mtp"):
        assert config.mtp_num_layers == 1, f"mtp_num_layers={config.mtp_num_layers} is not supported"
        assert not config.mtp_dedicated_embeddings, "the head with its own embedding table is not supported"
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaPlusOneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = LinearReplicated(2 * config.hidden_size, config.hidden_size, has_bias=False)
        self.layers = OPList([Qwen3_5MTPLayer(config, prefix=f"{prefix}.layers.0")])
        self.norm = GemmaPlusOneRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # the head writes the stream too; left alone it keeps drafting what the body no longer says
        self._ablate = DirectionAblation.from_config(config)
        self._window = int(config.mtp_window)
        self._k_ring: torch.Tensor | None = None
        self._v_ring: torch.Tensor | None = None
        self._fill: torch.Tensor | None = None
        self._written: torch.Tensor | None = None

    def alloc_rings(self, num_slots: int, device: torch.device, dtype: torch.dtype) -> None:
        shape = (num_slots, self._window, self.layers.op_list[0].self_attn.kv_attn_dim)
        self._k_ring = torch.zeros(shape, device=device, dtype=dtype)
        self._v_ring = torch.zeros(shape, device=device, dtype=dtype)
        self._fill = torch.zeros(num_slots, device=device, dtype=torch.int32)
        self._written = torch.zeros(num_slots, device=device, dtype=torch.int32)
        self._pred = torch.zeros(num_slots, device=device, dtype=torch.int32)
        self._conf = torch.ones(num_slots, device=device, dtype=torch.float32)

    def reset_slots(self, slots: torch.Tensor) -> None:
        self._fill.index_fill_(0, slots, 0)
        self._written.index_fill_(0, slots, 0)

    def remember(self, slots: torch.Tensor, pred: torch.Tensor) -> None:
        # index_copy_ keeps this capturable
        self._pred.index_copy_(0, slots, pred.to(self._pred.dtype))

    def remember_confidence(self, slots: torch.Tensor, conf: torch.Tensor) -> None:
        self._conf.index_copy_(0, slots, conf)

    def drafts(self, slots: torch.Tensor) -> torch.Tensor:
        """The last draft remembered for each of ``slots``."""
        return self._pred.index_select(0, slots)

    def confidence(self, slots: torch.Tensor) -> torch.Tensor:
        """The last top softmax probability remembered for each of ``slots``."""
        return self._conf.index_select(0, slots)

    @property
    def num_slots(self) -> int:
        return 0 if self._fill is None else self._fill.numel()

    @property
    def ring_bytes(self) -> int:
        return 0 if self._k_ring is None else 2 * self._k_ring.numel() * self._k_ring.element_size()

    @nvtx_annotate("MTPDraft")
    def forward(
        self,
        hidden: torch.Tensor,
        next_ids: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        embed_tokens,
        lm_head,
    ) -> torch.Tensor:
        """Logits over ``x_{i+2}`` for every row, advancing each row's ring by one token."""
        layer = self.layers.op_list[0]
        attn = layer.self_attn
        e = self.pre_fc_norm_embedding.forward(embed_tokens.forward(next_ids))
        h = self.pre_fc_norm_hidden.forward(hidden)
        x = self.fc.forward(torch.cat([e, h], dim=-1))
        if self._ablate is not None:
            x = self._ablate.cut_(x)

        residual = x
        q, k, v, gate = attn.project(layer.input_layernorm.forward(x), positions)
        w = (self._written.index_select(0, slots) % self._window).to(torch.long)
        self._k_ring[slots, w] = k.to(self._k_ring.dtype)
        self._v_ring[slots, w] = v.to(self._v_ring.dtype)
        self._written[slots] += 1
        self._fill[slots] = torch.clamp(self._fill.index_select(0, slots) + 1, max=self._window)

        n = q.shape[0]
        keys = self._k_ring.index_select(0, slots).view(n, self._window, attn.num_kv, attn.head_dim)
        values = self._v_ring.index_select(0, slots).view(n, self._window, attn.num_kv, attn.head_dim)
        live = (
            torch.arange(self._window, device=q.device)[None, :]
            < self._fill.index_select(0, slots)[:, None]
        )
        rep = attn.num_q // attn.num_kv
        o = torch.nn.functional.scaled_dot_product_attention(
            q.view(n, attn.num_q, 1, attn.head_dim),
            keys.repeat_interleave(rep, dim=2).transpose(1, 2),
            values.repeat_interleave(rep, dim=2).transpose(1, 2),
            attn_mask=live[:, None, None, :],
        ).view(n, attn.qo_attn_dim)
        x = attn.o_proj.forward(o.to(x.dtype) * torch.sigmoid(gate)) + residual
        if self._ablate is not None:
            x = self._ablate.cut_(x)

        residual = x
        x = layer.mlp.forward(layer.post_attention_layernorm.forward(x)) + residual
        if self._ablate is not None:
            x = self._ablate.cut_(x)
        return lm_head.logits(self.norm.forward(x))


__all__ = ["Qwen3_5MTPDraft", "Qwen3_5MTPHead", "Qwen3_5MTPLayer"]
