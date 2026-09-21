from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    LinearColParallelMerged,
    LinearReplicated,
    OPList,
    silu_and_mul,
)
from freetoken.layers.rotary import get_rope
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class _SkipProjection:
    """Stands in for the vocabulary projection when pricing it (``--mtp-skip-projection``)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :8]


class _MTPMlp(BaseOP):
    """The head's MoE block with the routed experts left out.

    The checkpoint ships them unquantized (1536 MiB for 256 experts), which does not fit in
    VRAM beside the main expert cache, and serving them from the host costs more draft latency
    than their acceptance is worth. ``gate`` is loaded so the state dict stays complete and a
    routed path can be switched on without touching the loader.
    """

    def __init__(self, config: ModelConfig, *, prefix: str = ""):
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _MTPSharedExpert(
            config, config.shared_expert_intermediate_size, prefix=f"{prefix}.shared_expert"
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared_expert.forward(x) * torch.sigmoid(self.shared_expert_gate.forward(x))


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
        self.q_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(head_dim, eps=config.rms_norm_eps)
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
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        self.pre_fc_norm_embedding = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = LinearReplicated(2 * config.hidden_size, config.hidden_size, has_bias=False)
        self.layers = OPList([Qwen3_5MTPLayer(config, prefix=f"{prefix}.layers.0")])
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        self._has_pred = torch.zeros(num_slots, device=device, dtype=torch.bool)
        self._hits = torch.zeros((), device=device, dtype=torch.int64)
        self._scored = torch.zeros((), device=device, dtype=torch.int64)

    def reset_slots(self, slots: torch.Tensor) -> None:
        self._fill.index_fill_(0, slots, 0)
        self._written.index_fill_(0, slots, 0)
        self._has_pred.index_fill_(0, slots, False)

    def score(self, slots: torch.Tensor, actual: torch.Tensor) -> None:
        """Charge last step's drafts against the tokens the model actually sampled."""
        live = self._has_pred.index_select(0, slots)
        self._hits += ((self._pred.index_select(0, slots) == actual) & live).sum()
        self._scored += live.sum()

    def remember(self, slots: torch.Tensor, pred: torch.Tensor) -> None:
        # index_copy_/index_fill_ keep this capturable: `self._has_pred[slots] = True` would
        # stage the python bool through an unpinned CPU tensor, which capture refuses.
        self._pred.index_copy_(0, slots, pred.to(self._pred.dtype))
        self._has_pred.index_fill_(0, slots, True)

    def acceptance(self) -> tuple[int, int]:
        return int(self._hits.item()), int(self._scored.item())

    def clear_counters(self) -> None:
        self._hits.zero_()
        self._scored.zero_()

    def draft_and_score(
        self,
        hidden: torch.Tensor,
        next_ids: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        embed_tokens,
        lm_head,
    ) -> None:
        """One step of bookkeeping and drafting, reading and writing only the head's own state.

        Every tensor it touches is persistent, so the whole thing captures into a CUDA graph and
        replays as a single launch off the engine's critical path.
        """
        self.score(slots, next_ids)
        logits = self.forward(hidden, next_ids, positions, slots, embed_tokens, lm_head)
        self.remember(slots, logits.argmax(dim=-1))

    def draft_probe(
        self,
        hidden: torch.Tensor,
        next_ids: torch.Tensor,
        positions: torch.Tensor,
        slots: torch.Tensor,
        embed_tokens,
    ) -> None:
        """Everything but the vocabulary projection, to price that projection under contention."""
        self.score(slots, next_ids)
        x = self.forward(hidden, next_ids, positions, slots, embed_tokens, _SkipProjection())
        self.remember(slots, x.argmax(dim=-1))

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

        residual = x
        x = layer.mlp.forward(layer.post_attention_layernorm.forward(x)) + residual
        return lm_head.forward(self.norm.forward(x))


__all__ = ["Qwen3_5MTPHead", "Qwen3_5MTPLayer"]
