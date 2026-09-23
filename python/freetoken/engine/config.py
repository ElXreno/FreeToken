from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import TYPE_CHECKING, List

import torch
from freetoken.distributed import DistributedInfo
from freetoken.layers.quantization import set_quant_config
from freetoken.mm.config import ENCODER_SECTIONS, MultimodalConfig
from freetoken.models.register import EncoderSpec, ModelSpec, _load_attr, checkpoint_quant_config, get_model_spec
from freetoken.utils import cached_load_hf_config, init_logger

if TYPE_CHECKING:
    from freetoken.models import ModelConfig

logger = init_logger(__name__)


def _resolve_mtp_top_k(requested: int, model_config: ModelConfig) -> int:
    """Router width of the draft head's MoE: 0 (shared expert only) or the model's own.

    The CPU MoE executor is built once, from the first MoE layer, and its pinned routing
    buffers are ``[batch, top_k]``; a head routing narrower than the model would not fit them.
    """
    width = model_config.num_experts_per_tok
    if requested < 0 or requested == width:
        return width
    if requested == 0:
        return 0
    raise ValueError(
        f"--mtp-routed-experts must be 0 or the model's router width {width}, got {requested}: "
        "the CPU MoE executor allocates one routing buffer for every layer"
    )


@dataclass(frozen=True)
class EngineConfig:
    model_path: str
    tp_info: DistributedInfo
    dtype: torch.dtype
    max_running_req: int = 4
    attention_backend: str = "auto"
    moe_strategy: str = "auto"
    # old name of moe_strategy; __post_init__ folds it in
    moe_backend: str | None = field(default=None, repr=False)
    # --quant-backend: layer[.kind]=kernel entries, comma separated
    quant_backend: str | None = None
    # PLE table backend: "disk" (default) reads rows from the checkpoint files per fill, "pinned" preloads the table into page-locked host RAM.
    ple_backend: str = "disk"
    # Expert-bank host load (--expert-load): auto|serial|parallel. "auto" reads scattered
    # experts in parallel but falls back to serial when free RAM can't cover the banks + the
    # parallel reader's extra (non-reclaimable) whole-shard buffer; "serial" forces the
    # low-memory reclaimable read; "parallel" forces the fast read.
    expert_load: str = "auto"
    moe_cache_size: int = 0
    moe_cache_rate: float | None = None
    moe_cache_auto: bool = False
    kv_reserve_tokens: int = 8192  # KV floor for --moe-cache-auto; small by design (MoE-priority)
    moe_cache_policy: str = "lru"
    moe_prefill_overlap: bool = True
    # Prefill hit/miss split: serve cache-resident experts D2D during prefill
    # prefetch instead of re-streaming the full layer over PCIe. Needs CUDA >= 12.8
    # (cudaMemcpyBatchAsync); no-op unless moe_cache_size > 2 * num_experts.
    moe_prefill_hit_d2d: bool = False
    moe_collect_stats: bool = False  # capture decode miss-rate counters into the cuda graph
    # Keep the input embedding table in pinned host memory (gathered over UVA) so its VRAM
    # goes to the runtime pools instead. Single rank only.
    embed_table_host: bool = False
    # Let running decodes ride each prefill chunk as one-token extends; composes with
    # decode_steps_per_prefill_chunk (they get the rider token AND the post-chunk burst).
    mixed_batch_decode: bool = False
    # Build the checkpoint's MTP head and draft one token per decode step; no-op without a head.
    mtp_draft: bool = False
    mtp_window: int = 512
    # routed experts the head takes per draft; 0 keeps only its shared expert, -1 follows the model
    mtp_top_k: int = -1
    # carry the head's draft as a second row and commit it when it held (speculative decode)
    mtp_verify: bool = False
    mtp_skip_projection: bool = False
    # remove row ``ablate_layer`` of this [layers, hidden] safetensors table from the residual
    # stream at every layer, scaled by ``ablate_alpha``; Qwen3.5 models only
    ablate_direction: str | None = None
    ablate_layer: int = 0
    ablate_alpha: float = 1.0
    # CPU MoE backend (--moe-strategy cpu): number of CPU worker threads computing
    # the decode experts. 0 = auto (physical cores). Ignored by other backends.
    moe_cpu_threads: int = 0
    # Hybrid CPU/GPU decode (--moe-strategy offload only): which MoE layers decode on
    # the CPU executor instead of the GPU offload/PCIe path. Spec is an explicit id
    # list ("3,7,11"), a count ("8" -> 8 layers evenly strided across depth), or a
    # fraction ("0.5"). None/"" = all layers on GPU (plain offload). --moe-strategy cpu
    # already means all layers on CPU and ignores this.
    moe_cpu_layers: str | None = None
    # Hybrid MoE backend (--moe-strategy hybrid): max experts fetched over PCIe per
    # (layer, decode step); the rest of that step's misses are computed on the CPU.
    # -1 (default) = auto: fetch the benched pcie_bw/cpu_bw fraction of each step's
    # misses so the PCIe fetch and the CPU compute finish together (perfect overlap);
    # falls back to a fixed cap of 1 without a usable `ft bench bw` profile.
    moe_hybrid_max_fetch: int = -1
    moe_hybrid_fetch_fraction: float = -1.0
    moe_hybrid_small_prefill: int = 0
    cuda_graph_bs: List[int] | None = None
    cuda_graph_max_bs: int | None = None
    page_size: int = 1
    # Paged KV slab dtype; None stores K/V in ``dtype``. torch.float8_e4m3fn halves the slab
    # (MHA/GQA pool on the fi backend only; static scale 1.0, saturating cast on store).
    kv_cache_dtype: torch.dtype | None = None
    # JSON with per-layer fp8 KV dequant scales ({"layers": {"<id>": {"k": s, "v": s}}}) from a
    # calibration pass: store_kv divides by them, attention folds them back. fp8 slab only.
    kv_cache_scales: str | None = None
    memory_ratio: float = 0.9
    # Hybrid GDN models default to the HybridRadixCache (cross-request GDN-state prefix reuse);
    # `--cache-type naive` opts out. linear_state_cache_ratio sizes the GDN snapshot cache as
    # ceil(ratio * max_running_req) extra slots.
    linear_state_cache_ratio: float = 2.0
    # Decode steps the scheduler runs between two prefill chunks while both kinds of work are
    # pending; 0 keeps strict prefill priority (a long chunked prefill stalls every running decode).
    decode_steps_per_prefill_chunk: int = 0
    # Host tier for the hybrid radix cache (--prefix-cache-dir): arena file + tree metadata
    # that keep committed KV spans and GDN snapshots across VRAM eviction and restarts.
    prefix_cache_dir: str | None = None
    prefix_cache_host_bytes: int = 32 << 30
    prefix_cache_flush_idle_seconds: float = 600.0
    prefix_cache_chunk_snapshots: bool = False
    # Window/full ratio for the SWA radix cache (`--cache-type radix` on SWA models) and the DSV4
    # window tier: the DEFAULT window-pool size = max(working-set floor, ratio x full-pool tokens).
    # < 1.0 trades retained window-prefix capacity for memory savings; must be in (0, 1]. It is the
    # DSV4 window/full ratio directly. Used only when swa_num_pages_override is None (a runtime
    # rebuild can pin an absolute window instead).
    swa_full_tokens_ratio: float = 0.2
    # Absolute window-pool size in the pool's own pages (usable, dummy excluded); None -> use the
    # ratio default above. A runtime cache rebuild sets this (num_swa_pages) to pin the window
    # regardless of the full anchor; the ratio is the startup default and the fallback.
    swa_num_pages_override: int | None = None
    distributed_timeout: float = 60.0
    use_dummy_weight: bool = False
    use_pynccl: bool = True
    max_seq_len_override: int | None = None
    num_page_override: int | None = None  # if not None, will override the number of pages
    # KV capacity in tokens; resolved into num_page_override by _adjust_config once page_size
    # is final. Mutually exclusive with num_page_override.
    num_token_override: int | None = None
    # Runtime knobs of the multimodal path; the architecture side (vision_config, mrope) lives in ModelConfig.
    mm: MultimodalConfig = field(default_factory=MultimodalConfig)

    def __post_init__(self):
        if self.moe_backend is None:
            return
        if self.moe_strategy != "auto":
            raise ValueError("moe_backend is the old name of moe_strategy; pass only moe_strategy")
        logger.warning("EngineConfig.moe_backend is deprecated; use moe_strategy")
        object.__setattr__(self, "moe_strategy", self.moe_backend)
        object.__setattr__(self, "moe_backend", None)

    @cached_property
    def hf_config(self):
        return cached_load_hf_config(self.model_path)

    @cached_property
    def model_spec(self) -> ModelSpec:
        return get_model_spec(self.hf_config.architectures[0])

    @cached_property
    def active_encoders(self) -> tuple[EncoderSpec, ...]:
        """The encoder towers this process builds: the family registers them, the checkpoint config carries their section, --mm-disable did not name them."""
        return tuple(
            e
            for e in self.model_spec.encoders
            if getattr(self.hf_config, e.config_key, None) is not None
            and e.kind not in self.mm.disabled_encoders
        )

    @cached_property
    def served_modalities(self) -> frozenset[str]:
        """Modalities this process accepts."""
        return frozenset(m for e in self.active_encoders for m in e.modalities)

    @cached_property
    def model_config(self) -> ModelConfig:
        # the parser sees no section for a tower this process does not build (for the vision tower that also means 1-D rope)
        hf_config = copy.copy(self.hf_config)
        built = {e.config_key for e in self.active_encoders}
        for key in set(ENCODER_SECTIONS) | {e.config_key for e in self.model_spec.encoders}:
            if key not in built:
                setattr(hf_config, key, None)
        spec = self.model_spec
        quant = checkpoint_quant_config(self.model_path, hf_config, spec)
        set_quant_config(quant)
        model_config = _load_attr(spec.module, spec.parse_config)(hf_config)
        return replace(
            model_config,
            quant=quant,
            mtp_draft=(self.mtp_draft or self.mtp_verify) and model_config.mtp_num_layers > 0,
            mtp_window=self.mtp_window,
            mtp_top_k=_resolve_mtp_top_k(self.mtp_top_k, model_config),
            mtp_skip_projection=self.mtp_skip_projection,
            ablate_direction=self.ablate_direction,
            ablate_layer=self.ablate_layer,
            ablate_alpha=self.ablate_alpha,
        )

    @property
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:
            return self.max_seq_len_override
        return self.model_config.rotary_config.max_position

    @property
    def max_forward_len(self) -> int:
        return self.max_seq_len

    @property
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"
