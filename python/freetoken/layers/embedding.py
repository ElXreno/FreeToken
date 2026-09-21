from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from freetoken.core import get_global_ctx
from freetoken.distributed import DistributedCommunicator, get_tp_info
from freetoken.utils import div_ceil, nvtx_annotate

from .base import BaseOP
from .quantization import LayerKind, QuantConfig, quant_method_for


class HostEmbeddingTable:
    """Vocabulary table left in pinned host memory, rows gathered over UVA by the PLE kernel.

    The table is read once per token (4 KiB at hidden 2048) against a decode step that already
    streams ~1 GiB of weights, so PCIe cost is noise -- but it hands the whole table's VRAM to
    the MoE expert cache. One staging buffer per captured decode size (a captured graph writes
    the same block on every replay), one growable buffer for eager calls."""

    def __init__(self, weight: torch.Tensor, device: torch.device) -> None:
        from freetoken.kernel.pinned import device_ptr
        from freetoken.moe.host_banks import HostBank

        assert weight.dtype in (torch.bfloat16, torch.float16, torch.float32), weight.dtype
        self.num_rows, self.embed_dim = weight.shape
        self.dtype = weight.dtype
        self.device = device
        self._bank = HostBank(tuple(weight.shape), weight.dtype)
        self._bank.tensor.copy_(weight.to("cpu", non_blocking=False))
        self._bank.pin()
        self._table_ptr = device_ptr(self._bank.tensor)
        self.nbytes = self._bank.nbytes
        self._staging: torch.Tensor | None = None
        self._graph_staging: Dict[int, torch.Tensor] = {}

    def _stage(self, rows: int) -> torch.Tensor:
        if torch.cuda.is_current_stream_capturing():
            buf = self._graph_staging.get(rows)
            if buf is None:
                buf = torch.empty((rows, self.embed_dim), dtype=self.dtype, device=self.device)
                self._graph_staging[rows] = buf
            return buf
        buf = self._staging
        if buf is None or buf.shape[0] < rows:
            buf = torch.empty((rows, self.embed_dim), dtype=self.dtype, device=self.device)
            self._staging = buf
        return buf[:rows]

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel.triton.ple import ple_gather_rows

        flat = indices.reshape(-1)
        out = ple_gather_rows(
            self._table_ptr, self.num_rows, self.embed_dim, flat,
            self._stage(flat.numel()), 1.0, False,
        )
        return out.view(*indices.shape, self.embed_dim)


class VocabParallelEmbedding(BaseOP):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        embed_scale: float | None = None,
    ):
        super().__init__()
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size
        self.num_embeddings = num_embeddings
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        # Gemma scales embeddings by sqrt(hidden_size). The scale is materialized in
        # the weight dtype (bf16) to match HF, which downcasts the scalar. The GPU
        # scalar is built lazily (model __init__ runs on the meta device) and cached
        # so it is not reallocated inside a captured CUDA graph.
        self._embed_scale = embed_scale
        self._embed_scale_t: torch.Tensor | None = None
        self._comm = DistributedCommunicator()
        self._host_table: HostEmbeddingTable | None = None

    def move_to_host(self) -> int:
        """Hand the table's VRAM to the runtime pools; returns the pinned host bytes taken."""
        assert self.tp_size == 1, "host embedding table is single-rank only"
        table = HostEmbeddingTable(self.weight, self.weight.device)
        self._host_table = table
        self.weight = torch.empty(0, dtype=table.dtype, device=table.device)
        torch.cuda.empty_cache()
        return table.nbytes

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from freetoken.kernel import indexing

        if self._host_table is not None:
            y = self._host_table.lookup(x)
        else:
            y = indexing(
                weights=self.weight,
                indices=x,
                vocab_range=self.vocab_range if self.tp_size > 1 else None,
            )

        if self.tp_size > 1:
            y = self._comm.all_reduce(y)
        if self._embed_scale is not None:
            if self._embed_scale_t is None:
                self._embed_scale_t = torch.tensor(
                    self._embed_scale, dtype=y.dtype, device=y.device
                )
            y = y * self._embed_scale_t
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """The head is a linear layer over the vocab shard: its weights come from ``quant_method``
    unless they are tied to the input embedding."""

    quant_layer_kind = LayerKind.LINEAR

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
        *,
        quant_config: QuantConfig | None = None,
        prefix: str = "",
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.has_bias = bias
        self.prefix = prefix
        self.tied_embedding = tied_embedding
        assert (tied_embedding is not None) == tie_word_embeddings
        self.in_features = embedding_dim
        self.out_features = self.num_embeddings_tp
        self.output_sizes = (self.num_embeddings_tp,)
        self.quant_method = None
        if tied_embedding is None:
            self.quant_method = quant_method_for(quant_config, self, prefix)
            self.quant_method.create_weights(self)
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None

    def finalize(self) -> None:
        if self.quant_method is not None:
            self.quant_method.finalize(self)

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx = get_global_ctx()
        batch = ctx.batch
        bs = batch.size
        if batch.is_prefill:
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        if self.tied_embedding is not None:
            logits = F.linear(x, self.tied_embedding.weight, self.bias)
        else:
            logits = self.quant_method.apply(self, x)
        if self.tp_size == 1:
            return logits
        input_shape = logits.shape
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        return output_tensor[:, : self.num_embeddings]