"""The Qwen3.5 MTP head as the draft side of speculative decoding (``--speculative-algorithm mtp``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from freetoken.core import Batch, get_global_ctx
from freetoken.engine.sample import make_device_tensor
from freetoken.spec.proposer import DraftContext, Proposer
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.core import Req

logger = init_logger(__name__)


class _FinalHidden:
    """Keeps the target's pre-LM-head rows the next draft reads: every row of a verify step, the last row of each prefill request."""

    wanted_layers = frozenset()

    def __init__(self, proposer: Qwen3_5MTPProposer, batch: Batch) -> None:
        self._proposer = proposer
        self._batch = batch

    def final_hidden(self, x: torch.Tensor) -> None:
        if self._batch.is_prefill:
            last = self._batch.attn_metadata.get_last_indices(self._batch.size)
            x = x[last]
        self._proposer._feat[: x.shape[0]].copy_(x)

    def layer_output(self, layer_id: int, x: torch.Tensor, num_streams: int = 1) -> None:
        pass


class Qwen3_5MTPProposer(Proposer):
    """One draft per step from the checkpoint's MTP head, on the target's last committed row.

    EAGLE alignment: the head takes the token ``x_{i+1}`` the target just emitted with the
    target's hidden ``h_i`` at position ``p_i`` and predicts ``x_{i+2}``. Its layer attends over a
    per-request ring of the tokens it was fed, keyed by ``table_idx``, so the paged KV cache and
    the prefix cache stay the target's alone. A draft step reads and writes only the head's own
    persistent buffers and replays as one captured graph per batch size.
    """

    default_k = 1
    max_k = 1

    @classmethod
    def adjust_config(cls, config, override) -> None:
        super().adjust_config(config, override)
        if config.model_config.mtp_num_layers == 0:
            raise ValueError("--speculative-algorithm mtp needs a checkpoint with an MTP head (mtp_num_hidden_layers)")
        if getattr(config, "mixed_batch_decode", False):
            raise ValueError("--speculative-algorithm mtp does not combine with --mixed-batch-decode yet")

    @classmethod
    def budget(cls, config) -> tuple[int, int]:
        mc = config.model_config
        slots = config.max_running_req + 1
        ring = 2 * slots * mc.mtp_window * mc.num_kv_heads * mc.head_dim * config.dtype.itemsize
        feat = slots * (max(config.speculative_num_steps, 1) + 2) * mc.hidden_size * config.dtype.itemsize
        return ring + feat, 0

    def __init__(self, config, target_model) -> None:
        super().__init__(config, target_model)
        self.head = target_model.mtp
        self.embed_tokens = target_model.model.embed_tokens
        self.lm_head = target_model.lm_head
        self.max_rows = config.max_running_req
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._rings = False

    def attach(self, ctx: DraftContext) -> None:
        super().attach(ctx)
        width = self.max_rows + 1
        dev, dtype, hidden = ctx.device, self.config.dtype, self.config.model_config.hidden_size
        if not self._rings:
            self.head.alloc_rings(width, dev, dtype)
            self._rings = True
        # verify rows: k+1 per request
        self._feat = torch.zeros(width * (self.k + 1), hidden, device=dev, dtype=dtype)
        self._hidden = torch.zeros(width, hidden, device=dev, dtype=dtype)
        self._tokens = torch.zeros(width, device=dev, dtype=torch.int32)
        self._positions = torch.zeros(width, device=dev, dtype=torch.int32)
        self._slots = torch.zeros(width, device=dev, dtype=torch.int64)
        logger.info_rank0(
            f"MTP draft head: window {self.config.model_config.mtp_window}, {width} ring slots, "
            f"{self.head.ring_bytes / 2**20:.0f} MiB of rings"
        )

    def detach(self, *, keep_kv: bool) -> None:
        self.graphs = {}
        super().detach(keep_kv=keep_kv)

    def feature_sink(self, batch: Batch) -> _FinalHidden:
        return _FinalHidden(self, batch)

    def _draft(self, n: int) -> None:
        logits = self.head.forward(
            self._hidden[:n], self._tokens[:n], self._positions[:n], self._slots[:n], self.embed_tokens, self.lm_head
        )
        self.head.remember(self._slots[:n], logits.argmax(dim=-1))

    def _run(self, n: int) -> None:
        graph = self.graphs.get(n)
        if graph is not None:
            graph.replay()
            return
        with get_global_ctx().forward_batch(self._draft_batch(n)):
            self._draft(n)

    def _draft_batch(self, n: int) -> Batch:
        batch = Batch(reqs=[self.ctx.dummy_req] * n, phase="decode")
        batch.padded_reqs = batch.reqs
        batch.positions = self._positions[:n]
        return batch

    def on_prefill(self, batch: Batch, next_tokens_gpu: torch.Tensor) -> None:
        from freetoken.scheduler.prefill import ChunkedReq

        rows = [i for i, r in enumerate(batch.reqs) if not isinstance(r, ChunkedReq) and r.table_idx != -1]
        if not rows:
            return
        n = len(rows)
        sel = make_device_tensor(rows, torch.int64, self.ctx.device)
        slots = make_device_tensor([batch.reqs[i].table_idx for i in rows], torch.int64, self.ctx.device)
        # a request's ring starts with its first draft, whoever held the table row before
        self.head.reset_slots(slots)
        self._slots[:n].copy_(slots)
        self._hidden[:n].copy_(self._feat.index_select(0, sel))
        self._tokens[:n].copy_(next_tokens_gpu.index_select(0, sel))
        last = batch.attn_metadata.get_last_indices(batch.size)
        self._positions[:n].copy_(batch.positions[last].index_select(0, sel))
        self._run(n)

    def post_verify(self, batch: Batch, out: torch.Tensor, emitted: torch.Tensor) -> None:
        n = batch.size
        width = self.k + 1
        base = torch.arange(n, device=out.device, dtype=torch.int64) * width
        sel = base + emitted[:n]
        slots = make_device_tensor([r.table_idx for r in batch.reqs[:n]], torch.int64, self.ctx.device)
        self._slots[:n].copy_(slots)
        self._hidden[:n].copy_(self._feat.index_select(0, sel))
        self._tokens[:n].copy_(out[:n].gather(1, emitted[:n].unsqueeze(1)).squeeze(1))
        self._positions[:n].copy_(batch.positions.index_select(0, sel))
        self._run(n)

    def propose(self, reqs: list[Req], rows: torch.Tensor) -> torch.Tensor:
        return self.head.drafts(rows).unsqueeze(1).to(torch.int64)

    @torch.inference_mode()
    def capture_graphs(self, bs_list: list[int], pool) -> None:
        stream = self.ctx.stream
        self.graphs = {}
        for n in range(1, self.max_rows + 1):
            with get_global_ctx().forward_batch(self._draft_batch(n)):
                side = torch.cuda.Stream(device=self.ctx.device)
                side.wait_stream(stream)
                with torch.cuda.stream(side):
                    for _ in range(3):
                        self._draft(n)
                stream.wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._draft(n)
            self.graphs[n] = graph
        self.head.reset_slots(torch.arange(self.head.num_slots, device=self.ctx.device))
        cache = self._head_cache()
        if cache is not None:
            # the warmup and capture routed dummy experts into the cache the target serves from
            cache.reset()

    def _head_cache(self):
        experts = self.head.layers.op_list[0].mlp.experts
        return None if experts is None else getattr(experts, "offload_cache", None)

    def release(self, req: Req) -> None:
        pass


__all__ = ["Qwen3_5MTPProposer"]
