from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    # [3, bs] t/h/w rope positions; allocated only for mrope models (else None).
    mrope_positions: torch.Tensor | None
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    # slots the verify step's second token writes its recurrent state to; None on decode buffers
    verify_out: torch.Tensor | None = None
    rows_per_req: int = 1

    @classmethod
    def init(
        cls, bs: int, vocab_size: int, device: torch.device, mrope: bool = False,
        rows_per_req: int = 1,
    ) -> GraphCaptureBuffer:
        rows = bs * rows_per_req
        return GraphCaptureBuffer(
            input_ids=torch.zeros(rows, dtype=torch.int32, device=device),
            out_loc=torch.zeros(rows, dtype=torch.int32, device=device),
            positions=torch.zeros(rows, dtype=torch.int32, device=device),
            mrope_positions=(
                torch.zeros(3, rows, dtype=torch.int32, device=device) if mrope else None
            ),
            logits=torch.empty(rows, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
            verify_out=(
                torch.zeros(bs, dtype=torch.int32, device=device) if rows_per_req > 1 else None
            ),
            rows_per_req=rows_per_req,
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        bs = batch.padded_size
        _slice = slice(bs)
        _rows = slice(bs * self.rows_per_req)
        batch.input_ids = self.input_ids[_rows]
        batch.out_loc = self.out_loc[_rows]
        batch.positions = self.positions[_rows]
        if self.mrope_positions is not None:
            batch.mrope_positions = self.mrope_positions[:, _rows]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1],
            cache_indices=self.table_idx[_slice],
            verify_out_indices=None if self.verify_out is None else self.verify_out[_slice],
        )

    def copy_from(self, batch: Batch) -> None:
        bs = batch.padded_size
        _slice = slice(bs)
        _rows = slice(bs * self.rows_per_req)
        self.input_ids[_rows] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_rows] = batch.out_loc
        self.positions[_rows] = batch.positions
        if self.mrope_positions is not None:
            self.mrope_positions[:, _rows] = batch.mrope_positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx
        if self.verify_out is not None and batch.fla_metadata is not None:
            self.verify_out[_slice] = batch.fla_metadata.verify_out_indices


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        mrope: bool = False,
        capture_verify: bool = False,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.mrope = mrope
        self.stream = stream
        self.device = device
        self.capture_verify = capture_verify
        self.verify_graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        self.verify_buffer: GraphCaptureBuffer | None = None
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(
            self.max_graph_bs, vocab_size, self.device, mrope=self.mrope
        )
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph
        self.pool = pool  # graphs captured later (the MTP draft) share it rather than open a second
        if self.capture_verify:
            self._capture_verify_graphs(vocab_size, model, pbar)

        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_verify_graphs(self, vocab_size: int, model: BaseLLMModel, pbar) -> None:
        """Second graph per size, for the two-rows-per-request shape of a verify step.

        Padding is not available here (a padded request would contribute one row, not two), so
        only the sizes captured outright can replay; other sizes fall back to eager.
        """
        self.verify_buffer = GraphCaptureBuffer.init(
            self.max_graph_bs, vocab_size, self.device, mrope=self.mrope, rows_per_req=2
        )
        dummy_slot = (self.dummy_req.linear_slot_idx
                      if self.dummy_req.linear_slot_idx is not None
                      else self.dummy_req.table_idx)
        # the wrapper plans off extend_len, so the dummy has to present the verify shape
        saved_device_len = self.dummy_req.device_len
        self.dummy_req.device_len = self.dummy_req.cached_len + 2
        for bs in sorted(self.graph_bs_list, reverse=True):
            pbar.desc = f"Capturing verify graphs: bs = {bs:<3}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.verify = True
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.verify_buffer.set_batch(batch)
            self.verify_buffer.table_idx[:bs].fill_(dummy_slot)
            self.verify_buffer.verify_out[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.verify_buffer.logits[: 2 * bs] = model.forward()
                with torch.cuda.graph(graph, pool=self.pool, stream=self.stream):
                    self.verify_buffer.logits[: 2 * bs] = model.forward()
                self._reset_moe_offload_cache()
            self.verify_graph_map[bs] = graph
        self.dummy_req.device_len = saved_device_len
        self._reset_moe_offload_cache()
        logger.info_rank0(
            f"Captured verify CUDA graphs for sizes: {sorted(self.verify_graph_map)}"
        )

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if batch.verify:
            # two rows per request: a padded request would add one, so only captured sizes run
            return batch.size in self.verify_graph_map
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        buffer = self.verify_buffer if batch.verify else self.buffer
        graphs = self.verify_graph_map if batch.verify else self.graph_map
        buffer.copy_from(batch)
        g = graphs[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return buffer.logits[: batch.size * buffer.rows_per_req]

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        gc.collect()
