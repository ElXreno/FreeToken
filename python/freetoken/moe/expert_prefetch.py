"""Next-layer expert prefetch for offload decode.

At MoE layer l a predictor scores layer l+1's experts from layer l's router input (``x_l @ P_l``: layer l+1's
router by default, or trained weights). Once layer l's experts are done (GPU GEMM and CPU share combined), the
best ``topn`` that are not resident are narrowed to the first ``budget``, each gets an LRU victim slot in layer
l+1's pool, and the slot tables are updated at once, so layer l+1's ensure sees them as hits; the CPU pool gets a
hint for the remaining cold ones. The rows are copied on a side stream with the same fused host-to-device kernel
as ``copy_missing``, during layer l+1's mixer, when host memory is otherwise idle; layer l+1 waits for that copy
after launching its own demand copy and before its experts are read. Everything is device-side with fixed shapes,
so it captures into the decode CUDA graph like the rest of the offload path.

``FREETOKEN_EXPERT_PREFETCH=<budget>`` enables it (0 or unset: off), ``FREETOKEN_EXPERT_PREFETCH_TOPN`` sets how
many predicted experts are considered (default 16), ``FREETOKEN_EXPERT_PREFETCH_WEIGHTS=<file>`` loads trained
predictor weights (``torch.save`` of ``{"W": [num_layers - 1, hidden, experts]}``) instead of the routers,
``FREETOKEN_EXPERT_PREFETCH_BLOCKS`` sets the copy kernel's blocks per bank (default: the demand copy's).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from freetoken.utils import init_logger

logger = init_logger(__name__)

_MAX_POOL = 8192


def prefetch_budget() -> int:
    return max(0, int(os.environ.get("FREETOKEN_EXPERT_PREFETCH", "0") or 0))


def next_layer_routers(model) -> dict[int, torch.Tensor]:
    """Router weight ``[experts, hidden]`` of every MoE layer, by MoE layer id (decoder layers with ``mlp.gate`` and ``mlp.experts``)."""
    root = getattr(model, "model", model)
    layers = getattr(root, "layers", None)
    out = {}
    for dec in getattr(layers, "op_list", layers) or []:
        mlp = getattr(dec, "mlp", None)
        experts, gate = getattr(mlp, "experts", None), getattr(mlp, "gate", None)
        weight = getattr(gate, "weight", None)
        layer_id = getattr(experts, "layer_id", None)
        if layer_id is not None and isinstance(weight, torch.Tensor):
            out[layer_id] = weight
    return out


class ExpertPrefetch:
    def __init__(self, cache, predictors: dict[int, torch.Tensor], *, budget: int, topn: int = 16):
        self.cache = cache
        self.budget = budget
        self.topn = max(topn, budget)
        self.predictors = predictors
        self.blocks = int(os.environ.get("FREETOKEN_EXPERT_PREFETCH_BLOCKS", "0") or 0)
        L, device = cache.num_layers, cache.device
        self.stream = torch.cuda.Stream(device=device)
        self.done = [torch.cuda.Event() for _ in range(L)]
        self.pending = [False] * L
        self.scores: list[torch.Tensor | None] = [None] * L
        self.evict = torch.zeros(L, budget, dtype=torch.int32, device=device)
        self.src = torch.zeros(L, budget, dtype=torch.int32, device=device)
        self.num = torch.zeros(L, 1, dtype=torch.int64, device=device)
        self.issued = torch.zeros(L, dtype=torch.int64, device=device)

    def targets(self, layer_id: int) -> int | None:
        """The layer prefetched from ``layer_id``, or None when it cannot take one."""
        t = layer_id + 1
        cache = self.cache
        if layer_id not in self.predictors or t >= cache.num_layers:
            return None
        if cache.is_cpu_layer(t) or cache.is_unpinned_layer(t) or t in cache.locked_hybrid:
            return None
        if cache.pool_slots(t) > _MAX_POOL or not cache._copy_fused_ok:
            return None
        return t

    def issue(self, layer_id: int, hidden: torch.Tensor) -> None:
        self.score(layer_id, hidden)
        self.launch(layer_id)

    def score(self, layer_id: int, hidden: torch.Tensor) -> None:
        """Score layer ``layer_id + 1``'s experts from this layer's router input (before the expert GEMM may overwrite it)."""
        t = self.targets(layer_id)
        self.scores[layer_id] = None
        if t is None:
            return
        logits = torch.nn.functional.linear(hidden, self.predictors[layer_id]).float()
        self.scores[layer_id] = torch.softmax(logits, dim=-1).sum(0)
        if self.budget == 0:
            self._hint(t, self.scores[layer_id])

    def launch(self, layer_id: int) -> None:
        """Once this layer's experts are done: fetch the next layer's best cold experts and hint the CPU pool at the
        rest, so both move bytes while the next layer's mixer runs and the host memory bus is otherwise idle."""
        score = self.scores[layer_id]
        if score is None:
            return
        if self.budget > 0:
            self._fetch(layer_id + 1, score)
            self._hint(layer_id + 1, score)

    def _hint(self, t: int, score: torch.Tensor) -> None:
        if getattr(self.cache.cpu_executor, "prefetch_hint", None) is not None:
            from freetoken.moe.cpu_prefetch_hint import write_cpu_prefetch_hint

            write_cpu_prefetch_hint(self.cache, t, score)

    def _fetch(self, t: int, score: torch.Tensor) -> None:
        cache = self.cache
        id_of_slot, usage = cache.pool_state(t)
        num_slots = id_of_slot.numel()
        _prefetch_select_kernel[(1,)](
            score, cache.slot_for_id, id_of_slot, usage, cache.step,
            self.evict[t], self.src[t], self.num[t], self.issued[t:],
            t * cache.num_experts, num_slots, self.topn, self.budget, cache.num_experts,
            BLOCK_E=triton.next_power_of_2(cache.num_experts),
            BLOCK_B=triton.next_power_of_2(max(self.budget, 2)),
            CHUNK=triton.next_power_of_2(num_slots),
            num_warps=8 if num_slots >= 4096 else 4,
        )
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            fast_index_copy_multi_jit(
                cache._copy_dst_ptrs_pool[cache.pool_of_layer[t]],
                cache._copy_src_ptrs[t],
                cache._copy_feat_bytes[cache.layer_layout[t]],
                self.evict[t],
                self.src[t],
                self.num[t],
                blocks_per_bank=self.blocks or cache._copy_blocks_per_bank,
            )
            self.done[t].record(self.stream)
        self.pending[t] = True

    def wait(self, layer_id: int) -> None:
        if self.pending[layer_id]:
            torch.cuda.current_stream().wait_event(self.done[layer_id])
            self.pending[layer_id] = False

    def issued_per_layer(self) -> list[int]:
        return self.issued.tolist()


@triton.jit(do_not_specialize=["base", "cache_size", "topn", "budget"])
def _prefetch_select_kernel(
    score_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    evict_ptr,
    src_ptr,
    num_ptr,
    issued_ptr,
    base,
    cache_size,
    topn,
    budget,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_B: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Take the best-scored of the top ``topn`` experts that are not resident, at most ``budget``, give each the
    least-used slot of the pool and stamp it two steps ahead: the next layer's ensure (one step on) then sees a
    hit and cannot pick the slot as a victim while the copy into it may still be running."""
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    s = tl.load(score_ptr + off_e, mask=e_mask, other=float("-inf"))
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=0)
    cold = e_mask & (slot == -1)
    step = tl.load(step_ptr)
    off_b = tl.arange(0, BLOCK_B)
    cand = tl.full((BLOCK_B,), 0, dtype=tl.int32)
    found = tl.zeros((), dtype=tl.int32)
    for i in tl.range(topn):
        e = tl.argmax(s, axis=0).to(tl.int32)
        s = tl.where(off_e == e, float("-inf"), s)
        is_cold = tl.sum(tl.where(off_e == e, cold.to(tl.int32), 0)) > 0
        keep = is_cold & (found < budget)
        cand = tl.where((off_b == found) & keep, e, cand)
        found = found + keep.to(tl.int32)
    off_k = tl.arange(0, CHUNK)
    u = tl.load(usage_ptr + off_k, mask=off_k < cache_size, other=9223372036854775807).to(tl.int64)
    n = tl.zeros((), dtype=tl.int32)
    for j in tl.range(budget):
        take = j < found
        e = tl.sum(tl.where(off_b == j, cand, 0))
        victim = tl.argmin(u, axis=0).to(tl.int32)
        u = tl.where((off_k == victim) & take, 9223372036854775807, u)
        old_id = tl.load(id_of_slot_ptr + victim)
        tl.store(slot_for_id_ptr + old_id, -1, mask=take & (old_id >= 0))
        tl.store(id_of_slot_ptr + victim, base + e, mask=take)
        tl.store(slot_for_id_ptr + base + e, victim, mask=take)
        tl.store(usage_ptr + victim, step + 2, mask=take)
        tl.store(evict_ptr + n, victim, mask=take)
        tl.store(src_ptr + n, e, mask=take)
        n = n + take.to(tl.int32)
        tl.debug_barrier()
    tl.store(num_ptr, n.to(tl.int64))
    tl.store(issued_ptr, tl.load(issued_ptr) + n.to(tl.int64))


__all__ = ["ExpertPrefetch", "next_layer_routers", "prefetch_budget"]
