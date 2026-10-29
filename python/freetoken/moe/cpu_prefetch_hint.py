"""GPU side of the CPU pool's cache prefetch: the predicted cold experts of a layer, written into the executor's hint.

``write_cpu_prefetch_hint(cache, layer, score)`` takes a predictor's per-expert score for ``layer`` (the next layer
of the one running), keeps the best-scored of the top ``topn`` that are not resident in the GPU slot cache, at most
``max_n``, and copies ``stamp, count, ids`` into ``cache.cpu_executor.prefetch_hint[layer]``. The stamp is the cache
step, so every decode step gives the pool a fresh hint; the pool prefetches it once the current layer's CPU work is
done. Device-side with a fixed-size device-to-host copy, so it captures into the decode CUDA graph. Run it after any
GPU prefetch has claimed its experts: those are resident by then and stay out of the hint.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def write_cpu_prefetch_hint(cache, layer: int, score: torch.Tensor, *, max_n: int = 4, topn: int = 16) -> None:
    executor = cache.cpu_executor
    hint = getattr(executor, "prefetch_hint", None)
    if hint is None or layer >= cache.num_layers:
        return
    staging = getattr(executor, "_prefetch_hint_dev", None)
    if staging is None:
        staging = torch.zeros(hint.shape, dtype=torch.int32, device=score.device)
        executor._prefetch_hint_dev = staging
    width = hint.shape[1]
    _hint_kernel[(1,)](
        score, cache.slot_for_id, cache.step, staging[layer],
        layer * cache.num_experts, min(max_n, width - 2), topn, cache.num_experts,
        BLOCK_E=triton.next_power_of_2(cache.num_experts),
    )
    hint[layer].copy_(staging[layer], non_blocking=True)


@triton.jit(do_not_specialize=["base", "max_n", "topn"])
def _hint_kernel(score_ptr, slot_for_id_ptr, step_ptr, out_ptr, base, max_n, topn, num_experts: tl.constexpr,
                 BLOCK_E: tl.constexpr):
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    s = tl.load(score_ptr + off_e, mask=e_mask, other=float("-inf"))
    cold = e_mask & (tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=0) == -1)
    n = tl.zeros((), dtype=tl.int32)
    for i in tl.range(topn):
        e = tl.argmax(s, axis=0).to(tl.int32)
        s = tl.where(off_e == e, float("-inf"), s)
        take = (tl.sum(tl.where(off_e == e, cold.to(tl.int32), 0)) > 0) & (n < max_n)
        tl.store(out_ptr + 2 + n, e, mask=take)
        n = n + take.to(tl.int32)
    tl.store(out_ptr + 1, n)
    tl.store(out_ptr, (tl.load(step_ptr) & 0x7FFFFFFF).to(tl.int32))


__all__ = ["write_cpu_prefetch_hint"]
