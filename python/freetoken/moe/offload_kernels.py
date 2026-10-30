from __future__ import annotations

import os
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from flashlib.kernels.slot_cache import lru_ensure

# Hybrid backend: which of a step's missing experts to fetch (when capped below the miss
# count). "recency" (default) fetches the experts most-recently active before this step
# (LRU on the expert -> prioritizes recurring misses, lowering the steady miss rate);
# "lowest_id" fetches the smallest expert ids (the original, routing-blind heuristic).
_HYBRID_FETCH_BY_RECENCY = (
    os.getenv("FREETOKEN_HYBRID_FETCH", "recency").strip().lower() != "lowest_id"
)
# How fetch_fraction * misses becomes a fetch count: "min-wait" (default) picks the integer
# that makes the PCIe fetch and the CPU compute finish closest together, "stochastic" rounds
# up with probability equal to the fractional part, "prefer-fetch" always rounds up.
_HYBRID_FETCH_ROUNDING = {"min-wait": 0, "stochastic": 1, "prefer-fetch": 2}
_hybrid_fetch_round = (os.getenv("FREETOKEN_HYBRID_FETCH_ROUND") or "min-wait").strip().lower()
if _hybrid_fetch_round not in _HYBRID_FETCH_ROUNDING:
    raise ValueError(f"FREETOKEN_HYBRID_FETCH_ROUND={_hybrid_fetch_round!r}: expected one of {sorted(_HYBRID_FETCH_ROUNDING)}")
_HYBRID_FETCH_ROUND = _HYBRID_FETCH_ROUNDING[_hybrid_fetch_round]
_HYBRID_FETCH_SEED = 0
# >0: evict by hit frequency decayed with this time constant in ensure steps (LFU-decay), else LRU
_HYBRID_LFU_TAU = float(os.getenv("FREETOKEN_HYBRID_LFU_TAU", "0") or 0)
# Phase 2 of the hybrid ensure kernel scans a pool wider than this in tiles of this width.
_HYBRID_EVICT_CHUNK = 8192


def ensure_experts(cache, layer_id: int, expert_ids: torch.Tensor) -> None:
    """Make this layer's routed experts resident; rewrite ``expert_ids`` to slot ids.

    Delegates to flashlib's slot cache. ``id_base`` maps this layer's expert ids into the
    flat ``layer * num_experts + expert`` space the cache indexes by, and maps
    ``src_indices`` back, so ``copy_missing`` still resolves against this layer's own host
    tensor. ``out_indices`` aliases the input, preserving the in-place rewrite every
    downstream GEMM depends on.
    """
    id_of_slot, usage = cache.pool_state(layer_id)
    lru_ensure(
        expert_ids,
        cache.slot_for_id.view(-1),
        id_of_slot,
        usage,
        cache.step,
        expert_ids,
        cache.src_indices,
        cache.evict_slots,
        cache.num_indices,
        stats=cache.lru_stats[layer_id] if cache.collect_stats else None,
        id_base=layer_id * cache.num_experts,
    )


class HybridRoutes(NamedTuple):
    """One layer's routes split between the GPU and the CPU, each [bs, top_k]."""

    gpu_slots: torch.Tensor  # cache slot of each route; a CPU route is -1, or 0 with clamp
    gpu_w: torch.Tensor  # router weights, 0 for a CPU route
    cpu_ids: torch.Tensor  # expert ids, -1 for a GPU route


def _frac_q16(fetch_fraction: float) -> int:
    """Q16 fixed point of the fetch share; a share above 0 keeps at least one unit, so it never reads as the fixed cap."""
    return min(1 << 16, max(int(fetch_fraction > 0), round(fetch_fraction * (1 << 16))))


def ensure_experts_hybrid(
    cache,
    layer_id: int,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    max_fetch: int,
    fetch_fraction: float = 0.0,
    clamp: bool = False,
    fetch_cap: int | None = None,
    route_src: torch.Tensor | None = None,
) -> HybridRoutes:
    """Capped-fetch variant of ``ensure_experts`` (hybrid backend).

    Identical LRU bookkeeping, but only the first ``max_fetch`` of this step's missing
    experts are given a slot and scheduled for copy; the overflow misses stay
    non-resident and their ``expert_ids`` positions are rewritten to ``-1`` (compute on
    the CPU). ``fetch_fraction`` > 0 replaces the fixed cap with the bandwidth-matched
    split (fraction = pcie_bw / cpu_bw): fetch ~fraction of the step's misses, rounded as
    ``FREETOKEN_HYBRID_FETCH_ROUND`` says (default: the integer that makes the PCIe fetch and the CPU overflow compute finish closest together).
    ``num_indices`` = capped fetch count (copy_missing); ``num_missing_full`` =
    pre-cap miss count (stats).

    The same launch splits the routes: ``expert_ids`` becomes the GPU slots in place, and
    ``clamp`` gives CPU routes slot 0 instead of -1 for GEMM kernels that read every slot.
    ``fetch_cap`` bounds the fetch count after ``max_fetch`` / ``fetch_fraction`` (a fixed-size staging buffer).
    ``route_src`` (int32 per row of ``expert_ids``) first gives each row the experts of its source row; a row redirected
    to another one computes none of its routes (slot -1 or 0 with ``clamp``, weight 0, CPU id -1)."""
    assert expert_ids.is_cuda, "the hybrid ensure kernel runs on the GPU"
    # Q16 fixed point: the kernel caps in integers.
    frac_q16 = _frac_q16(fetch_fraction)
    weights = weights.contiguous()
    routes = HybridRoutes(expert_ids, torch.empty_like(weights), torch.empty_like(expert_ids))
    cap = cache.num_experts if fetch_cap is None else fetch_cap
    _ensure_experts_hybrid_gpu(cache, layer_id, expert_ids, weights, routes, clamp, max_fetch, frac_q16, cap, route_src)
    return routes


def prefill_hit_compact(cache, layer_id: int) -> None:
    """Compact this layer's cache-resident experts into gather indices, device-side.

    hit = slot_for_id[layer_id][e] >= the pool's staging threshold (the double buffer owns the
    slots below, so those bytes are volatile within a prefill chunk and classify
    as miss). Writes fixed-shape ``_prefill_hit_dst``/``_prefill_hit_src`` (buffer
    row / pool-local slot) and the count into ``_prefill_hit_num``; one launch on the
    current stream, no host sync. Safe against the concurrent buffer invalidation
    on the copy stream: that only rewrites entries already below the threshold."""
    num_experts = cache.num_experts
    _prefill_hit_compact_kernel[(1,)](
        cache.slot_for_id[layer_id],
        cache._prefill_hit_dst,
        cache._prefill_hit_src,
        cache._prefill_hit_num,
        cache._pool_stable_from[cache.pool_of_layer[layer_id]],
        num_experts,
        BLOCK=triton.next_power_of_2(num_experts),
    )


def materialize_layer(cache, layer_id: int) -> None:
    if len(cache.pool_sizes) == 1:
        _materialize_layer_gpu(cache, layer_id)
        return
    # staging buffer 0 may overlap several pools, so it only holds the layer for this forward and registers nothing
    E = cache.num_experts
    cache._invalidate_prefill_buffer(0, layer_id)
    rows = torch.arange(E, dtype=torch.int32, device=cache.device)
    cache.evict_slots[:E].copy_(rows)
    cache.src_indices[:E].copy_(rows)
    cache.num_indices.fill_(E)


def reset_cache(cache) -> None:
    _reset_cache_gpu(cache)






def _ensure_experts_hybrid_gpu(
    cache,
    layer_id: int,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    routes: HybridRoutes,
    clamp: bool,
    max_fetch: int,
    frac_q16: int,
    fetch_cap: int,
    route_src: torch.Tensor | None = None,
) -> None:
    id_of_slot, usage = cache.pool_state(layer_id)
    num_slots = id_of_slot.numel()
    block_e = triton.next_power_of_2(cache.num_experts)
    chunk = min(_HYBRID_EVICT_CHUNK, triton.next_power_of_2(num_slots))
    # with 4 warps a 4096+ chunk spills on sm80-sm89
    num_warps = 8 if chunk >= 4096 else 4
    ready = getattr(cache, "_ce_ready_pool", None)
    p = cache.pool_of_layer[layer_id]
    lfu = _HYBRID_LFU_TAU > 0 and num_slots <= chunk
    freq, last = cache.pool_lfu(layer_id) if lfu else (id_of_slot, id_of_slot)
    _ensure_experts_hybrid_kernel[(1,)](
        expert_ids,
        cache.slot_for_id,
        id_of_slot,
        usage,
        ready[p] if ready is not None else id_of_slot,
        cache._ce_gen_pool[p] if ready is not None else id_of_slot,
        cache.step,
        cache.active_mask,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        cache.num_missing_full,
        cache.expert_recency,
        weights,
        routes.cpu_ids,
        routes.gpu_w,
        *(cache.hybrid_stats if cache.collect_stats else (expert_ids,) * 2),
        layer_id,
        expert_ids.numel(),
        int(max_fetch),
        int(frac_q16),
        int(fetch_cap),
        cache.num_experts,
        num_slots,
        _HYBRID_FETCH_SEED,
        freq,
        last,
        _HYBRID_LFU_TAU if lfu else 1.0,
        route_src if route_src is not None else id_of_slot,
        expert_ids.shape[-1] if expert_ids.dim() > 1 else 1,
        BLOCK_E=block_e,
        CHUNK=chunk,
        SMALL=num_slots <= chunk,
        BY_RECENCY=_HYBRID_FETCH_BY_RECENCY,
        ROUND=_HYBRID_FETCH_ROUND,
        CLAMP=clamp,
        STATS=cache.collect_stats,
        PENDING=ready is not None,
        LFU=lfu,
        ROUTE_SRC=route_src is not None,
        num_warps=num_warps,
    )


def _materialize_layer_gpu(cache, layer_id: int) -> None:
    block = triton.next_power_of_2(max(cache.num_experts, cache.cache_size))
    _materialize_layer_kernel[(1,)](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        layer_id,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


def _reset_cache_gpu(cache) -> None:
    block = 256
    total_ids = cache.num_layers * cache.num_experts
    grid = (triton.cdiv(max(total_ids, cache.cache_size), block),)
    _reset_cache_kernel[grid](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.num_indices,
        total_ids,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


@triton.jit
def _reset_cache_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    active_mask_ptr,
    num_indices_ptr,
    total_ids: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(slot_for_id_ptr + off, -1, mask=off < total_ids)
    tl.store(id_of_slot_ptr + off, -1, mask=off < cache_size)
    tl.store(usage_ptr + off, 0, mask=off < cache_size)
    tl.store(active_mask_ptr + off, 0, mask=off < num_experts)
    if tl.program_id(0) == 0:
        tl.store(step_ptr, 0)
        tl.store(num_indices_ptr, 0)


@triton.jit
def _materialize_layer_kernel(
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    step_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    layer_id: tl.constexpr,
    num_experts: tl.constexpr,
    cache_size: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.arange(0, BLOCK)
    expert_mask = off < num_experts
    slot_mask = off < cache_size
    slot = off

    base = layer_id * num_experts
    old_id = tl.load(id_of_slot_ptr + slot, mask=slot_mask, other=-1)
    # Flat ids make "belongs to this layer" a range check instead of a field compare.
    same_layer = slot_mask & (old_id >= base) & (old_id < base + num_experts)
    tl.store(id_of_slot_ptr + slot, -1, mask=same_layer)
    tl.store(usage_ptr + slot, 0, mask=same_layer)

    old_valid = expert_mask & (old_id >= 0) & (~same_layer)
    tl.store(slot_for_id_ptr + old_id, -1, mask=old_valid)

    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    tl.store(id_of_slot_ptr + slot, base + off, mask=expert_mask)
    tl.store(slot_for_id_ptr + base + off, slot, mask=expert_mask)
    tl.store(usage_ptr + slot, step, mask=expert_mask)
    tl.store(evict_slots_ptr + off, slot, mask=expert_mask)
    tl.store(src_indices_ptr + off, off, mask=expert_mask)  # layer-local row
    tl.store(num_indices_ptr, num_experts)




@triton.jit(do_not_specialize=["layer_id", "num_active", "max_fetch", "fetch_frac_q16", "fetch_cap", "cache_size", "id_of_slot_ptr", "usage_ptr", "ready_ptr", "gen_ptr", "top_k"])
def _ensure_experts_hybrid_kernel(
    expert_ids_ptr,
    slot_for_id_ptr,
    id_of_slot_ptr,
    usage_ptr,
    ready_ptr,
    gen_ptr,
    step_ptr,
    active_mask_ptr,
    evict_slots_ptr,
    src_indices_ptr,
    num_indices_ptr,
    num_missing_full_ptr,
    expert_recency_ptr,
    weights_ptr,
    cpu_ids_ptr,
    gpu_w_ptr,
    stats_ptr,
    stats_layer_ptr,
    layer_id,
    num_active,
    max_fetch,
    fetch_frac_q16,
    fetch_cap,
    num_experts: tl.constexpr,
    cache_size,
    seed,
    freq_ptr,
    last_ptr,
    lfu_tau,
    route_src_ptr,
    top_k,
    BLOCK_E: tl.constexpr,
    CHUNK: tl.constexpr,
    SMALL: tl.constexpr,
    BY_RECENCY: tl.constexpr,
    ROUND: tl.constexpr,
    CLAMP: tl.constexpr,
    STATS: tl.constexpr,
    PENDING: tl.constexpr,
    LFU: tl.constexpr,
    ROUTE_SRC: tl.constexpr,
):
    """Capped-fetch timestamp-LRU (hybrid backend).

    Same as ``_ensure_experts_lru_v2_kernel`` but only ``min(num_missing, max_fetch)``
    missing experts are evicted-into / scheduled for copy; the overflow misses stay
    non-resident, so Phase 3 rewrites their positions to -1 (the layer computes those on
    the CPU). ``fetch_frac_q16`` > 0 (Q16 fixed point) replaces the fixed cap with the
    bandwidth-matched split ``~frac * num_missing`` (see the Phase-1 comment), computed
    in-kernel because ``num_missing`` only exists device-side (CUDA graph). ``num_indices``
    = the capped fetch count (copy_missing), ``num_missing_full`` = the pre-cap miss count
    (stats).

    Which misses to fetch is the cap policy. ``BY_RECENCY`` (default) fetches the experts
    most-recently active before this step (LRU on the expert, via ``expert_recency``),
    breaking ties toward the lower expert id -- this prioritizes *recurring* misses for
    caching, lowering the steady miss rate. Otherwise the lowest expert ids are fetched
    (``missing_rank``), the original routing-blind heuristic.

    Victims are the least-used slots, ties to the lowest slot. Every usage is at most the
    step before this launch, so the hit slots stamped with ``step`` in Phase 1 are never
    picked. A pool of at most CHUNK slots stays in registers (SMALL); a larger one is scanned
    from memory in CHUNK tiles per victim.

    ``PENDING`` (copy-engine demand copy): a slot whose ``ready`` lags its ``gen`` is still being
    written. Its expert is neither a hit nor a miss (its route goes to the CPU), and the slot is
    no victim; every victim's ``gen`` steps on, so it stays pending until the copy thread
    stamps it, and ``num_indices`` is the count of victims actually found."""
    step = tl.load(step_ptr) + 1
    tl.store(step_ptr, step)
    base = layer_id * num_experts

    if ROUTE_SRC:
        # a row whose source is another row routes to that row's experts; sources map to themselves
        for i in tl.range(num_active):
            src = tl.load(route_src_ptr + i // top_k) * top_k + i % top_k
            e = tl.load(expert_ids_ptr + src)
            tl.store(expert_ids_ptr + i, e, mask=src != i)
        tl.debug_barrier()

    # ---- Phase 1: active + missing over experts ----
    off_e = tl.arange(0, BLOCK_E)
    e_mask = off_e < num_experts
    is_active = tl.zeros((BLOCK_E,), dtype=tl.int1)
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        is_active = is_active | (off_e == e)
    tl.store(active_mask_ptr + off_e, is_active.to(tl.int32), mask=e_mask)
    slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
    is_missing = is_active & (slot == -1) & e_mask
    num_missing = tl.sum(is_missing.to(tl.int32))
    if PENDING:
        held = is_active & (slot >= 0) & e_mask
        rd_e = tl.load(ready_ptr + slot, mask=held, other=0, volatile=True)
        gn_e = tl.load(gen_ptr + slot, mask=held, other=0)
        pend = held & (rd_e != gn_e)
    else:
        pend = tl.zeros((BLOCK_E,), dtype=tl.int1)
    num_pend = tl.sum(pend.to(tl.int32))
    # Cap the fetches; the overflow misses are computed on the CPU (left non-resident).
    if fetch_frac_q16 > 0:
        # Bandwidth-matched split (fetch_frac = pcie_bw / cpu_bw): fetch time scales with
        # F * (1 - frac), CPU time with (M - F) * frac; they balance at F = frac * M. Pick
        # the integer neighbor that minimizes the slower (max) side of the overlap.
        lo = (num_missing * fetch_frac_q16) >> 16
        if ROUND == 1:
            # step is unique per launch, so each (layer, step) draws its own number
            rem = (num_missing * fetch_frac_q16) & 0xFFFF
            u = tl.rand(seed, (step & 0x7FFFFFFF).to(tl.int32))
            max_fetch = lo + ((u * 65536.0) < rem.to(tl.float32)).to(tl.int32)
        elif ROUND == 2:
            max_fetch = (num_missing * fetch_frac_q16 + 0xFFFF) >> 16
        else:
            cost_lo = tl.maximum(lo * ((1 << 16) - fetch_frac_q16), (num_missing - lo) * fetch_frac_q16)
            cost_hi = tl.maximum(
                (lo + 1) * ((1 << 16) - fetch_frac_q16), (num_missing - lo - 1) * fetch_frac_q16
            )
            max_fetch = tl.where(cost_lo <= cost_hi, lo, lo + 1)
    is_hit = is_active & (slot >= 0) & ~pend
    # a pool smaller than num_experts may hold fewer free slots than misses
    num_fetch = tl.minimum(tl.minimum(num_missing, tl.minimum(max_fetch, fetch_cap)), cache_size - tl.sum(is_hit.to(tl.int32)))
    tl.store(num_missing_full_ptr, (num_missing + num_pend).to(tl.int64))
    tl.store(num_indices_ptr, num_fetch.to(tl.int64))
    if LFU:
        # key = last step + tau * log2(decayed hits): argmin by key is argmin by decayed frequency
        f0 = tl.load(freq_ptr + slot, mask=is_hit, other=0.0)
        t0 = tl.load(last_ptr + slot, mask=is_hit, other=0)
        f1 = f0 * tl.exp2(-(step - t0).to(tl.float32) / lfu_tau) + 1.0
        tl.store(freq_ptr + slot, f1, mask=is_hit)
        tl.store(last_ptr + slot, step + tl.zeros_like(t0), mask=is_hit)
        tl.store(usage_ptr + slot, step + (lfu_tau * tl.log2(f1)).to(tl.int64), mask=is_hit)
    else:
        tl.store(usage_ptr + slot, step, mask=is_hit)

    # Fetch-selection priority: encode (recency desc, id asc) into one strictly-ordered
    # score so argmax has no ties (rec deltas are multiples of num_experts; the id term
    # spans only [0, num_experts), so it can only break exact-recency ties).
    if BY_RECENCY:
        rec = tl.load(expert_recency_ptr + base + off_e, mask=e_mask, other=-1).to(tl.int64)
        score = tl.where(
            is_missing, rec * num_experts + (num_experts - 1 - off_e), -1152921504606846976
        ).to(tl.int64)
    else:
        missing_rank = tl.cumsum(is_missing.to(tl.int32)) - 1

    # ---- Phase 2: evict victims by argmin(usage), only for the capped fetches ----
    n = tl.zeros((), dtype=tl.int32)
    if num_fetch > 0:
        off_k = tl.arange(0, CHUNK)
        tl.debug_barrier()  # the scan below must see the hit stamps
        if SMALL:
            u_reg = tl.load(usage_ptr + off_k, mask=off_k < cache_size, other=9223372036854775807).to(tl.int64)
            if PENDING:
                rd = tl.load(ready_ptr + off_k, mask=off_k < cache_size, other=0, volatile=True)
                gn = tl.load(gen_ptr + off_k, mask=off_k < cache_size, other=0)
                u_reg = tl.where(rd != gn, 9223372036854775807, u_reg)
            if LFU:
                # a hit's key need not top the pool's: keep this step's hits out by hand
                hit_slot = tl.where(is_hit, slot, -1)
                for j in tl.range(num_active):
                    hs = tl.max(tl.where(off_e == tl.load(expert_ids_ptr + j), hit_slot, -1), axis=0)
                    u_reg = tl.where(off_k == hs, 9223372036854775807, u_reg)
        for i in tl.range(num_fetch):
            if SMALL:
                victim = tl.argmin(u_reg, axis=0).to(tl.int32)
                best_u = tl.min(u_reg, axis=0)
                if LFU:
                    u_reg = tl.where(off_k == victim, 9223372036854775807, u_reg)
                else:
                    u_reg = tl.where(off_k == victim, tl.maximum(u_reg, step), u_reg)
            else:
                best_u = tl.full((), 9223372036854775807, tl.int64)
                best_s = tl.full((), 0, tl.int32)
                for c0 in tl.range(0, cache_size, CHUNK):
                    offs = c0 + off_k
                    uu = tl.load(usage_ptr + offs, mask=offs < cache_size, other=9223372036854775807).to(tl.int64)
                    if PENDING:
                        rd = tl.load(ready_ptr + offs, mask=offs < cache_size, other=0, volatile=True)
                        gn = tl.load(gen_ptr + offs, mask=offs < cache_size, other=0)
                        uu = tl.where(rd != gn, 9223372036854775807, uu)
                    cmin = tl.min(uu, axis=0)
                    carg = (tl.argmin(uu, axis=0) + c0).to(tl.int32)
                    take = cmin < best_u
                    best_s = tl.where(take, carg, best_s)
                    best_u = tl.where(take, cmin, best_u)
                victim = best_s
            if LFU:
                free = best_u < 140737488355328
            elif PENDING:
                # every usage but this step's hits and victims lies below step
                free = best_u < step
            else:
                free = tl.full((), 1, tl.int1)
            old_id = tl.load(id_of_slot_ptr + victim)
            tl.store(slot_for_id_ptr + old_id, -1, mask=free & (old_id >= 0))
            if BY_RECENCY:
                e = tl.argmax(score, axis=0).to(tl.int32)
                score = tl.where(off_e == e, -1152921504606846976, score)
            else:
                e = tl.sum(tl.where((missing_rank == i) & is_missing, off_e, 0))
            tl.store(id_of_slot_ptr + victim, base + e, mask=free)
            tl.store(slot_for_id_ptr + base + e, victim, mask=free)
            tl.store(usage_ptr + victim, step, mask=free)
            if LFU:
                tl.store(freq_ptr + victim, 1.0, mask=free)
                tl.store(last_ptr + victim, step, mask=free)
            if PENDING:
                tl.store(gen_ptr + victim, tl.load(gen_ptr + victim) + 1, mask=free)
            tl.store(evict_slots_ptr + n, victim, mask=free)
            tl.store(src_indices_ptr + n, e, mask=free)  # layer-local row
            n = n + free.to(tl.int32)
            tl.debug_barrier()  # the next scan must see this victim's stamp
    if PENDING:
        tl.store(num_indices_ptr, n.to(tl.int64))
    tl.debug_barrier()  # Phase 3 reads slot_for_id entries written above

    # ---- Phase 3: rewrite expert_ids -> slot id (hit/fetched) or -1 (overflow -> CPU) ----
    for i in tl.range(num_active):
        e = tl.load(expert_ids_ptr + i)
        s = tl.load(slot_for_id_ptr + base + e)
        # every warp loads e, one thread stores below; a warp that loads after that store indexes with a slot id
        tl.debug_barrier()
        on_gpu = s >= 0
        if PENDING:
            on_gpu = on_gpu & (tl.sum(tl.where(off_e == e, pend.to(tl.int32), 0)) == 0)
            s = tl.where(on_gpu, s, -1)
        if ROUTE_SRC:
            # a redirected row only shapes the step's expert set: none of its routes is computed
            live = tl.load(route_src_ptr + i // top_k) * top_k + i % top_k == i
            s = tl.where(live, s, -1)
            tl.store(cpu_ids_ptr + i, tl.where(on_gpu | ~live, -1, e))
            tl.store(gpu_w_ptr + i, tl.where(on_gpu & live, tl.load(weights_ptr + i), 0.0))
        else:
            tl.store(cpu_ids_ptr + i, tl.where(on_gpu, -1, e))
            tl.store(gpu_w_ptr + i, tl.where(on_gpu, tl.load(weights_ptr + i), 0.0))
        if CLAMP:
            s = tl.maximum(s, 0)
        tl.store(expert_ids_ptr + i, s)

    # stats_ptr: (missing, fetched, active, calls); stats_layer_ptr: the same per layer, [num_layers, 4]
    if STATS:
        n_active = tl.sum((is_active & e_mask).to(tl.int64))
        # a pending expert is computed on the CPU like an overflow miss
        tl.store(stats_ptr + 0, tl.load(stats_ptr + 0) + num_missing + num_pend)
        tl.store(stats_ptr + 1, tl.load(stats_ptr + 1) + n)
        tl.store(stats_ptr + 2, tl.load(stats_ptr + 2) + n_active)
        tl.store(stats_ptr + 3, tl.load(stats_ptr + 3) + 1)
        row = stats_layer_ptr + layer_id * 4
        tl.store(row + 0, tl.load(row + 0) + num_missing + num_pend)
        tl.store(row + 1, tl.load(row + 1) + n)
        tl.store(row + 2, tl.load(row + 2) + n_active)
        tl.store(row + 3, tl.load(row + 3) + 1)

    # Bump every active expert's recency to this step (LRU on the expert): an overflow miss
    # computed on the CPU now ranks high if it recurs, so it gets fetched next time.
    if BY_RECENCY:
        step_vec = tl.zeros((BLOCK_E,), dtype=tl.int64) + step
        tl.store(expert_recency_ptr + base + off_e, step_vec, mask=is_active & e_mask)


@triton.jit(do_not_specialize=["threshold"])
def _prefill_hit_compact_kernel(
    slot_ptr,     # [num_experts] int32: this layer's slot_for_id row
    dst_ptr,      # [num_experts] int32 out: buffer rows (expert ids), compacted
    src_ptr,      # [num_experts] int32 out: pool-local slots, compacted
    num_ptr,      # [1] int64 out: hit count
    threshold,    # first pool-local slot outside the staging buffers
    num_experts,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    lane = offs < num_experts
    slots = tl.load(slot_ptr + offs, mask=lane, other=-1)
    is_hit = lane & (slots >= threshold)
    pos = tl.cumsum(is_hit.to(tl.int32)) - 1
    tl.store(dst_ptr + pos, offs.to(tl.int32), mask=is_hit)
    tl.store(src_ptr + pos, slots, mask=is_hit)
    tl.store(num_ptr, tl.sum(is_hit.to(tl.int64)))


@triton.jit
def _routed_slot_alphas_kernel(
    slot_ptr, id_of_slot_ptr, gate_up_ptr, down_ptr, out_gate_up_ptr, out_down_ptr,
    n, base, num_experts: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    slot = tl.load(slot_ptr + offs, mask=offs < n, other=-1)
    routed = (offs < n) & (slot >= 0)
    flat = tl.load(id_of_slot_ptr + slot, mask=routed, other=0)
    idx = base + tl.maximum(flat, 0) % num_experts
    tl.store(out_gate_up_ptr + slot, tl.load(gate_up_ptr + idx, mask=routed), mask=routed)
    tl.store(out_down_ptr + slot, tl.load(down_ptr + idx, mask=routed), mask=routed)


def routed_slot_alphas(cache, layer_id: int, slot_ids: torch.Tensor) -> None:
    """Write the global scales of ``slot_ids``' experts (local slots of ``layer_id``'s pool) into the per-slot
    buffers (same values ``alphas_for_slots`` gathers for every slot, only where they are read)."""
    flat_ids = slot_ids.reshape(-1)
    n = flat_ids.numel()
    id_of_slot, _ = cache.pool_state(layer_id)
    _routed_slot_alphas_kernel[(1,)](
        flat_ids, id_of_slot, cache.gate_up_alpha, cache.down_alpha,
        cache._slot_gate_up_alpha, cache._slot_down_alpha,
        n, layer_id * cache.num_experts, cache.num_experts,
        BLOCK=triton.next_power_of_2(max(n, 16)),
    )


@triton.jit
def _split_fetched_routes_kernel(
    slot_ptr, w_ptr, evict_ptr, num_ptr, hit_slot_ptr, hit_w_ptr, new_slot_ptr, new_w_ptr, n,
    HOLE: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    lane = offs < n
    slot = tl.load(slot_ptr + offs, mask=lane, other=0)
    w = tl.load(w_ptr + offs, mask=lane, other=0.0)
    num = tl.load(num_ptr).to(tl.int32)
    fetched = tl.zeros((BLOCK,), dtype=tl.int1)
    for j in tl.range(num):
        fetched = fetched | (slot == tl.load(evict_ptr + j))
    tl.store(hit_slot_ptr + offs, tl.where(fetched, HOLE, slot), mask=lane)
    tl.store(hit_w_ptr + offs, tl.where(fetched, 0.0, w), mask=lane)
    tl.store(new_slot_ptr + offs, tl.where(fetched, slot, HOLE), mask=lane)
    tl.store(new_w_ptr + offs, tl.where(fetched, w, 0.0), mask=lane)


def split_fetched_routes(cache, slots: torch.Tensor, weights: torch.Tensor, hole: int = 0):
    """(hit slots, hit weights, fetched slots, fetched weights): a route is fetched when its slot is one this step's
    ensure scheduled for copy (``evict_slots[:num_indices]``); each half keeps the other's routes at slot ``hole``
    (0, or -1 for a kernel that skips inactive slots), weight 0."""
    flat_s, flat_w = slots.reshape(-1).contiguous(), weights.reshape(-1).contiguous()
    out = [torch.empty_like(flat_s), torch.empty_like(flat_w), torch.empty_like(flat_s), torch.empty_like(flat_w)]
    n = flat_s.numel()
    _split_fetched_routes_kernel[(1,)](
        flat_s, flat_w, cache.evict_slots, cache.num_indices, *out, n, HOLE=hole, BLOCK=triton.next_power_of_2(max(n, 16))
    )
    return tuple(t.view(slots.shape) for t in out)
