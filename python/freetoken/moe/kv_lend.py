"""Lend the idle tail of the paged KV pool to the MoE slot cache.

The KV buffer is built from physical chunks (kernel/vmm.py). Chunks that hold only high pages of
one KV slab are mapped a second time behind extra slots at the end of the expert pool. KV may use
pages below ``boundary``; the extra slots whose bytes lie at or above it are live, the rest are
unusable (``UNAVAILABLE`` usage, never an LRU victim). The page allocator asks for pages past the
boundary with ``grow_kv`` (evicting the slots over them) and gives them back with ``shrink_kv``.
Only which experts sit in VRAM changes, never what a kernel computes.

``FREETOKEN_KV_LEND=1`` turns it on; ``FREETOKEN_KV_LEND_MARGIN`` is the KV headroom in tokens kept
free below the boundary (default 4096). ``FREETOKEN_KV_LEND_GDN=1`` also lends the GDN state slots no
running request holds, the same way (a second borrowed segment, its boundary in state slots).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

# fewer than this many slots gained is not worth a lend
_MIN_GAIN_SLOTS = 8


def kv_lend_enabled() -> bool:
    return os.environ.get("FREETOKEN_KV_LEND", "0").strip().lower() in {"1", "true", "yes", "on"}


def gdn_lend_enabled() -> bool:
    return os.environ.get("FREETOKEN_KV_LEND_GDN", "0").strip().lower() in {"1", "true", "yes", "on"}


def kv_lend_margin_tokens() -> int:
    return max(1, int(os.environ.get("FREETOKEN_KV_LEND_MARGIN", "4096") or 0))


@dataclass(frozen=True)
class LendPlan:
    granule: int
    base_slots: int
    # per bank: chunks of the bank's own memory at the front of its address range
    own_chunks: tuple[int, ...]
    # per bank: the KV chunk behind each address chunk past the own ones
    ext_chunks: tuple[tuple[int, ...], ...]
    # per extra slot, non-increasing: the slot's bytes lie in KV pages at or above this
    thresholds: tuple[int, ...]

    @property
    def slots(self) -> int:
        return len(self.thresholds)


def plan_lend(
    *,
    slab_bytes: int,
    num_slabs: int,
    page_bytes: int,
    usable_pages: int,
    granule: int,
    base_slots: int,
    row_bytes: list[int],
    max_slots: int,
    min_page: int = 0,
) -> LendPlan:
    """Map extra slots ``base_slots..`` onto KV chunks, highest pages first.

    The KV buffer is ``num_slabs`` slabs of ``slab_bytes``, page ``p`` at ``p * page_bytes`` in each; pages at
    or above ``usable_pages`` (the dummy page) or below ``min_page`` are never lent, nor is a chunk that spans
    two slabs. Each bank row of ``row_bytes`` keeps every chunk that holds a base slot's byte in its own memory."""
    first_page = {}
    for c in range(num_slabs * slab_bytes // granule):
        lo, hi = c * granule, (c + 1) * granule - 1
        slab = lo // slab_bytes
        first = (lo - slab * slab_bytes) // page_bytes
        if hi // slab_bytes != slab or (hi - slab * slab_bytes) // page_bytes >= usable_pages or first < min_page:
            continue
        first_page[c] = first
    order = sorted(first_page, key=lambda c: (-first_page[c], c))
    own = [-(-base_slots * r // granule) for r in row_bytes]
    backing: list[dict[int, int]] = [{} for _ in row_bytes]
    thresholds: list[int] = []
    taken = 0
    while base_slots + len(thresholds) < max_slots:
        slot = base_slots + len(thresholds)
        spans = [
            (k, v)
            for k, r in enumerate(row_bytes)
            if r
            for v in range(slot * r // granule, ((slot + 1) * r - 1) // granule + 1)
            if v >= own[k]
        ]
        new = [(k, v) for k, v in spans if v not in backing[k]]
        if taken + len(new) > len(order):
            break
        for k, v in new:
            backing[k][v] = order[taken]
            taken += 1
        t = min((first_page[backing[k][v]] for k, v in spans), default=usable_pages)
        thresholds.append(min(t, thresholds[-1]) if thresholds else t)
    end = base_slots + len(thresholds)
    ext = tuple(
        tuple(backing[k][v] for v in range(own[k], -(-end * r // granule))) for k, r in enumerate(row_bytes)
    )
    return LendPlan(granule, base_slots, tuple(own), ext, tuple(thresholds))


class KvLend:
    """Where the boundary between KV pages and borrowed expert slots sits, and moving it."""

    def __init__(self, cache, plan: LendPlan, usable_pages: int, margin: int, segment: int = 0, name: str = "KV") -> None:
        self.cache = cache
        self.thresholds = plan.thresholds
        self.usable_pages = usable_pages
        self.margin = margin
        self.segment = segment
        self.name = name
        self.slots = 0
        self._set(self.slots_for(margin))

    @property
    def capacity(self) -> int:
        return len(self.thresholds)

    @property
    def boundary(self) -> int:
        """KV may use pages below this one."""
        return self.boundary_for(self.slots)

    def boundary_for(self, slots: int) -> int:
        return self.thresholds[slots - 1] if slots else self.usable_pages

    def slots_for(self, boundary: int) -> int:
        """The most borrowed slots that lie entirely at or above page ``boundary``."""
        lo, hi = 0, len(self.thresholds)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.thresholds[mid] >= boundary:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def grow_kv(self, pages: int) -> int:
        """KV wants ``pages`` more pages past the boundary: give up the slots over them and the margin; returns the new boundary."""
        slots = self.slots_for(self.boundary + pages + self.margin)
        if slots < self.slots:
            self._set(slots)
        return self.boundary

    def shrink_kv(self, highest_used: int) -> int:
        """No KV page above ``highest_used`` is held: lend what lies past it and the margin; returns the new boundary."""
        slots = self.slots_for(highest_used + 1 + self.margin)
        if slots >= self.slots + _MIN_GAIN_SLOTS:
            self._set(slots)
        return self.boundary

    def lend_gap(self) -> int:
        """Free pages below the boundary a worthwhile lend needs on top of the margin; 0 if none is possible."""
        target = self.slots + _MIN_GAIN_SLOTS
        if target > self.capacity:
            return 0
        return self.boundary - self.boundary_for(target) + self.margin

    def _set(self, slots: int) -> None:
        old = self.boundary
        cuda = self.cache.device.type == "cuda"
        if cuda:
            torch.cuda.synchronize(self.cache.device)
        self.cache.set_live_slots(slots, self.segment)
        self.slots = slots
        if cuda:
            torch.cuda.synchronize(self.cache.device)
        logger.info_rank0(
            f"{self.name} lend: {slots} of {self.capacity} borrowed MoE slots live, {self.name} boundary {old} -> {self.boundary} of {self.usable_pages}"
        )


def _unusable_reason(cache, kv_pool) -> str | None:
    from freetoken.distributed import get_tp_info
    from freetoken.kvcache.mha_pool import MHAKVCache

    if cache.device.type != "cuda":
        return "the MoE cache is not on a CUDA device"
    if get_tp_info().size != 1:
        return "tensor parallel"
    if type(kv_pool) is not MHAKVCache:
        return f"the {type(kv_pool).__name__} KV pool is not a plain paged pool"
    if len(cache.pool_sizes) != 1 or cache._raw_banks:
        return "the MoE cache has several slot pools or padded bank layouts"
    if cache.decode_target not in ("gpu", "hybrid") or cache.locked_hybrid:
        return f"decode target {cache.decode_target!r} does not read the GPU slot cache"
    if cache.cache_size < 2 * cache.num_experts:
        return "the prefill staging buffers would reach the borrowed slots"
    try:
        import cuda.bindings.driver  # noqa: F401
    except ImportError:
        return "cuda-python (cuda.bindings) is not installed"
    return None


def _align(slots: int, row_bytes: list[int], granule: int) -> int:
    """The first slot from ``slots`` on where every bank row starts on a chunk boundary."""
    step = math.lcm(*(granule // math.gcd(granule, r) for r in row_bytes if r))
    return -(-slots // step) * step


def attach_kv_lend(cache, kv_pool, *, usable_pages: int, page_size: int, linear_pool=None) -> KvLend | None:
    """Rebuild the KV buffer (and with ``linear_pool`` the GDN recurrent state) on chunks and grow the MoE pool by the
    slots they can back; before graph capture."""
    from freetoken.kernel.vmm import granularity

    if reason := _unusable_reason(cache, kv_pool):
        logger.warning_rank0(f"KV lend off: {reason}")
        return None
    two, layers, pages, ps, heads, dim = kv_pool._kv_buffer.shape
    page_bytes = ps * heads * dim * kv_pool._kv_buffer.element_size()
    limit = cache.max_slots if cache.max_slots is not None else 1 << 15
    granule = granularity(cache.device)
    rows = list(cache._row_bytes[0])
    plan = plan_lend(
        slab_bytes=pages * page_bytes, num_slabs=two * layers, page_bytes=page_bytes, usable_pages=usable_pages,
        granule=granule, base_slots=cache.cache_size, row_bytes=rows, max_slots=limit,
    )
    if not plan.slots:
        logger.warning_rank0("KV lend off: no KV chunk can back a whole expert slot")
        return None
    segments = [(plan, kv_pool.move_to_chunks())]
    gdn = None
    if linear_pool is not None:
        state = linear_pool.recurrent_states
        n_layers, num_slots = state.shape[:2]
        gdn = plan_lend(
            slab_bytes=state[0].numel() * state.element_size(), num_slabs=n_layers,
            page_bytes=state[0, 0].numel() * state.element_size(), usable_pages=num_slots, granule=granule,
            base_slots=_align(plan.base_slots + plan.slots, rows, granule), row_bytes=rows, max_slots=limit, min_page=1,
        )
        if gdn.slots:
            segments.append((gdn, linear_pool.move_recurrent_to_chunks()))
        else:
            logger.warning_rank0("GDN lend off: no state chunk can back a whole expert slot")
            gdn = None
    cache.borrow_slots(segments)
    for (p, chunks), what in zip(segments, ("the KV tail", "idle GDN state slots")):
        logger.info_rank0(
            f"KV lend: MoE pool slots {p.base_slots}.. + up to {p.slots} borrowed from {what} "
            f"({sum(map(len, p.ext_chunks)) * granule / 2**30:.2f} GiB of {len(chunks) * granule / 2**30:.2f} GiB)"
        )
    lend = KvLend(cache, plan, usable_pages, -(-kv_lend_margin_tokens() // page_size))
    if gdn is not None:
        linear_pool.lend_slots_to(KvLend(cache, gdn, linear_pool.num_slots, 0, segment=1, name="GDN"))
    return lend


__all__ = ["KvLend", "LendPlan", "attach_kv_lend", "gdn_lend_enabled", "kv_lend_enabled", "plan_lend"]
