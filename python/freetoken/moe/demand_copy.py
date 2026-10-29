"""Copy-engine demand copy for hybrid decode.

``FREETOKEN_DEMAND_COPY=ce`` (with ``FREETOKEN_HYBRID_SPLIT_GEMM=copy`` and the CPU pool's flag coordinator) moves
the experts a hybrid step fetches with the copy engine instead of the SM pull. The hybrid ensure picks the victims
on the GPU and makes each one pending (``gen[slot] += 1`` while ``ready[slot]`` holds the old generation).
``publish`` snapshots them into a channel, one per (layer, lane), and writes them as ``(slot, expert, gen)`` into
the channel's pinned row, ``seq`` last with a system-scope release; the CPU pool's copy thread copies every bank row
on its own stream, in row order, then stamps ``ready[slot] = gen`` and ``ack[channel] = seq`` behind the copies. The
hit routes' GEMM runs meanwhile, then ``wait`` spins one program on ``ack`` and, past
``FREETOKEN_DEMAND_COPY_TIMEOUT_US`` (default 3000, 0 waits forever), pulls the rows itself with the SM copy. The
graph thus never depends on the copy thread: a host thread inside the driver can stall it (a lazy module load waits
for the GPU and holds back every copy queued meanwhile), and that only costs the timeout. A copy that lands after
such a fallback writes the same expert into a slot that is still pending, which nothing reads or reuses until its
stamp. A channel whose previous row is not acked yet publishes nothing: the SM copy pulls its rows and the GPU
stamps them, since no copy can be in flight into a slot that was not pending. Prefill, reset and rebuild drain the
copies first.

Lanes let one layer run several demand copies per step, e.g. one per verify row from its own stream: each lane's
``publish`` follows that row's ensure on the row's stream, its ``wait`` precedes that row's fetched-route GEMM, and
the copy thread serves lane 0 of a layer before lane 1 and both before the next layer.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import globaltimer

from freetoken.utils import init_logger

logger = init_logger(__name__)

_MAXB = 32
_WIDTH = 4 + 3 * _MAXB


def demand_copy_mode() -> str:
    return os.environ.get("FREETOKEN_DEMAND_COPY", "sm").strip().lower() or "sm"


class DemandCopy:
    max_fetch = _MAXB

    def __init__(self, cache, lanes: int | None = None):
        executor = cache.cpu_executor
        if executor is None or not getattr(executor, "_flag_sync", False):
            raise ValueError("the copy-engine demand copy needs the CPU pool's flag coordinator (hybrid decode)")
        if not cache._copy_fused_ok:
            raise ValueError("the copy-engine demand copy needs the fused copy plan for its fallback")
        self.cache = cache
        self.lanes = lanes or int(os.environ.get("FREETOKEN_DEMAND_COPY_LANES", "2"))
        self.timeout_ns = int(os.environ.get("FREETOKEN_DEMAND_COPY_TIMEOUT_US", "3000")) * 1000
        C, device = cache.num_layers * self.lanes, cache.device
        from freetoken.kernel.pinned import alloc_pinned_tensor

        self.rows = alloc_pinned_tensor(C, _WIDTH, dtype=torch.int32)
        self.rows.zero_()
        self.seq = torch.zeros(C, dtype=torch.int32, device=device)
        self.ack = torch.zeros(C, dtype=torch.int32, device=device)
        self.t_pub = torch.zeros(C, dtype=torch.int64, device=device)
        self.state = torch.zeros(C, dtype=torch.int32, device=device)
        self.slots = torch.zeros(C, _MAXB, dtype=torch.int32, device=device)
        self.experts = torch.zeros(C, _MAXB, dtype=torch.int32, device=device)
        self.num = torch.zeros(C, 1, dtype=torch.int64, device=device)
        self.fallback = torch.zeros(C, 1, dtype=torch.int64, device=device)
        # rows waited for, publish-to-ack ns, ns waited after the hit GEMM, timeouts, unpublished rows
        self.counters = torch.zeros(5, dtype=torch.int64, device=device)
        self._last = (0,) * 9
        self._alloc()

    def _alloc(self) -> None:
        cache = self.cache
        self.ready = torch.zeros(cache.cache_size, dtype=torch.int32, device=cache.device)
        self.gen = torch.zeros(cache.cache_size, dtype=torch.int32, device=cache.device)
        cut = list(zip(cache.pool_bases, cache.pool_sizes, strict=True))
        cache._ce_ready_pool = [self.ready[b : b + s] for b, s in cut]
        cache._ce_gen_pool = [self.gen[b : b + s] for b, s in cut]
        self._setup()

    def _setup(self) -> None:
        """Hand the copy plan to the copy thread; it serves every row whose ``seq`` it has not seen."""
        cache = self.cache
        dst = [[d + cache._pool_offsets[k][p] for k, d in enumerate(cache._copy_dst_ptrs_host)] for p in range(len(cache.pool_sizes))]
        layers = [c // self.lanes for c in range(cache.num_layers * self.lanes)]
        cache.cpu_executor._ext.ce_setup(
            self.rows.data_ptr(), _WIDTH, _MAXB, self.ready.data_ptr(), self.ack.data_ptr(),
            [list(map(int, cache._copy_src_ptrs_host[l])) for l in layers], dst, [list(map(int, r)) for r in cache._row_bytes],
            [cache.pool_of_layer[l] for l in layers], [cache.layer_layout[l] for l in layers], list(cache.pool_bases),
            cache.device.index or 0,
        )

    def serves(self, layer_id: int) -> bool:
        cache = self.cache
        return layer_id not in cache.locked_hybrid and not cache.is_unpinned_layer(layer_id)

    def publish(self, layer_id: int, lane: int = 0) -> None:
        """Right after this lane's ensure, on its stream: snapshot the victims and hand them to the copy thread."""
        c = layer_id * self.lanes + lane
        cache = self.cache
        _demand_publish_kernel[(1,)](
            cache.evict_slots, cache.src_indices, cache.num_indices, cache._ce_gen_pool[cache.pool_of_layer[layer_id]],
            self.slots[c], self.experts[c], self.num[c], self.seq[c:], self.ack[c:], self.state[c:], self.rows[c], self.t_pub[c:],
            layer_id, MAXB=_MAXB,
        )

    def wait(self, layer_id: int, lane: int = 0) -> None:
        """Before this lane's fetched-route GEMM, on its stream: the copies landed, or the SM copy pulled the rows."""
        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

        c = layer_id * self.lanes + lane
        cache = self.cache
        p = cache.pool_of_layer[layer_id]
        _demand_wait_kernel[(1,)](
            self.ack[c:], self.seq[c:], self.state[c:], self.num[c], self.slots[c], cache._ce_ready_pool[p], cache._ce_gen_pool[p],
            self.t_pub[c:], self.fallback[c], self.counters, self.timeout_ns, MAXB=_MAXB,
        )
        fast_index_copy_multi_jit(
            cache._copy_dst_ptrs_pool[p],
            cache._copy_src_ptrs[layer_id],
            cache._copy_feat_bytes[cache.layer_layout[layer_id]],
            self.slots[c],
            self.experts[c],
            self.fallback[c],
            blocks_per_bank=cache._copy_blocks_per_bank,
        )

    def drain(self) -> None:
        """Every published row copied and stamped: before anything else writes slots or slot maps."""
        torch.cuda.synchronize(self.cache.device)
        self.cache.cpu_executor._ext.ce_drain()

    def detach(self) -> None:
        """Before a rebuild frees the slots: drain, then stop the copy thread from serving the old rows."""
        self.drain()
        self.cache.cpu_executor._ext.ce_detach()

    def on_rebuild(self) -> None:
        """After a rebuild: the slots moved, so the rows restart from an empty ledger."""
        self.seq.zero_()
        self.ack.zero_()
        self.rows.zero_()
        self._alloc()

    def stats(self) -> tuple[int, ...]:
        """(rows, experts, bytes, issue ns) the copy thread served, then (rows waited, publish-to-ack ns, wait ns,
        timeouts, unpublished rows) on the GPU; cumulative."""
        return tuple(self.cache.cpu_executor._ext.ce_stats()) + tuple(self.counters.tolist())

    def stats_line(self) -> str | None:
        now = self.stats()
        rows, experts, nbytes, issue, waited, latency, spin, timeouts, unpublished = (a - b for a, b in zip(now, self._last, strict=True))
        self._last = now
        if not rows and not unpublished:
            return None
        rate = nbytes / latency if latency else 0.0
        return (
            f"ce demand: {experts} experts in {rows} rows, {nbytes / 2**20:.0f} MiB, issue {issue / max(rows, 1) / 1e3:.1f} us/row, "
            f"publish-to-ack {latency / max(waited, 1) / 1e3:.1f} us/row ({rate:.1f} GB/s), "
            f"waited {spin / max(waited, 1) / 1e3:.1f} us/row, {timeouts} timeouts, {unpublished} unpublished"
        )


@triton.jit(do_not_specialize=["layer"])
def _demand_publish_kernel(
    evict_ptr, src_ptr, num_ptr, gen_ptr, slots_ptr, experts_ptr, cnum_ptr, seq_ptr, ack_ptr, state_ptr, row_ptr, tpub_ptr,
    layer, MAXB: tl.constexpr,
):
    """``state``: 2 nothing to fetch, 1 published, 0 the previous row is not acked (the GPU pulls this one)."""
    n = tl.minimum(tl.load(num_ptr).to(tl.int32), MAXB)
    off = tl.arange(0, MAXB)
    m = off < n
    slots = tl.load(evict_ptr + off, mask=m, other=0)
    experts = tl.load(src_ptr + off, mask=m, other=0)
    tl.store(slots_ptr + off, slots)
    tl.store(experts_ptr + off, experts)
    tl.store(cnum_ptr, n.to(tl.int64))
    seq = tl.load(seq_ptr)
    ack = tl.load(ack_ptr, volatile=True)
    state = tl.where(n == 0, 2, tl.where(ack == seq, 1, 0))
    if state == 1:
        tl.store(row_ptr + 4 + off, slots, mask=m)
        tl.store(row_ptr + 4 + MAXB + off, experts, mask=m)
        tl.store(row_ptr + 4 + 2 * MAXB + off, tl.load(gen_ptr + slots, mask=m, other=0), mask=m)
        tl.store(seq_ptr, seq + 1)
        tl.store(row_ptr + 1, layer)
        tl.store(row_ptr + 2, n)
        tl.store(tpub_ptr, globaltimer())
        tl.debug_barrier()
        tl.atomic_xchg(row_ptr, seq + 1, sem="release", scope="sys")
    tl.store(state_ptr, state)


@triton.jit(do_not_specialize=["timeout_ns"])
def _demand_wait_kernel(
    ack_ptr, seq_ptr, state_ptr, num_ptr, slots_ptr, ready_ptr, gen_ptr, tpub_ptr, fallback_ptr, counters_ptr, timeout_ns,
    MAXB: tl.constexpr,
):
    """``fallback``: how many of the channel's slots the SM copy pulls after this, 0 once the copy thread acked."""
    state = tl.load(state_ptr)
    n = tl.load(num_ptr)
    pull = tl.zeros((), dtype=tl.int64)
    if state == 1:
        target = tl.load(seq_ptr)
        t0 = globaltimer()
        now = t0
        ack = tl.load(ack_ptr, volatile=True)
        while ((ack - target) < 0) & ((timeout_ns == 0) | (now - t0 < timeout_ns)):
            ack = tl.load(ack_ptr, volatile=True)
            now = globaltimer()
        if (ack - target) >= 0:
            tl.atomic_add(counters_ptr + 0, 1)
            tl.atomic_add(counters_ptr + 1, (now - tl.load(tpub_ptr)))
            tl.atomic_add(counters_ptr + 2, (now - t0))
        else:
            # the late copy stamps these slots itself
            pull = n
            tl.atomic_add(counters_ptr + 3, 1)
    elif state == 0:
        pull = n
        tl.atomic_add(counters_ptr + 4, 1)
        off = tl.arange(0, MAXB)
        m = off < n
        slots = tl.load(slots_ptr + off, mask=m, other=0)
        tl.store(ready_ptr + slots, tl.load(gen_ptr + slots, mask=m, other=0), mask=m)
    tl.store(fallback_ptr, pull)


__all__ = ["DemandCopy", "demand_copy_mode"]
