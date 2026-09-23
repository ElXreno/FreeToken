"""Hybrid radix cache over a host tier: the arena is the source of truth, VRAM pages are a
cache over it, and GDN snapshots live only in the arena (restored into the hitting request's
own live slot). Eviction from VRAM drops the resident copy; eviction from the arena drops
the node.
"""
from __future__ import annotations

import heapq
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from freetoken.utils import align_down

from .base import BaseCacheHandle
from .host_tier import HostRef, HostTier, decode_tokens, encode_tokens
from .hybrid_radix_cache import EvictResult, HybridRadixCache
from .radix_cache import RadixTreeNode


class TieredMatch(NamedTuple):
    cached_len: int              # truncated to the deepest node whose end boundary has a snapshot
    snap: HostRef | None         # arena snapshot to restore into the request's live slot
    node: RadixTreeNode
    promote_tokens: int          # tokens on the path that are host-only and need VRAM pages


@dataclass(frozen=True)
class TieredCacheHandle(BaseCacheHandle):
    node: RadixTreeNode
    promote_tokens: int = 0

    def get_matched_indices(self) -> torch.Tensor:
        vals: list[torch.Tensor] = []
        n = self.node
        while not n.is_root():
            assert n.resident, "matched path used before promotion"
            vals.append(n.value)
            n = n.parent
        vals.reverse()
        return torch.cat(vals) if vals else self.node.value[:0]


class TieredHybridRadixCache(HybridRadixCache):
    def __init__(self, device: torch.device, page_size: int, tier: HostTier) -> None:
        super().__init__(device, page_size)
        self.tier = tier
        tier.evict = self.evict_host
        self.freed_kv: list[torch.Tensor] = []   # VRAM pages released by arena eviction, drained by the CacheManager
        self.host_evictions = 0

    # ---------------------------------------------------------------- match / insert
    def match_prefix(self, input_ids: torch.Tensor) -> TieredMatch:
        node, _ = self._walk(input_ids)
        cur, end_len = node, self._path_len(node)
        while not cur.is_root():
            if cur.host_snap is not None:
                promote = sum(n.length for n in self._path(cur) if not n.resident)
                return TieredMatch(end_len, cur.host_snap, cur, promote)
            end_len -= cur.length
            cur = cur.parent
        return TieredMatch(0, None, self.root, 0)

    def insert(self, input_ids: torch.Tensor, kv_indices: torch.Tensor) -> tuple[int, RadixTreeNode, list[tuple[int, int]]]:
        """Commit ``kv_indices`` for ``input_ids``. Existing resident nodes on the path are
        reported back as duplicate ranges (the caller frees the request's pages there);
        host-only nodes adopt the request's pages instead. Returns the end boundary node."""
        insert_len = align_down(len(input_ids), self.page_size)
        ids, kv = input_ids[:insert_len], kv_indices[:insert_len]
        prefix_len, node, dups = 0, self.root, []
        tic = time.monotonic_ns()
        while prefix_len < insert_len:
            child = node.children.get(self.key_fn(ids[prefix_len:]))
            if child is None:
                break
            match_len = align_down(child.get_match_len(ids[prefix_len:]), self.page_size)
            if match_len == 0:
                break
            partial = match_len != child.length
            if partial:
                child = child.split_at(match_len)
            child.timestamp = tic
            if child.resident:
                dups.append((prefix_len, prefix_len + child.length))
            else:
                self._adopt(child, kv[prefix_len : prefix_len + child.length])
            prefix_len += child.length
            node = child
            if partial:
                break
        if prefix_len != insert_len:
            new_node = RadixTreeNode(self.key_fn, tic)
            new_node.set_key_value(ids[prefix_len:], kv[prefix_len:].clone())
            new_node.set_parent(node)
            self.full_evictable += new_node.length
            node = new_node
        return prefix_len, node, dups

    def _adopt(self, node: RadixTreeNode, kv_slice: torch.Tensor) -> None:
        node._value = kv_slice.clone()
        node.resident = True
        if node.ref_count == 0:
            self.full_evictable += node.length
        else:
            self.full_protected += node.length

    # ---------------------------------------------------------------- locking
    def inc_lock(self, node: RadixTreeNode) -> None:
        cur = node
        while not cur.is_root():
            if cur.ref_count == 0 and cur.resident:
                self.full_evictable -= cur.length
                self.full_protected += cur.length
            cur.ref_count += 1
            cur = cur.parent

    def dec_lock(self, node: RadixTreeNode) -> None:
        cur = node
        while not cur.is_root():
            cur.ref_count -= 1
            assert cur.ref_count >= 0
            if cur.ref_count == 0 and cur.resident:
                self.full_evictable += cur.length
                self.full_protected -= cur.length
            cur = cur.parent

    # ---------------------------------------------------------------- promotion
    def promote(self, node: RadixTreeNode, alloc_tokens: Callable[[int], torch.Tensor]) -> int:
        """Bring every host-only node on root..node back into VRAM. Call with the path locked so
        the page allocation underneath cannot evict it."""
        todo = [n for n in self._path(node) if not n.resident]
        if not todo:
            return 0
        total = sum(n.length for n in todo)
        slots = alloc_tokens(total)
        off = 0
        for n in todo:
            piece = slots[off : off + n.length]
            off += n.length
            self.tier.get_kv(n.host_kv, piece)
            n._value = piece.clone()
            n.resident = True
            if n.ref_count == 0:
                self.full_evictable += n.length
            else:
                self.full_protected += n.length
        return total

    # ---------------------------------------------------------------- eviction
    def evict_full(self, num_tokens: int) -> EvictResult:
        """Release VRAM pages: LRU over unlocked resident nodes. A node with a host copy just
        loses residency; one without drops with its whole subtree."""
        cands = [n for n in self._all_nodes() if n.ref_count == 0 and n.resident]
        heapq.heapify(cands)
        kv: list[torch.Tensor] = []
        freed = 0
        while freed < num_tokens and cands:
            node = heapq.heappop(cands)
            if node.ref_count != 0 or not node.resident or node.is_root() or node._parent is None:
                continue
            if node.host_kv is not None:
                kv.append(node.value)
                freed += node.length
                self.full_evictable -= node.length
                node._value = node._key[:0].clone()
                node.resident = False
            else:
                freed += self._drop_subtree(node, kv)
        return EvictResult(torch.cat(kv) if kv else self.empty, [])

    def evict_mamba(self, num: int) -> EvictResult:
        return EvictResult(self.empty, [])

    def evict_host(self, nbytes: int) -> bool:
        """Arena pressure: drop the LRU unlocked leaf that owns host data (a leaf that cannot
        resume its GDN state is worthless), cascading up over snapshot-less leaves."""
        leaves = [n for n in self._leaves() if n.ref_count == 0 and (n.host_kv is not None or n.host_snap is not None)]
        if not leaves:
            return False
        node = min(leaves)
        parent = node.parent
        self._drop_subtree(node, self.freed_kv)
        while (not parent.is_root() and parent.is_leaf() and parent.ref_count == 0
               and parent.host_snap is None):
            nxt = parent.parent
            self._drop_subtree(parent, self.freed_kv)
            parent = nxt
        self.host_evictions += 1
        return True

    def take_freed_kv(self) -> torch.Tensor:
        if not self.freed_kv:
            return self.empty
        out = torch.cat(self.freed_kv)
        self.freed_kv = []
        return out

    def _drop_subtree(self, node: RadixTreeNode, kv_out: list[torch.Tensor]) -> int:
        freed, stack = 0, [node]
        while stack:
            n = stack.pop()
            stack.extend(n.children.values())
            if n.resident:
                kv_out.append(n.value)
                self.full_evictable -= n.length
                freed += n.length
            self.tier.free(n.host_kv)
            self.tier.free(n.host_snap)
            n.host_kv = n.host_snap = None
            n.children = {}
            n.resident = False
        self._unlink(node)
        node._parent = None
        return freed

    # ---------------------------------------------------------------- accounting / checks
    @property
    def mamba_evictable_size(self) -> int:
        return 0

    def check_integrity(self) -> None:
        for n in self._all_nodes():
            if n.resident:
                assert len(n.value) == n.length, "resident node with a short page span"
            else:
                assert n.host_kv is not None, "host-only node without an arena span"
            assert n.ref_count >= 0

    def host_stats(self) -> dict[str, Any]:
        nodes = self._all_nodes()
        return {
            **self.tier.stats(),
            "nodes": len(nodes),
            "resident_tokens": sum(n.length for n in nodes if n.resident),
            "host_tokens": sum(n.length for n in nodes if n.host_kv is not None),
            "snapshots": sum(1 for n in nodes if n.host_snap is not None),
            "host_evictions": self.host_evictions,
        }

    # ---------------------------------------------------------------- persistence
    def export_nodes(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        ids: dict[int, int] = {id(self.root): -1}
        queue = [c for c in self.root.children.values()]
        while queue:
            n = queue.pop(0)
            if n.host_kv is None:
                continue
            ids[id(n)] = len(out)
            out.append({
                "parent": ids[id(n.parent)],
                "tokens": encode_tokens(n._key),
                "kv": [n.host_kv.offset, n.host_kv.nbytes, n.host_kv.released()],
                "snap": [n.host_snap.offset, n.host_snap.nbytes] if n.host_snap is not None else None,
                "ts": n.timestamp,
            })
            queue.extend(n.children.values())
        return out

    def import_nodes(self, nodes: list[dict[str, Any]]) -> int:
        """Rebuild the tree from a flush, every node host-only. Snapshot-less leaves are pruned
        (they cannot resume) and their spans returned to the arena."""
        built: list[RadixTreeNode] = []
        for rec in nodes:
            parent = self.root if rec["parent"] < 0 else built[rec["parent"]]
            node = RadixTreeNode(self.key_fn, int(rec["ts"]))
            node.set_key_host(decode_tokens(rec["tokens"]))
            node.host_kv = HostRef(*rec["kv"])
            node.host_snap = HostRef(*rec["snap"]) if rec["snap"] is not None else None
            node.set_parent(parent)
            built.append(node)
        pruned = True
        while pruned:
            pruned = False
            for n in self._leaves():
                if n.host_snap is None:
                    self._drop_subtree(n, self.freed_kv)
                    pruned = True
        return len(self._all_nodes())

    # ---------------------------------------------------------------- helpers
    def _path(self, node: RadixTreeNode) -> list[RadixTreeNode]:
        out: list[RadixTreeNode] = []
        n = node
        while not n.is_root():
            out.append(n)
            n = n.parent
        out.reverse()
        return out

    def _all_nodes(self) -> list[RadixTreeNode]:
        out, stack = [], list(self.root.children.values())
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children.values())
        return out


__all__ = ["TieredCacheHandle", "TieredHybridRadixCache", "TieredMatch"]
