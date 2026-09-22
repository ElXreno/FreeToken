"""Debug probe (FREETOKEN_PAGE_PROBE=1): whenever no forward is in flight, every KV page has
exactly one owner -- the free list, a resident tree node, the arena-eviction backlog, or the
span a live request allocated past its prefix handle. A breach is logged once per page, with
the history of the requests that last held it."""
from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import torch

from freetoken.env import ENV
from freetoken.message import AbortBackendMsg
from freetoken.utils import init_logger

from .prefill import ChunkedReq

if TYPE_CHECKING:
    from freetoken.core import Req

    from .scheduler import Scheduler

logger = init_logger(__name__)

FREE, TREE, BACKLOG, NOBODY = -1, -2, -3, -9
_LABELS = {FREE: "free", TREE: "tree", BACKLOG: "backlog", NOBODY: "never"}


class PageProbe:
    def __init__(self, sched: Scheduler) -> None:
        self.s = sched
        self.events: deque = deque(maxlen=50_000)
        cm = sched.cache_manager
        self.alloc_owner = torch.full((cm.num_pages,), NOBODY, dtype=torch.int64, device=cm.device)
        self.prev_owner: torch.Tensor | None = None
        self.reported: set[int] = set()
        self.checks = 0
        self.breaches = 0
        self._wrap()
        logger.info_rank0("page probe armed")

    def log(self, kind: str, req: Req | None = None, **kw) -> None:
        if req is not None:
            handle = req.cache_handle
            kw.update(
                uid=req.uid, cached=req.cached_len, device=req.device_len,
                handle=None if handle is None else handle.cached_len,
                chunked=isinstance(req, ChunkedReq), table=req.table_idx,
            )
        self.events.append((self.s._forward_iter, kind, kw))

    def _wrap(self) -> None:
        s, cm, probe = self.s, self.s.cache_manager, self

        orig_msg = s._process_one_msg

        def process_one_msg(msg):
            if isinstance(msg, AbortBackendMsg):
                req = next((r for r in s.decode_manager.running_reqs if r.uid == msg.uid), None)
                if req is None:
                    p = next((p for p in s.prefill_manager.pending_list if p.uid == msg.uid), None)
                    req = None if p is None else p.chunked_req
                last = getattr(s, "_last_data", None)
                inflight = req is not None and last is not None and req in last[0].batch.reqs
                probe.log("abort", req, msg_uid=msg.uid, inflight=inflight)
            return orig_msg(msg)

        s._process_one_msg = process_one_msg

        orig_free = s._free_req_resources

        def free_req_resources(req):
            probe.log("free", req, aborted=req.aborted)
            return orig_free(req)

        s._free_req_resources = free_req_resources

        orig_commit = s._commit_verify

        def commit_verify(batch, next_tokens_cpu):
            before = [(r.cached_len, r.device_len) for r in batch.reqs]
            out = orig_commit(batch, next_tokens_cpu)
            for b, r in zip(before, batch.reqs):
                probe.log("commit", r, before=b)
            return out

        s._commit_verify = commit_verify

        orig_alloc = cm.allocate_paged

        def allocate_paged(reqs):
            spans = [(r, r.cached_len, r.device_len) for r in reqs]
            orig_alloc(reqs)
            for r, a, b in spans:
                probe.log("alloc", r, span=(a, b))
                if b > a:
                    pages = cm.page_table[r.table_idx, a:b].to(torch.int64) // cm.page_size
                    probe.alloc_owner[pages] = r.uid

        cm.allocate_paged = allocate_paged

        orig_drain = s._process_last_data

        def process_last_data(last_data):
            orig_drain(last_data)
            if last_data is not None and s._verify_enabled and not ENV.VERIFY_NOSYNC:
                probe.check("drain")

        s._process_last_data = process_last_data

        orig_idle = s.run_when_idle

        def run_when_idle():
            probe.check("idle")
            return orig_idle()

        s.run_when_idle = run_when_idle

    def _owners(self) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.s
        cm = s.cache_manager
        pc = cm.prefix_cache
        idx: list[torch.Tensor] = []
        lab: list[torch.Tensor] = []

        def add(tokens: torch.Tensor, label: int) -> None:
            if tokens.numel():
                pages = tokens.to(cm.device, torch.int64) // cm.page_size
                if cm.page_size > 1:
                    pages = torch.unique(pages)
                idx.append(pages)
                lab.append(torch.full_like(pages, label))

        add(cm.free_slots, FREE)
        for n in pc._all_nodes():
            if getattr(n, "resident", True):
                add(n.value, TREE)
        for t in getattr(pc, "freed_kv", []):
            add(t, BACKLOG)
        live = set(s.decode_manager.running_reqs)
        live.update(p.chunked_req for p in s.prefill_manager.pending_list if p.chunked_req is not None)
        for r in live:
            if r.table_idx != -1:
                add(cm.page_table[r.table_idx, r.cache_handle.cached_len : r.cached_len], r.uid)
        return torch.cat(idx), torch.cat(lab)

    def check(self, where: str) -> None:
        n = self.s.cache_manager.num_pages
        pages, labels = self._owners()
        count = torch.bincount(pages, minlength=n)
        owner = torch.full((n,), NOBODY, dtype=torch.int64, device=pages.device)
        owner[pages] = labels
        bad = ((count == 0) | (count > 1)).nonzero().flatten().tolist()
        self.checks += 1
        fresh = [p for p in bad if p not in self.reported]
        if fresh:
            self.breaches += 1
            self.reported.update(fresh)
            self._report(where, fresh, count, owner)
        elif self.checks % 2000 == 0:
            logger.info_rank0(f"page probe: {self.checks} checks, {len(self.reported)} bad pages so far")
        self.prev_owner = owner

    def _report(self, where: str, pages: list[int], count: torch.Tensor, owner: torch.Tensor) -> None:
        prev = self.prev_owner
        lines = [f"PAGE PROBE BREACH #{self.breaches} at {where} iter {self.s._forward_iter}: {len(pages)} pages"]
        uids: set[int] = set()
        for p in pages[:16]:
            c = int(count[p])
            before = NOBODY if prev is None else int(prev[p])
            alloc = int(self.alloc_owner[p])
            lines.append(
                f"  page {p}: owners now {c}, before {_LABELS.get(before, f'req {before}')}, "
                f"last alloc {_LABELS.get(alloc, f'req {alloc}')}"
            )
            uids.update(u for u in (before, alloc) if u >= 0)
        for uid in sorted(uids):
            hist = [e for e in self.events if e[2].get("uid") == uid or e[2].get("msg_uid") == uid]
            lines.append(f"  history of uid {uid} ({len(hist)} events, last 40):")
            lines.extend(f"    it{it} {kind} {kw}" for it, kind, kw in hist[-40:])
        tail = list(self.events)[-12:]
        lines.append("  last events:")
        lines.extend(f"    it{it} {kind} {kw}" for it, kind, kw in tail)
        logger.warning_rank0("\n".join(lines))
