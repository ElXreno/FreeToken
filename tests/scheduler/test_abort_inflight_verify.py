"""Abort of a request whose verify step is in flight, draft accepted or not."""
from __future__ import annotations

from types import SimpleNamespace

import torch
from freetoken.core import Batch
from freetoken.message import AbortBackendMsg
from freetoken.scheduler.scheduler import Scheduler

from .test_abort_inflight_prefill import UID, _launch_req, _setup

DRAFT = 60
PROMPT = torch.arange(1, 13, dtype=torch.int32)


def _verify_stub():
    pool, cm, tm, dm, pm, sent, stub = _setup()
    stub.engine = SimpleNamespace(
        page_table=cm.page_table,
        linear_state_pool=pool,
        drafted_tokens=lambda n: [DRAFT] * n,
        stage_verify_draft=lambda *_: None,
    )
    stub.status_reporter.add_generated = lambda _n: None
    stub._verify_steps = 0
    stub._verify_hits = 0
    stub._commit_verify = lambda batch, toks: Scheduler._commit_verify(stub, batch, toks)
    return pool, cm, tm, dm, pm, sent, stub


def _armed_verify(pool, cm, tm, dm):
    """A decoding request with a verify step in flight, armed the way _arm_verify does it."""
    req = _launch_req(pool, cm, tm, PROMPT)
    req.append_host(torch.tensor([50], dtype=torch.int32))
    dm.filter_reqs([req])
    req.verify_slot = pool.alloc(1)[0]
    req.device_len += 1
    req.grow_host(1)
    cm.allocate_paged([req])
    batch = Batch(reqs=[req], phase="decode")
    batch.verify = True
    return req, batch


def _in_flight(batch, row0, row1):
    return (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([row0, row1], dtype=torch.int32), SimpleNamespace(synchronize=lambda: None)),
    )


def _abort_and_drain(row0):
    pool, cm, tm, dm, _pm, sent, stub = _verify_stub()
    req, batch = _armed_verify(pool, cm, tm, dm)
    live_slot, scratch_slot = req.linear_slot_idx, req.verify_slot
    stub._last_data = _in_flight(batch, row0, 77)
    Scheduler._process_one_msg(stub, AbortBackendMsg(uid=UID))
    assert req.aborted and req.table_idx != -1
    Scheduler._process_last_data(stub, stub._last_data)
    assert req.table_idx == -1
    assert sent == []
    return cm, pool, live_slot, scratch_slot


def test_abort_with_accepted_draft_returns_every_page():
    cm, _pool, _live, _scratch = _abort_and_drain(row0=DRAFT)
    cm.check_integrity()


def test_abort_with_accepted_draft_donates_state_matching_the_node():
    cm, _pool, live, scratch = _abort_and_drain(row0=DRAFT)
    tokens = torch.cat([PROMPT, torch.tensor([50], dtype=torch.int32)])
    m = cm.prefix_cache.match_prefix(tokens)
    assert m.cached_len == len(tokens)
    assert m.mamba_value == live, f"node of {len(tokens)} tokens carries slot {m.mamba_value}, scratch was {scratch}"


def test_abort_with_rejected_draft_returns_every_page():
    cm, _pool, _live, _scratch = _abort_and_drain(row0=DRAFT + 1)
    cm.check_integrity()


def test_accepted_draft_without_abort_still_commits_both_tokens():
    pool, cm, tm, dm, _pm, sent, stub = _verify_stub()
    req, batch = _armed_verify(pool, cm, tm, dm)
    Scheduler._process_last_data(stub, _in_flight(batch, DRAFT, 77))
    assert [m.next_token for m in sent] == [DRAFT, 77]
    assert req.cached_len == len(PROMPT) + 2
    assert req.input_ids.tolist()[-3:] == [50, DRAFT, 77]
    Scheduler._free_req_resources(stub, req)
    cm.check_integrity()
