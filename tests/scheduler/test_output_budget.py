"""A request asking for N tokens gets N, with overlap scheduling and with a verify pair."""
from __future__ import annotations

import torch
from freetoken.core import Batch, Req, SamplingParams
from freetoken.message import DetokenizeMsg
from freetoken.scheduler.scheduler import Scheduler

from .test_abort_inflight_prefill import UID, _as_last_data
from .test_abort_inflight_verify import DRAFT, _in_flight, _verify_stub

PROMPT = torch.arange(1, 13, dtype=torch.int32)


def _admit(pool, cm, tm, max_tokens):
    mr = cm.match_req(type("P", (), {"input_ids": PROMPT, "input_len": len(PROMPT)}))
    req = Req(input_ids=PROMPT, table_idx=tm.allocate(), cached_len=0, output_len=max_tokens,
              uid=UID, sampling_params=SamplingParams(max_tokens=max_tokens), cache_handle=mr.cuda_handle)
    req.linear_slot_idx = pool.alloc(1)[0]
    req.mamba_ping_pong = tuple(pool.alloc(2))
    req.mamba_next_track_idx = 1
    cm.lock(mr.cuda_handle)
    return req


def _launch(cm, dm, req):
    cm.allocate_paged([req])
    req.complete_one()
    dm.filter_reqs([req])


def _tokens(sent):
    return [m.next_token for m in sent if isinstance(m, DetokenizeMsg)]


def test_overlap_plain_steps_emit_every_requested_token():
    pool, cm, tm, dm, _pm, sent, stub = _verify_stub()
    req = _admit(pool, cm, tm, max_tokens=2)
    _launch(cm, dm, req)
    prefill = Batch(reqs=[req], phase="prefill")
    _launch(cm, dm, req)
    decode = Batch(reqs=[req], phase="decode")
    Scheduler._process_last_data(stub, _as_last_data(prefill))
    Scheduler._process_last_data(stub, _as_last_data(decode))
    assert len(_tokens(sent)) == 2
    assert req.table_idx == -1
    cm.check_integrity()


def test_accepted_pair_on_the_last_step_emits_both_tokens():
    pool, cm, tm, dm, _pm, sent, stub = _verify_stub()
    req = _admit(pool, cm, tm, max_tokens=3)
    _launch(cm, dm, req)
    Scheduler._process_last_data(stub, _as_last_data(Batch(reqs=[req], phase="prefill")))
    assert req.remain_len == 2
    req.verify_slot = pool.alloc(1)[0]
    req.device_len += 1
    req.grow_host(1)
    cm.allocate_paged([req])
    batch = Batch(reqs=[req], phase="decode")
    batch.verify = True
    Scheduler._process_last_data(stub, _in_flight(batch, DRAFT, 77))
    assert _tokens(sent)[-2:] == [DRAFT, 77]
    assert len(_tokens(sent)) == 3
    assert [m.finished for m in sent if isinstance(m, DetokenizeMsg)] == [False, False, True]
    cm.check_integrity()
