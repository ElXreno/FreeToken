"""Confidence-gated verify: a request whose draft confidence is below FREETOKEN_SPEC_GATE rejects its draft even when the
target agrees and routes its draft row to its pending row's experts during the forward; the others verify as usual."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 8


class _Proposer:
    def __init__(self, conf):
        self.conf = torch.tensor(conf, device="cuda")
        self.verified = None

    def track_confidence(self):
        return True

    def confidence(self, rows):
        return self.conf.index_select(0, rows)

    def feature_sink(self, batch):
        return None

    def post_verify(self, batch, out, emitted):
        self.verified = emitted.tolist()


def _runner(monkeypatch, conf, gate=0.5):
    from freetoken.engine.sample import Sampler
    from freetoken.spec import runner as R

    monkeypatch.setattr(R, "_GATE", gate)
    cache = SimpleNamespace(route_src=None)
    monkeypatch.setattr(R, "iter_ops", lambda model: [SimpleNamespace(offload_cache=cache)])
    seen = []
    logits = torch.full((4, VOCAB), -10.0, device="cuda")
    for row, t in enumerate([3, 5, 4, 6]):
        logits[row, t] = 10.0

    def forward_logits(batch):
        seen.append(None if cache.route_src is None else cache.route_src[:4].tolist())
        return logits

    engine = SimpleNamespace(model=None, forward_logits=forward_logits, stream=torch.cuda.current_stream(),
                             sampler=Sampler(device=torch.device("cuda"), vocab_size=VOCAB))
    runner = R.SpecRunner.__new__(R.SpecRunner)
    runner.config = SimpleNamespace(max_running_req=2)
    runner.device = torch.device("cuda")
    runner.k = 1
    runner.proposer = _Proposer(conf)
    runner._route_src = None
    runner.engine = engine
    runner._host = torch.zeros(3, 3, dtype=torch.int64, pin_memory=True)
    runner._attach_gate(engine)
    return runner, seen


def _verify(runner):
    from freetoken.engine.sample import BatchSamplingArgs

    # each draft matches its request's first target row, so plain verify accepts both
    batch = SimpleNamespace(size=2, input_ids=torch.tensor([9, 3, 9, 4], device="cuda"),
                            reqs=[SimpleNamespace(table_idx=0), SimpleNamespace(table_idx=1)])
    out = runner.verify_step(batch, BatchSamplingArgs(temperatures=None))
    out.copy_done_event.synchronize()
    return out


@cuda
def test_a_low_confidence_draft_is_rejected_and_its_row_routes_to_the_pending_row(monkeypatch):
    runner, seen = _runner(monkeypatch, [0.2, 0.9])
    _verify(runner)
    assert runner.proposer.verified == [0, 1]
    assert seen == [[0, 0, 2, 3]]


@cuda
def test_route_sources_reset_once_the_drafts_are_confident(monkeypatch):
    runner, seen = _runner(monkeypatch, [0.2, 0.9])
    _verify(runner)
    runner.proposer.conf.fill_(0.9)
    _verify(runner)
    assert runner.proposer.verified == [1, 1]
    assert seen[-1] == [0, 1, 2, 3]


@cuda
def test_no_gate_leaves_routing_and_acceptance_alone(monkeypatch):
    runner, seen = _runner(monkeypatch, [0.2, 0.9], gate=0.0)
    _verify(runner)
    assert runner.proposer.verified == [1, 1]
    assert seen == [None]
