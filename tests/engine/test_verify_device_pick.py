"""A verify step's next draft staged from the device-side pick matches the host's commit decision."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from freetoken.engine.engine import Engine

HIDDEN = 16
WIDTH = 5


def _engine(rows_hidden: torch.Tensor):
    eng = SimpleNamespace(
        device=torch.device("cpu"),
        _mtp_slots_host=torch.zeros(WIDTH, dtype=torch.int64),
        _mtp_fresh_host=torch.zeros(WIDTH, dtype=torch.int64),
        _mtp_slots=torch.zeros(WIDTH, dtype=torch.int64),
        _mtp_tokens=torch.zeros(WIDTH, dtype=torch.int32),
        _mtp_positions=torch.zeros(WIDTH, dtype=torch.int32),
        _mtp_hidden=torch.zeros(WIDTH, HIDDEN),
        _verify_draft_dev=torch.zeros(WIDTH, dtype=torch.int32),
        _verify_row0=torch.arange(0, 2 * WIDTH, 2, dtype=torch.int64),
        _mtp_pending=0,
        _mtp_cpu_ns=0,
        mtp_head=SimpleNamespace(reset_slots=lambda _slots: None),
        model=SimpleNamespace(decode_hidden=lambda rows: rows_hidden[:rows]),
    )
    return eng


def _batch(n: int):
    reqs = [SimpleNamespace(table_idx=10 + i, mtp_drafted=3) for i in range(n)]
    positions = torch.arange(100, 100 + 2 * n, dtype=torch.int32)
    return SimpleNamespace(reqs=reqs, size=n, positions=positions)


def _staged(eng, n):
    return (eng._mtp_hidden[:n].clone(), eng._mtp_tokens[:n].clone(), eng._mtp_positions[:n].clone())


@pytest.mark.parametrize("n", [1, 2, 4])
def test_device_pick_stages_the_row_the_host_commits_from(n):
    gen = torch.Generator().manual_seed(n)
    for trial in range(8):
        hidden = torch.randn(2 * n, HIDDEN, generator=gen)
        drafts = torch.randint(0, 50, (n,), dtype=torch.int32, generator=gen)
        tokens = torch.randint(0, 50, (2 * n,), dtype=torch.int32, generator=gen)
        held = torch.rand(n, generator=gen) < 0.5
        tokens[0::2] = torch.where(held, drafts, (drafts + 1) % 50)
        # the host's rule in Scheduler._commit_verify: second row when row 0 reproduced the draft
        host_pick = [2 * i + int(int(tokens[2 * i]) == int(drafts[i])) for i in range(n)]

        ref = _engine(hidden)
        Engine._stage_mtp_draft(ref, _batch(n), tokens, 2 * n, pick=host_pick)

        dev = _engine(hidden)
        dev._verify_draft_dev[:n] = drafts
        pick = Engine._device_pick(dev, tokens, n)
        Engine._stage_mtp_draft(dev, _batch(n), tokens, 2 * n, pick=pick)

        assert pick.tolist() == host_pick, (trial, held.tolist())
        for a, b in zip(_staged(ref, n), _staged(dev, n)):
            assert torch.equal(a, b), (trial, held.tolist())
        assert dev._mtp_pending == n
