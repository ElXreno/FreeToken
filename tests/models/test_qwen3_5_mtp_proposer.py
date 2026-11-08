"""The MTP proposer drafts from the target's last committed row: its hidden, the token sampled there, its position."""

from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace

import torch
from freetoken.models.qwen3_5_moe import mtp_proposer as mp
from freetoken.scheduler.prefill import ChunkedReq

H, V, ROWS = 4, 16, 3


class _Head:
    """Draft = (token + position + step + slot) mod V, output hidden = hidden + 1; records every call."""

    def __init__(self):
        self.calls = []
        self.steps = []
        self._conf = torch.ones(ROWS + 1)
        self.reset = []
        self.ring_bytes = 0

    def alloc_rings(self, num_slots, device, dtype, steps=1):
        self._pred = torch.zeros(steps, num_slots, dtype=torch.int32)

    def reset_slots(self, slots):
        self.reset.append(slots.tolist())

    def forward(self, hidden, tokens, positions, slots, embed_tokens, lm_head, step=0):
        self.calls.append((hidden.clone(), tokens.clone(), positions.clone(), slots.clone()))
        self.steps.append(step)
        pred = (tokens.long() + positions.long() + step + slots) % V
        return torch.nn.functional.one_hot(pred, V).float(), hidden + 1

    def remember(self, slots, pred, step=0):
        self._pred[step].index_copy_(0, slots, pred.to(self._pred.dtype))

    def drafts(self, slots):
        return self._pred.index_select(1, slots).t()

    def remember_confidence(self, slots, conf):
        self._conf.index_copy_(0, slots, conf)

    def confidence(self, slots):
        return self._conf.index_select(0, slots)


def _proposer(monkeypatch, k=1):
    monkeypatch.setattr(mp, "make_device_tensor", lambda data, dtype, device: torch.tensor(data, dtype=dtype))
    monkeypatch.setattr(mp, "get_global_ctx", lambda: SimpleNamespace(forward_batch=lambda batch: contextlib.nullcontext()))
    config = SimpleNamespace(
        speculative_num_steps=k, max_running_req=ROWS, dtype=torch.float32,
        model_config=SimpleNamespace(hidden_size=H, mtp_window=8),
    )
    head = _Head()
    model = SimpleNamespace(mtp=head, model=SimpleNamespace(embed_tokens=None), lm_head=None)
    p = mp.Qwen3_5MTPProposer(config, model)
    p.attach(SimpleNamespace(device=torch.device("cpu"), dummy_req=SimpleNamespace(table_idx=ROWS)))
    return p, head


def _req(table_idx, chunked=False):
    req = ChunkedReq.__new__(ChunkedReq) if chunked else SimpleNamespace()
    req.table_idx = table_idx
    return req


def test_post_verify_drafts_from_the_last_accepted_row(monkeypatch):
    p, head = _proposer(monkeypatch)
    reqs = [_req(2), _req(0)]
    batch = SimpleNamespace(size=2, reqs=reqs, positions=torch.tensor([10, 11, 20, 21], dtype=torch.int32))
    feats = torch.arange(4 * H, dtype=torch.float32).view(4, H)
    p.feature_sink(SimpleNamespace(is_prefill=False)).final_hidden(feats)
    out = torch.tensor([[5, 6], [7, 8]])
    emitted = torch.tensor([1, 0])
    p.post_verify(batch, out, emitted)
    hidden, tokens, positions, slots = head.calls[-1]
    assert torch.equal(hidden, feats[[1, 2]])
    assert tokens.tolist() == [6, 7]
    assert positions.tolist() == [11, 20]
    assert slots.tolist() == [2, 0]
    drafts = p.propose(reqs, torch.tensor([2, 0]), torch.tensor([6, 7])).draft_token_ids
    assert drafts.shape == (2, 1) and drafts.dtype == torch.int64
    assert drafts.view(-1).tolist() == [(6 + 11 + 2) % V, (7 + 20 + 0) % V]


def test_prefill_drafts_from_each_finished_prompt_and_resets_its_ring(monkeypatch):
    p, head = _proposer(monkeypatch)
    reqs = [_req(1), _req(2, chunked=True), _req(0)]
    last = torch.tensor([4, 9, 12])
    batch = SimpleNamespace(
        size=3, reqs=reqs, is_prefill=True,
        positions=torch.arange(13, dtype=torch.int32) + 100,
        attn_metadata=SimpleNamespace(get_last_indices=lambda bs: last[:bs]),
    )
    feats = torch.arange(13 * H, dtype=torch.float32).view(13, H)
    p.feature_sink(batch).final_hidden(feats)
    p.on_prefill(batch, torch.tensor([30, 31, 32], dtype=torch.int32))
    hidden, tokens, positions, slots = head.calls[-1]
    assert head.reset == [[1, 0]]
    assert torch.equal(hidden, feats[[4, 12]])
    assert tokens.tolist() == [30, 32]
    assert positions.tolist() == [104, 112]
    assert slots.tolist() == [1, 0]


def test_a_prefill_of_intermediate_chunks_drafts_nothing(monkeypatch):
    p, head = _proposer(monkeypatch)
    batch = SimpleNamespace(size=1, reqs=[_req(1, chunked=True)], is_prefill=True)
    p.on_prefill(batch, torch.tensor([3], dtype=torch.int32))
    assert head.calls == [] and head.reset == []


def test_a_tracked_draft_keeps_its_top_probability_per_slot(monkeypatch):
    p, head = _proposer(monkeypatch)
    assert p.track_confidence()
    batch = SimpleNamespace(size=1, reqs=[_req(2)], positions=torch.tensor([10, 11], dtype=torch.int32))
    p.feature_sink(SimpleNamespace(is_prefill=False)).final_hidden(torch.zeros(2, H))
    p.post_verify(batch, torch.tensor([[5, 6]]), torch.tensor([0]))
    # the stub's logits are one-hot, so the top probability is e / (e + V - 1)
    top = math.e / (math.e + V - 1)
    assert torch.allclose(p.confidence(torch.tensor([2, 0])), torch.tensor([top, 1.0]))


def test_a_chain_drafts_each_step_from_the_previous_steps_output_and_draft(monkeypatch):
    k = 3
    p, head = _proposer(monkeypatch, k=k)
    reqs = [_req(2), _req(0)]
    positions = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23], dtype=torch.int32)
    batch = SimpleNamespace(size=2, reqs=reqs, positions=positions)
    feats = torch.arange(8 * H, dtype=torch.float32).view(8, H)
    p.feature_sink(SimpleNamespace(is_prefill=False)).final_hidden(feats)
    out = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]])
    p.post_verify(batch, out, torch.tensor([2, 0]))
    assert head.steps == list(range(k))
    hidden, tokens, pos, slots = head.calls[0]
    assert torch.equal(hidden, feats[[2, 4]]) and tokens.tolist() == [7, 9] and pos.tolist() == [12, 20]
    chain, token = [], torch.tensor([7, 9])
    for step in range(k):
        hidden, tokens, pos, _ = head.calls[step]
        assert tokens.tolist() == token.tolist()
        # the head shifts the positions of a chained step itself
        assert pos.tolist() == [12, 20]
        if step:
            assert torch.equal(hidden, head.calls[step - 1][0] + 1)
        token = (token + torch.tensor([12, 20]) + step + torch.tensor([2, 0])) % V
        chain.append(token.tolist())
    drafts = p.propose(reqs, torch.tensor([2, 0]), torch.tensor([7, 9])).draft_token_ids
    assert drafts.shape == (2, k) and drafts.dtype == torch.int64
    assert drafts.tolist() == [list(t) for t in zip(*chain)]


def test_chained_steps_attend_to_their_own_keys_and_leave_the_ring_to_confirmed_tokens():
    """With a zero query every live key weighs the same, so each output is the mean of the values the step may read."""
    from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MTPHead

    head = Qwen3_5MTPHead.__new__(Qwen3_5MTPHead)
    attn = SimpleNamespace(num_q=2, num_kv=1, head_dim=4, qo_attn_dim=8, kv_attn_dim=4)
    head.layers = SimpleNamespace(op_list=[SimpleNamespace(self_attn=attn)])
    head._window = 4
    head.alloc_rings(3, torch.device("cpu"), torch.float32, steps=3)
    slots = torch.tensor([2])
    q, k = torch.zeros(1, 8), torch.zeros(1, 4)

    def attend(value, step):
        return head._attend(q, k, torch.full((1, 4), float(value)), slots, step)[0, 0].item()

    assert attend(1, 0) == 1
    assert attend(10, 1) == 5.5
    assert attend(100, 2) == 37
    # the next draft reads the ring alone: the chain's keys are gone, its first step's stays
    assert attend(2, 0) == 1.5
    assert head._written.tolist() == [0, 0, 2] and head._fill.tolist() == [0, 0, 2]
    assert head._v_ring[2, :, 0].tolist() == [1, 2, 0, 0, 10, 100]
