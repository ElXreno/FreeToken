from __future__ import annotations

import json

import pytest
import torch
from freetoken.core import SamplingParams
from freetoken.engine.grammar import GrammarManager

pytest.importorskip("xgrammar")

VOCAB = ["<unk>", "<eos>", "<think>", "</think>", "{", "}", '"', "a", ":", " ", "1", "2", ",", "b", "x"]
ID = {t: i for i, t in enumerate(VOCAB)}
SCHEMA = json.dumps({"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]})


@pytest.fixture
def manager() -> GrammarManager:
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel(ID, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", additional_special_tokens=["<think>", "</think>"]
    )
    return GrammarManager(tokenizer, len(VOCAB), [ID["<eos>"]])


class _Req:
    def __init__(self, uid: int) -> None:
        self.uid = uid


def _allowed(manager: GrammarManager, uid: int) -> set[str]:
    logits = torch.zeros(1, len(VOCAB))
    batch = manager.batch([_Req(uid)], type(None))
    if batch is not None:
        logits = batch.apply(logits)
    return {VOCAB[i] for i in torch.nonzero(logits[0] > float("-inf")).flatten().tolist()}


def _feed(manager: GrammarManager, uid: int, tokens: list[str]) -> None:
    for t in tokens:
        manager.accept(uid, ID[t])


def test_grammar_masks_follow_the_schema_until_eos(manager):
    assert manager.register(1, SamplingParams(json_schema=SCHEMA)) is None

    assert "{" in _allowed(manager, 1) and "a" not in _allowed(manager, 1)
    _feed(manager, 1, ["{", '"', "a", '"', ":"])
    assert {"1", "2"} <= _allowed(manager, 1) and "b" not in _allowed(manager, 1)
    _feed(manager, 1, ["1", "}"])
    assert "<eos>" in _allowed(manager, 1)

    manager.release(1)
    assert manager.batch([_Req(1)], type(None)) is None


def test_grammar_waits_for_the_reasoning_end_marker(manager):
    manager.register(2, SamplingParams(json_schema=SCHEMA, json_schema_after="</think>"))

    _feed(manager, 2, ["x", "b"])
    assert _allowed(manager, 2) == set(VOCAB)
    _feed(manager, 2, ["</think>"])
    assert "{" in _allowed(manager, 2) and "x" not in _allowed(manager, 2)


def test_grammar_off_token_leaves_the_request_unconstrained(manager):
    manager.register(3, SamplingParams(json_schema=SCHEMA))
    _feed(manager, 3, ["x"])

    assert manager.batch([_Req(3)], type(None)) is None


def test_grammar_rejects_a_schema_it_cannot_compile(manager):
    assert manager.register(4, SamplingParams(json_schema="{not json")) is not None
    assert 4 not in manager


def test_verify_rows_follow_the_drafts_and_leave_the_matcher_untouched(manager):
    manager.register(5, SamplingParams(json_schema=SCHEMA))
    _feed(manager, 5, ["{", '"', "a", '"'])
    batch = manager.batch([_Req(5)], type(None))

    logits = batch.apply_verify(torch.zeros(3, len(VOCAB)), torch.tensor([[ID[":"], ID["1"]]]))
    allowed = [{VOCAB[i] for i in torch.nonzero(row > float("-inf")).flatten().tolist()} for row in logits]
    assert ":" in allowed[0] and "1" not in allowed[0]
    assert "1" in allowed[1] and ":" not in allowed[1]
    assert "}" in allowed[2]

    # a draft the grammar rejects leaves the rows after it unmasked: they can never be committed
    logits = batch.apply_verify(torch.zeros(3, len(VOCAB)), torch.tensor([[ID["x"], ID["1"]]]))
    assert torch.isinf(logits[0]).any() and not torch.isinf(logits[1:]).any()

    assert ":" in _allowed(manager, 5) and "1" not in _allowed(manager, 5)


def test_verify_rows_after_a_drafted_stop_token_stay_free(manager):
    manager.register(9, SamplingParams(json_schema=SCHEMA))
    _feed(manager, 9, ["{", '"', "a", '"', ":", "1", "}"])
    batch = manager.batch([_Req(9)], type(None))

    logits = batch.apply_verify(torch.zeros(3, len(VOCAB)), torch.tensor([[ID["<eos>"], ID["x"]]]))

    assert not torch.isinf(logits[0, ID["<eos>"]]) and torch.isinf(logits[0, ID["x"]])
    assert not torch.isinf(logits[1:]).any()
    assert "<eos>" in _allowed(manager, 9)

    manager.accept(9, ID["<eos>"])
    assert _allowed(manager, 9) == set(VOCAB)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grammar_masks_cuda_logits_in_place_and_skips_free_rows(manager):
    manager.register(7, SamplingParams(json_schema=SCHEMA))
    logits = torch.zeros(2, len(VOCAB), device="cuda")

    out = manager.batch([_Req(7), _Req(8)], type(None)).apply(logits)

    assert out.data_ptr() == logits.data_ptr()
    assert torch.isinf(logits[0, ID["a"]]) and not torch.isinf(logits[0, ID["{"]])
    assert not torch.isinf(logits[1]).any()


def test_verify_rows_switch_on_at_a_drafted_reasoning_end(manager):
    manager.register(6, SamplingParams(json_schema=SCHEMA, json_schema_after="</think>"))
    batch = manager.batch([_Req(6)], type(None))

    logits = batch.apply_verify(torch.zeros(2, len(VOCAB)), torch.tensor([[ID["</think>"]]]))

    assert not torch.isinf(logits[0]).any()
    assert torch.isinf(logits[1, ID["x"]]) and not torch.isinf(logits[1, ID["{"]])
