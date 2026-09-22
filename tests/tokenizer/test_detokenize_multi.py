"""A speculative step commits two tokens for one request, so one batch carries two messages.

``detokenize`` reads every message's offsets before any of them move, so a second message for
the same uid re-emits the first one's text: the pair ``[" is", " asking"]`` comes out as
`` is is asking``. The signature matters -- it is a doubled leading token, not merely different
text, and a hash comparison between two speculative runs does NOT catch it, because both runs
double the same way.
"""

from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager

VOCAB = {1: " is", 2: " asking", 3: " about", 4: "g", 5: "ated", 9: "<eos>"}
EOS = 9


class FakeTokenizer:
    eos_token_id = EOS

    def batch_decode(self, seqs):
        return ["".join(VOCAB[i] for i in seq) for seq in seqs]


def manager() -> DetokenizeManager:
    return DetokenizeManager(FakeTokenizer(), eos_token_ids=frozenset({EOS}))


def msg(uid: int, token: int) -> DetokenizeMsg:
    return DetokenizeMsg(
        uid=uid, next_token=token, finished=False, finish_reason=None,
        matched_stop=None, stop_strs=None,
    )


def test_one_batch_equals_one_message_at_a_time():
    """Same uid, N messages in one batch == N batches of one. No reference decode needed."""
    for tokens in ([1, 2], [4, 5], [1, 2, 3]):
        batched = "".join(manager().detokenize([msg(1, t) for t in tokens]))
        mgr = manager()
        streamed = "".join("".join(mgr.detokenize([msg(1, t)])) for t in tokens)
        want = "".join(VOCAB[t] for t in tokens)
        assert batched == streamed == want, f"{tokens}: {batched!r} vs {streamed!r} vs {want!r}"


def test_batched_round_still_doubles():
    """Guards the test above: without per-message rounds the pair must still come out wrong.

    If _decode_round ever stopped doubling, the invariant test would pass for free and prove
    nothing about detokenize's round splitting."""
    doubled = "".join(manager()._decode_round([msg(1, 1), msg(1, 2)]))
    assert doubled == " is is asking"


def test_interleaved_uids_keep_message_order():
    """A verify batch mixes requests: two rows for uid 1, one for uid 2, order preserved."""
    msgs = [msg(1, 1), msg(2, 3), msg(1, 2)]
    out = manager().detokenize(msgs)
    assert out == [" is", " about", " asking"]


def test_finished_eos_is_not_emitted_twice():
    """An accepted pair whose second row is EOS: the eos token itself never reaches the text."""
    pair = [msg(1, 1), DetokenizeMsg(
        uid=1, next_token=EOS, finished=True, finish_reason="stop",
        matched_stop=None, stop_strs=None,
    )]
    assert "".join(manager().detokenize(pair)) == " is"
