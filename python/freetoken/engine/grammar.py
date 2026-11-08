"""JSON-schema constrained decoding (OpenAI response_format) with xgrammar matchers kept per request on the scheduler."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from freetoken.utils import init_logger

if TYPE_CHECKING:
    from freetoken.core import Req, SamplingParams

logger = init_logger(__name__)

# a capped run keeps a model from looping on newlines inside the JSON
MAX_WHITESPACE = 32


@dataclass(eq=False)
class GrammarState:
    matcher: Any
    # token that switches the grammar on (the reasoning end marker); None once active
    activate_id: int | None = None
    broken: bool = False

    @property
    def active(self) -> bool:
        return self.activate_id is None and not self.broken


class GrammarManager:
    def __init__(self, tokenizer, vocab_size: int, stop_token_ids: Iterable[int]) -> None:
        self._tokenizer = tokenizer
        self._vocab_size = vocab_size
        self._stop_token_ids = sorted(stop_token_ids)
        self._compiler = None
        self._states: dict[int, GrammarState] = {}

    def _get_compiler(self):
        if self._compiler is None:
            import xgrammar as xgr

            info = xgr.TokenizerInfo.from_huggingface(
                self._tokenizer, vocab_size=self._vocab_size, stop_token_ids=self._stop_token_ids
            )
            self._compiler = xgr.GrammarCompiler(info)
        return self._compiler

    def register(self, uid: int, params: SamplingParams) -> str | None:
        """Compile the request's schema and park its matcher; returns an error message for a schema that cannot be served."""
        if params.json_schema is None:
            return None
        try:
            import xgrammar as xgr

            compiled = self._get_compiler().compile_json_schema(
                params.json_schema, max_whitespace_cnt=MAX_WHITESPACE
            )
        except Exception as e:  # noqa: BLE001 -- any compile failure is the request's schema
            return f"response_format schema cannot be compiled: {e}"
        activate_id = None
        if params.json_schema_after:
            activate_id = self._single_token_id(params.json_schema_after)
            if activate_id is None:
                logger.warning_rank0(
                    f"{params.json_schema_after!r} is not one token; request {uid} is constrained from its first token"
                )
        self._states[uid] = GrammarState(xgr.GrammarMatcher(compiled), activate_id)
        return None

    def _single_token_id(self, text: str) -> int | None:
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return int(ids[0]) if len(ids) == 1 else None

    def __contains__(self, uid: int) -> bool:
        return uid in self._states

    def __len__(self) -> int:
        return len(self._states)

    def has_any(self, reqs: Iterable[Req]) -> bool:
        return any(r.uid in self._states for r in reqs)

    def accept(self, uid: int, token: int) -> None:
        state = self._states.get(uid)
        if state is None or state.broken:
            return
        if state.activate_id is not None:
            if token == state.activate_id:
                state.activate_id = None
            return
        try:
            accepted = state.matcher.accept_token(token)
        except Exception as e:  # noqa: BLE001 -- a matcher fault frees its request, never the engine
            _break(state, e)
            return
        if not accepted:
            state.broken = True
            logger.warning_rank0(f"request {uid}: token {token} breaks its response_format grammar; decoding unconstrained")

    def release(self, uid: int) -> None:
        self._states.pop(uid, None)

    def batch(self, reqs: Sequence[Req], uncommitted: type) -> GrammarBatch | None:
        """Per-row states of a batch, rows of ``uncommitted`` requests (prefill chunks) left free; None when no row is constrained."""
        if not self._states:
            return None
        states = [None if isinstance(r, uncommitted) else self._states.get(r.uid) for r in reqs]
        if not any(s is not None and not s.broken for s in states):
            return None
        return GrammarBatch(states)


class GrammarBatch:
    def __init__(self, states: list[GrammarState | None]) -> None:
        self.states = states

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Mask one row per request in place; returns the fp32 logits the sampler should read."""
        import xgrammar as xgr

        logits = logits.float()
        bitmask = xgr.allocate_token_bitmask(logits.shape[0], logits.shape[-1])
        rows = []
        for i, state in enumerate(self.states):
            if state is None or not state.active or state.matcher.is_terminated():
                continue
            try:
                state.matcher.fill_next_token_bitmask(bitmask, i)
            except Exception as e:  # noqa: BLE001 -- a matcher fault frees its request, never the engine
                _break(state, e)
                continue
            rows.append(i)
        if rows:
            xgr.apply_token_bitmask_inplace(logits, _to_device(bitmask, logits.device), indices=rows)
        return logits

    def apply_verify(self, logits: torch.Tensor, drafts: torch.Tensor) -> torch.Tensor:
        """Mask the k+1 verify rows of each request in place, row j under the state after its first j drafts."""
        import xgrammar as xgr

        logits = logits.float()
        k = drafts.shape[1]
        width = k + 1
        bitmask = xgr.allocate_token_bitmask(logits.shape[0], logits.shape[-1])
        rows = []
        for i, state in enumerate(self.states):
            if state is None or state.broken:
                continue
            active, accepted = state.activate_id is None, 0
            try:
                for j in range(width):
                    if active:
                        # a drafted stop token ends the grammar; the rows after it are never committed
                        if state.matcher.is_terminated():
                            break
                        state.matcher.fill_next_token_bitmask(bitmask, i * width + j)
                        rows.append(i * width + j)
                    if j == k:
                        break
                    draft = int(drafts[i, j])
                    if active:
                        # an invalid draft cannot match its masked row, so the rows after it are never committed
                        if not state.matcher.accept_token(draft):
                            break
                        accepted += 1
                    elif draft == state.activate_id:
                        active = True
                if accepted:
                    state.matcher.rollback(accepted)
            except Exception as e:  # noqa: BLE001 -- a matcher fault frees its request, never the engine
                _break(state, e)
        if rows:
            xgr.apply_token_bitmask_inplace(logits, _to_device(bitmask, logits.device), indices=rows)
        return logits


def _break(state: GrammarState, error: Exception) -> None:
    state.broken = True
    logger.warning_rank0(f"response_format grammar failed ({error!r}); the request decodes unconstrained")


def _to_device(bitmask: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type != "cuda":
        return bitmask.to(device)
    return bitmask.pin_memory().to(device, non_blocking=True)
