"""Per-phase host and device time for one decode iteration, enabled by FREETOKEN_HOST_TIMING.

Each phase carries its call count as well as its total, which is what separates "this costs a
millisecond every step" from "one warmup iteration cost a second". Reports are windowed for the
same reason: a cumulative mean hides that outlier for hundreds of steps.
"""

from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter_ns

from freetoken.env import ENV

_TOTALS: dict[str, int] = {}
_COUNTS: dict[str, int] = {}
_PREV: dict[str, tuple[int, int]] = {}
_ORDER: list[str] = []
_OFF = nullcontext()


class _Timer:
    __slots__ = ("name", "t0")

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.t0 = perf_counter_ns()
        return self

    def __exit__(self, *exc):
        add(self.name, perf_counter_ns() - self.t0)
        return False


def phase(name: str):
    """Time the block, or hand back a shared no-op when the counters are off."""
    return _Timer(name) if ENV.HOST_TIMING else _OFF


def add(name: str, ns: int) -> None:
    if not ENV.HOST_TIMING:
        return
    if name not in _TOTALS:
        _ORDER.append(name)
        _TOTALS[name] = 0
        _COUNTS[name] = 0
    _TOTALS[name] += ns
    _COUNTS[name] += 1


def reset() -> None:
    _TOTALS.clear()
    _COUNTS.clear()
    _PREV.clear()
    _ORDER.clear()


def report(iters: int) -> str:
    """Averages over the window since the last report, each with the calls that made it."""
    out = []
    for name in _ORDER:
        p_ns, p_n = _PREV.get(name, (0, 0))
        d_ns, d_n = _TOTALS[name] - p_ns, _COUNTS[name] - p_n
        _PREV[name] = (_TOTALS[name], _COUNTS[name])
        out.append(f"{name} {d_ns / iters / 1e6:.2f}/{d_n}")
    return " ".join(out)
