"""An idle scheduler flushes a dirty host tier once it has been idle long enough, without waiting
for the next request to wake it."""
from __future__ import annotations

import time
from types import SimpleNamespace

from freetoken.scheduler.io import SchedulerIOMixin
from freetoken.scheduler.scheduler import Scheduler


class NeverReady:
    def __init__(self):
        self.waits: list[int] = []

    def wait(self, timeout_ms: int) -> bool:
        self.waits.append(timeout_ms)
        return False

    def get(self):
        return "msg"

    def empty(self) -> bool:
        return True


class Loop(SchedulerIOMixin):
    """The receive side of the scheduler over a queue that stays empty until the tier is clean."""

    def __init__(self, dirty_ticks: int):
        self._recv_from_tokenizer = NeverReady()
        self.dirty_ticks = dirty_ticks
        self.ticks = 0

    def run_when_idle(self) -> None:
        pass

    def idle_wait_ms(self) -> int | None:
        return 5 if self.ticks < self.dirty_ticks else None

    def run_idle_tick(self) -> None:
        self.ticks += 1


def test_the_idle_loop_ticks_while_the_tier_is_dirty_then_blocks_for_a_message():
    loop = Loop(dirty_ticks=2)
    assert loop._recv_msg_single_rank(blocking=True) == ["msg"]
    assert loop.ticks == 2
    assert loop._recv_from_tokenizer.waits == [5, 5]


def _sched(dirty: bool, idle_for: float, flush_idle: float = 3.0):
    flushed: list[float] = []

    def maybe_flush(idle_seconds: float, min_idle: float):
        if idle_seconds < min_idle:
            return None
        flushed.append(idle_seconds)
        tier.dirty = False
        return {"nodes": 1}

    tier = SimpleNamespace(dirty=dirty)
    cm = SimpleNamespace(is_tiered=True, host_tier=tier, maybe_flush_host_meta=maybe_flush)
    s = SimpleNamespace(cache_manager=cm, _prefix_cache_flush_idle=flush_idle,
                        _last_activity=time.monotonic() - idle_for)
    return s, flushed


def test_the_wait_runs_to_the_flush_deadline_and_the_tick_flushes_once_it_passed():
    s, flushed = _sched(dirty=True, idle_for=1.0)
    wait = Scheduler.idle_wait_ms(s)
    assert 1900 <= wait <= 2100
    Scheduler.run_idle_tick(s)
    assert flushed == []
    s._last_activity -= 2.5
    Scheduler.run_idle_tick(s)
    assert len(flushed) == 1 and Scheduler.idle_wait_ms(s) is None


def test_a_clean_or_untiered_cache_never_wakes_the_idle_loop():
    s, _ = _sched(dirty=False, idle_for=100.0)
    assert Scheduler.idle_wait_ms(s) is None
    s.cache_manager.is_tiered = False
    s.cache_manager.host_tier.dirty = True
    assert Scheduler.idle_wait_ms(s) is None


def test_a_failing_flush_keeps_the_scheduler_serving_and_retries_after_another_idle_period():
    import errno

    s, _ = _sched(dirty=True, idle_for=10.0)

    def full(idle_seconds, min_idle):
        raise OSError(errno.ENOSPC, "No space left on device")

    s.cache_manager.maybe_flush_host_meta = full
    Scheduler.run_idle_tick(s)
    assert s.cache_manager.host_tier.dirty
    assert Scheduler.idle_wait_ms(s) >= 2900
