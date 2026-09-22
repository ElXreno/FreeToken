"""systemd signals every process in the cgroup, so SIGTERM arrives more than once.

The scheduler turns SIGTERM into KeyboardInterrupt to reach its graceful path, where the host
tier flushes its prefix-cache metadata. A handler that stays armed lets the second signal
interrupt that flush, and the arena is left without the metadata it just spent a session
building -- visible as a traceback through flush_host_meta and no "flushed on stop" line.
"""

from __future__ import annotations

import os
import signal


def arm() -> None:
    """The handler as run_forever installs it."""

    def _term(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)


def test_second_sigterm_does_not_interrupt_the_flush():
    previous = signal.getsignal(signal.SIGTERM)
    try:
        arm()
        flushed = False
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except KeyboardInterrupt:
            # the graceful path: a second signal must not land in here
            os.kill(os.getpid(), signal.SIGTERM)
            os.kill(os.getpid(), signal.SIGTERM)
            flushed = True
        assert flushed, "first SIGTERM did not reach the graceful path"
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_handler_is_installed_and_restorable():
    previous = signal.getsignal(signal.SIGTERM)
    try:
        arm()
        assert callable(signal.getsignal(signal.SIGTERM))
    finally:
        signal.signal(signal.SIGTERM, previous)
        assert signal.getsignal(signal.SIGTERM) is previous
