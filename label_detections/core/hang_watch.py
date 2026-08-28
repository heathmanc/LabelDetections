"""Notice when the GUI thread stops responding, and say where it is stuck.

faulthandler already covers the crash case: a segfault in torch, CUDA or Qt
prints every thread's C-level stack. A hang prints nothing at all. The process
is alive, the window is up, and the only thing anyone can report is that
clicking does nothing -- which is the same sentence for a blocked event loop, a
deadlock on a camera lock, and a worker thread that never came back.

So: the GUI thread stamps a heartbeat, and a daemon thread that touches nothing
else watches it. When the stamp goes stale, every thread's stack goes to a file.
The watcher is deliberately plain -- no Qt, no locks it shares with anything --
because a watchdog that can be blocked by the thing it watches is not one.

The decisions are here and tested; the thread and the timer are in the UI.
"""
from __future__ import annotations

import time

# How long the GUI thread may go without stamping before it counts as stuck.
# Long enough that a slow-but-working operation is not reported -- loading a
# model, exporting a dataset, one heavy inference -- short enough that an
# operator who has just watched the window die does not sit through a minute of
# nothing before there is any evidence.
DEFAULT_THRESHOLD_S = 5.0

# A hang usually persists, and dumping every thread's stack on every check
# would fill a log with the same picture. Once, then not again for this long.
DEFAULT_COOLDOWN_S = 30.0

# How often the watcher looks. Fine enough to catch the threshold promptly,
# coarse enough that the thread is asleep essentially all of the time.
DEFAULT_POLL_S = 1.0


class Heartbeat:
    """A timestamp the watched thread updates and the watcher reads.

    No lock. A float assignment is atomic under the GIL, and a watchdog that
    takes a lock the stuck thread might hold is a watchdog that hangs with it.
    """

    def __init__(self, now: float | None = None) -> None:
        self._at = time.monotonic() if now is None else float(now)

    def beat(self, now: float | None = None) -> None:
        self._at = time.monotonic() if now is None else float(now)

    def stale_for(self, now: float | None = None) -> float:
        """Seconds since the last beat."""
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self._at)


def should_dump(stale_s: float, since_last_dump_s: float,
                threshold_s: float = DEFAULT_THRESHOLD_S,
                cooldown_s: float = DEFAULT_COOLDOWN_S) -> bool:
    """Is this the moment to write every thread's stack out?

    Both conditions, not either: stale enough to be a hang rather than slow
    work, and long enough since the last dump that the file gets one picture of
    each hang instead of one per poll.
    """
    return float(stale_s) >= float(threshold_s) \
        and float(since_last_dump_s) >= float(cooldown_s)


def header(stale_s: float, now_text: str = "") -> str:
    """What goes above a dump, so a log of several is readable.

    Says what was observed rather than what it means: the stack underneath is
    the evidence, and a header that guesses at a cause is a header that will
    eventually be wrong and believed anyway.
    """
    when = now_text or time.strftime("%Y-%m-%d %H:%M:%S")
    return (f"\n{'=' * 72}\n"
            f"GUI THREAD UNRESPONSIVE for {float(stale_s):.1f}s at {when}\n"
            f"Every thread's stack follows. The GUI thread is the one inside "
            f"the Qt event loop.\n"
            f"{'=' * 72}\n")
