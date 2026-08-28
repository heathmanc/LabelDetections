"""The hang watchdog's decisions.

Tested with injected clocks rather than by hanging anything: the value of this
module is that it fires at the right moment and not at the wrong ones, and both
of those are arithmetic.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import hang_watch as hw


def test_a_fresh_heartbeat_is_not_stale():
    beat = hw.Heartbeat(now=100.0)
    assert beat.stale_for(now=100.0) == 0.0
    assert beat.stale_for(now=103.0) == 3.0


def test_beating_resets_the_clock():
    beat = hw.Heartbeat(now=100.0)
    assert beat.stale_for(now=110.0) == 10.0
    beat.beat(now=110.0)
    assert beat.stale_for(now=110.5) == 0.5


def test_a_clock_that_goes_backwards_reads_as_fresh_not_negative():
    """monotonic should not, but a caller passing wall time might."""
    beat = hw.Heartbeat(now=100.0)
    assert beat.stale_for(now=95.0) == 0.0


# --- when to write the file -------------------------------------------------

def test_slow_work_is_not_reported_as_a_hang():
    """Loading a model or exporting a dataset blocks the GUI legitimately, and
    a watchdog that cries at every one of those is a watchdog nobody reads."""
    assert hw.should_dump(stale_s=2.0, since_last_dump_s=1e9) is False


def test_a_genuine_hang_is_reported():
    assert hw.should_dump(stale_s=6.0, since_last_dump_s=1e9) is True


def test_one_hang_produces_one_dump_not_one_per_poll():
    """A hang persists, and every poll would write the same picture again."""
    assert hw.should_dump(stale_s=30.0, since_last_dump_s=5.0) is False
    assert hw.should_dump(stale_s=30.0, since_last_dump_s=31.0) is True


def test_the_thresholds_can_be_tightened_for_a_specific_hunt():
    assert hw.should_dump(stale_s=1.0, since_last_dump_s=1e9,
                          threshold_s=0.5) is True
    assert hw.should_dump(stale_s=1.0, since_last_dump_s=2.0,
                          threshold_s=0.5, cooldown_s=1.0) is True


def test_the_header_says_what_was_seen_and_for_how_long():
    text = hw.header(7.25, now_text="2026-08-28 13:40:00")
    assert "7.2s" in text
    assert "2026-08-28 13:40:00" in text
    # It reports an observation; the stack under it is the evidence.
    assert "UNRESPONSIVE" in text


# --- it really fires, on a really blocked event loop -------------------------

def test_a_blocked_gui_thread_produces_a_stack_that_names_the_blocker(tmp_path):
    """The point of the whole module. Wiring that looks right and never fires
    is worth nothing, so this blocks the Qt event loop for real and reads the
    file afterwards."""
    import time

    import pytest
    qt = pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("LABELVISION_DATA_DIR", str(tmp_path / "data"))

    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from label_detections.ui.main_window import MainWindow

    win = MainWindow()
    log = tmp_path / "hang.log"
    win.start_hang_watch(threshold_s=0.4, path=log)

    # Let the heartbeat tick a few times so the watcher sees a healthy window.
    for _ in range(6):
        QApplication.processEvents()
        time.sleep(0.05)
    assert not log.exists(), "reported a hang while the event loop was running"

    def the_thing_that_hangs():
        # A distinctive frame name, so the assertion is about this stack and
        # not about any stack at all.
        time.sleep(2.0)

    the_thing_that_hangs()

    deadline = time.time() + 5.0
    while not log.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert log.exists(), "the GUI thread stopped for 2s and nothing was written"

    text = log.read_text(encoding="utf-8", errors="replace")
    assert "UNRESPONSIVE" in text
    assert "the_thing_that_hangs" in text, (
        "the dump did not capture the thread that was actually stuck")


def test_it_stops_reporting_once_the_window_recovers(tmp_path):
    import time

    import pytest
    pytest.importorskip("PySide6.QtWidgets")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("LABELVISION_DATA_DIR", str(tmp_path / "data2"))

    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from label_detections.ui.main_window import MainWindow

    win = MainWindow()
    log = tmp_path / "recover.log"
    win.start_hang_watch(threshold_s=0.4, path=log)

    time.sleep(1.2)                       # hang
    deadline = time.time() + 5.0
    while not log.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert log.exists()
    after_hang = log.stat().st_size

    # Now keep the event loop healthy and confirm nothing more is written.
    for _ in range(30):
        QApplication.processEvents()
        time.sleep(0.05)
    assert log.stat().st_size == after_hang, "kept reporting a recovered window"
