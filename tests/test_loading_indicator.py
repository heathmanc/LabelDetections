"""A model load says so, in front of the window rather than under it.

Loading a model is the longest thing this application does with the least to
show for it: no progress to report, no partial result, and on a first click a
download of up to 1.2 GB. All it had was a line in the status bar at the bottom
of the window -- which is where a message goes when it does not matter.

Two of the three loads block the GUI thread, so what they produce without this
is a window that stops repainting. That is indistinguishable from a hang, and
the usual response to it is to kill the application part way through writing a
1.2 GB file.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-loading-"))

import pytest

try:
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception as exc:  # pragma: no cover - depends on the environment
    HAVE_QT = False
    _WHY = exc

pytestmark = pytest.mark.skipif(not HAVE_QT, reason="PySide6 not available")

_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


@pytest.fixture(autouse=True)
def _no_dialog_left_behind():
    """The window is shared, and a dialog left up is modal over every later test."""
    yield
    _window()._hide_loading()


# --- the indicator itself ----------------------------------------------------

def test_it_is_up_and_readable_while_something_loads():
    win = _window()
    win._show_loading("Loading sam_l.pt  (~1.2 GB)", "Downloads once.")
    dialog = win._loading_dialog
    assert dialog is not None and dialog.isVisible()
    assert "sam_l.pt" in dialog.labelText()
    assert "Downloads once" in dialog.labelText()


def test_it_reports_no_progress_because_there_is_none():
    """Neither Ultralytics nor torch hands back a byte count. A bar that moved
    without knowing anything would be a lie told to look reassuring, so the
    range is empty and Qt draws a marquee."""
    win = _window()
    win._show_loading("Loading")
    assert (win._loading_dialog.minimum(), win._loading_dialog.maximum()) == (0, 0)


def test_it_cannot_be_cancelled():
    """A torch load and an in-flight download cannot be stopped part way and
    left in a state anything downstream would understand."""
    win = _window()
    win._show_loading("Loading")
    # QProgressDialog draws no cancel button when it has been given none, and
    # cancellation is what auto-reset would key off.
    assert not win._loading_dialog.autoReset()
    assert not win._loading_dialog.autoClose()


def test_showing_twice_leaves_one_dialog():
    win = _window()
    win._show_loading("First")
    first = win._loading_dialog
    win._show_loading("Second")
    assert win._loading_dialog is not first
    assert not first.isVisible()
    assert "Second" in win._loading_dialog.labelText()


def test_hiding_when_nothing_is_up_is_not_an_error():
    win = _window()
    win._hide_loading()
    win._hide_loading()
    assert win._loading_dialog is None


# --- the three places a model is loaded --------------------------------------

def test_the_test_model_load_is_wrapped_where_every_caller_goes_through():
    """Three places load a test model -- Run Model, pre-label, and the queue --
    and only a cache miss is slow. Wrapping the loader covers all three and
    fires on none of the hits."""
    import inspect

    from label_detections.ui.main_window import MainWindow

    source = inspect.getsource(MainWindow._load_test_model)
    assert "_show_loading" in source and "_hide_loading" in source
    # After the cache check, or every repeat click would flash a dialog.
    assert source.index("self._test_model_path == str(p)") < source.index("_show_loading")


def test_a_failed_load_still_takes_the_dialog_down():
    """It is in a finally. A model that fails to load is exactly the case where
    a stuck modal is worst: the error dialog behind it cannot be reached."""
    import inspect

    from label_detections.ui.main_window import MainWindow

    for method in (MainWindow._load_test_model, MainWindow._outline_at):
        source = inspect.getsource(method)
        assert "_show_loading" in source, f"{method.__name__} shows nothing"
        assert "finally:" in source, f"{method.__name__} can leave it up"
        assert "_hide_loading" in source.split("finally:")[-1], \
            f"{method.__name__} does not take it down on the way out"


def test_live_detect_takes_it_down_on_success_failure_and_stop():
    """It loads on a worker thread, so nothing in the start path can close it.
    Three things can end that wait and all three have to."""
    import inspect

    from label_detections.ui.main_window import MainWindow

    for name in ("_on_live_loaded", "_on_live_failed", "stop_live_detect"):
        source = inspect.getsource(getattr(MainWindow, name))
        assert "_hide_loading" in source, f"{name} leaves the dialog up"


def test_closing_the_window_takes_it_down():
    import inspect

    from label_detections.ui.main_window import MainWindow

    assert "_hide_loading" in inspect.getsource(MainWindow.closeEvent)


def test_the_outline_load_names_the_model_and_its_size():
    """1.2 GB arriving with no number attached is what reads as a hang."""
    from label_detections.ui.segment_assist import KNOWN_MODELS, note_for

    assert note_for("sam_l.pt").startswith("~1.2 GB")
    assert note_for("nobody/typed/this.pt") == ""
    for name, note in KNOWN_MODELS:
        assert " · " in note, f"{name} has no size to show while it loads"
