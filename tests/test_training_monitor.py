"""Following a training run, rather than reading its console output.

Everything the monitor shows was already on screen: it was in a scrolling wall
of yolo output, in a pane narrow enough that one progress line wrapped three
times, under a chart small enough to be decorative. What was missing was not
the information but any arrangement of it that answered the questions somebody
watching a run actually has -- how far, how long, how much longer, and is it
still getting better.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-monitor-"))

import pytest

from label_detections.core import training as tr

# One real epoch line, of the shape Ultralytics prints for a detection run.
EPOCH_LINE = ("      34/100      2.35G      1.234      0.987      1.456"
              "         12        640: 100%|##########| 5/5 [00:02<00:00]")


def _rows(best_at: int = 9, last: int = 12) -> list[dict]:
    return [{"epoch": float(e),
             "train/box_loss": 1.9 - 0.06 * e,
             "metrics/mAP50(B)": 0.30 + 0.05 * e,
             "metrics/mAP50-95(B)": 0.18 + 0.05 * min(e, best_at)}
            for e in range(1, last + 1)]


# --- reading the epoch off the output ----------------------------------------

def test_the_epoch_counter_is_read_from_the_start_of_the_line():
    """The same line carries a batch counter further along. A pattern that
    matched anywhere would follow "5/5" and report epoch 5 of 5 on every epoch
    of a hundred."""
    assert tr.parse_epoch(EPOCH_LINE) == (34, 100)


def test_the_latest_epoch_in_a_chunk_wins():
    """stdout arrives in whatever sizes the pipe hands over, so a chunk can
    carry several epochs and the last one is the current one."""
    chunk = "\n".join(EPOCH_LINE.replace("34/100", f"{e}/100") for e in (7, 8, 9))
    assert tr.parse_epoch(chunk) == (9, 100)


def test_colour_codes_do_not_hide_the_epoch():
    assert tr.parse_epoch("\x1b[34m      2/50\x1b[0m   2.1G   1.0") == (2, 50)


def test_output_with_no_epoch_in_it_reports_nothing():
    for text in ("", "Ultralytics 8.3.0 starting", "Results saved to runs/obb/x",
                 "  Class  Images  Instances  Box(P  R  mAP50"):
        assert tr.parse_epoch(text) is None


def test_a_nonsense_counter_is_refused():
    """Past the total is not an epoch, it is some other pair of numbers."""
    assert tr.parse_epoch("      120/100   2.3G") is None
    assert tr.parse_epoch("      3/0   2.3G") is None


# --- how much longer ---------------------------------------------------------

def test_the_estimate_is_the_rate_so_far():
    # Twelve of a hundred in ten minutes: eighty-eight left at fifty seconds
    # each.
    assert tr.eta_seconds(12, 100, 600) == pytest.approx(4400.0)


def test_nothing_is_estimated_before_there_is_a_rate():
    assert tr.eta_seconds(0, 100, 0) == 0.0
    assert tr.eta_seconds(0, 100, 30) == 0.0
    assert tr.eta_seconds(100, 100, 600) == 0.0, "finished has nothing left"
    assert tr.eta_seconds(5, 0, 600) == 0.0, "no total, no estimate"


def test_the_progress_line_carries_all_three_numbers():
    line = tr.progress_text(12, 100, 600)
    assert "Epoch 12 of 100" in line
    assert "10m 0s elapsed" in line
    assert "about 1h 13m 20s left" in line


def test_the_progress_line_says_something_before_the_first_epoch():
    assert tr.progress_text(0, 0, 0) == "Starting..."


# --- is it still improving ---------------------------------------------------

def test_the_best_epoch_and_how_long_ago_it_was():
    note = tr.stall_note(_rows(best_at=9, last=12), patience=50)
    assert "at epoch 9" in note
    assert "3 epoch(s) ago" in note
    assert "stops at 50" in note


def test_a_run_still_improving_says_so():
    note = tr.stall_note(_rows(best_at=12, last=12), patience=50)
    assert "still improving" in note


def test_reaching_the_patience_is_called_out():
    note = tr.stall_note(_rows(best_at=9, last=12), patience=3)
    assert "at the 3-epoch patience, so it stops here" in note


def test_it_leads_with_the_metric_the_ranking_used():
    """summarize_results picks the best EPOCH by mAP50-95. Naming mAP50 beside
    that epoch number reports one metric's value at another metric's argmax --
    two right numbers that do not belong together."""
    note = tr.stall_note(_rows(), patience=0)
    assert note.startswith("Best mAP50-95 ")


def test_a_classifier_run_leads_with_its_own_metric():
    rows = [{"epoch": float(e), "metrics/accuracy_top1": 0.5 + 0.04 * e}
            for e in range(1, 11)]
    assert tr.stall_note(rows).startswith("Best accuracy_top1 ")


def test_nothing_measured_yet_says_nothing():
    assert tr.stall_note([], patience=50) == ""


# --- the window --------------------------------------------------------------

qt = pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


def _monitor():
    monitor = _window()._training_monitor
    monitor.begin("detector", "yolo11s-obb.pt", "data/exports/x/data.yaml")
    return monitor


def test_the_bar_counts_epochs_once_the_run_says_how_many():
    """Indeterminate until then. Ultralytics prints the count itself, so
    guessing it from the settings would be a second answer to a question that
    has one source."""
    monitor = _monitor()
    assert (monitor.bar.minimum(), monitor.bar.maximum()) == (0, 0)

    monitor.append_output(EPOCH_LINE)
    assert monitor.bar.maximum() == 100
    assert monitor.bar.value() == 34
    assert "epochs" in monitor.bar.format()


def test_the_curves_and_the_bar_agree_about_how_far_in_it_is():
    """results.csv is the authority: stdout can be buffered away for a while,
    and a bar that stalls while the curves move is a bar nobody believes."""
    monitor = _monitor()
    monitor.append_output(EPOCH_LINE.replace("34/100", "2/100"))
    monitor.set_metrics(_rows(last=12), patience=50)
    assert monitor.bar.value() == 12


def test_the_run_is_named_so_two_stages_are_not_confused():
    monitor = _monitor()
    assert "detector" in monitor.what_label.text()
    monitor.begin("classifier", "yolo11s-cls.pt", "data/exports/crops")
    assert "classifier" in monitor.what_label.text()


def test_starting_a_run_clears_the_last_one():
    monitor = _monitor()
    monitor.append_output(EPOCH_LINE)
    monitor.set_metrics(_rows(), patience=50)
    assert monitor.best_label.text()

    monitor.begin("detector", "m.pt", "d.yaml")
    assert monitor.log.toPlainText() == ""
    assert monitor.best_label.text() == ""
    assert not monitor.use_btn.isEnabled()


def test_the_finished_run_offers_its_weights_only_when_there_are_any(tmp_path):
    monitor = _monitor()
    monitor.finish("Finished", "2m 0s in total.", tmp_path / "nothing" / "best.pt")
    assert not monitor.use_btn.isEnabled()
    assert "No weights at" in monitor.detail_label.text()

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not really weights")
    monitor.finish("Finished", "2m 0s in total.", weights)
    assert monitor.use_btn.isEnabled()


def test_using_the_model_puts_it_where_the_rest_of_the_app_reads_it(tmp_path):
    """The step missing at the end of every run: the weights land under runs/
    with a name Ultralytics chose, and using them meant copying a path out of a
    log."""
    win = _window()
    monitor = _monitor()
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not really weights")

    win._train_stage = "detector"
    win._test_model = object()
    win._use_trained_model(str(weights))
    assert win.test_model_edit.text() == str(weights)
    # And the model already in memory is the previous one.
    assert win._test_model is None

    win._train_stage = "classifier"
    win._use_trained_model(str(weights))
    assert win.live_classifier_edit.text() == str(weights)


def test_stopping_is_wired_to_the_window_that_started_it():
    """Clicked, not introspected. The button is on the dialog and the process
    is on the main window, so nothing about that connection is obvious."""
    win = _window()
    called = []
    real = win.stop_training
    try:
        win.stop_training = lambda: called.append(True)
        # Reconnect through the stub the same way the window does.
        win._training_monitor.stop_btn.clicked.disconnect()
        win._training_monitor.stop_btn.clicked.connect(win.stop_training)
        win._training_monitor.stop_btn.click()
        assert called == [True]
    finally:
        win.stop_training = real
        win._training_monitor.stop_btn.clicked.disconnect()
        win._training_monitor.stop_btn.clicked.connect(win.stop_training)


def test_the_pane_no_longer_carries_the_log_or_the_chart():
    """Both are in the monitor. A yolo progress line wrapped three times in a
    340 px rail, and the chart there was decorative."""
    win = _window()
    page = win.tabs.widget([win.tabs.tabText(i)
                            for i in range(win.tabs.count())].index("Train"))
    assert win.train_log is win._training_monitor.log
    assert win.train_metrics_chart is win._training_monitor.chart
    assert not _is_descendant(win.train_log, page)
    assert not _is_descendant(win.train_metrics_chart, page)


def _is_descendant(widget: QWidget, ancestor: QWidget) -> bool:
    node = widget.parentWidget()
    while node is not None:
        if node is ancestor:
            return True
        node = node.parentWidget()
    return False
