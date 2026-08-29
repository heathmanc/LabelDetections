"""Live detection: pacing, keeping hard frames, and the coordinate scaling.

The policy is stdlib and tested bare. The UI half is tested offscreen with a
stand-in worker, because the thing worth pinning is the wiring -- what gets
handed to the model, what comes back, and where the overlays land -- not
whether Ultralytics works.
"""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-live-"))

import pytest

from label_detections.core import live_detect as ld


# --- pacing ----------------------------------------------------------------

def test_a_busy_model_never_gets_another_frame():
    """Skipping while busy is what keeps the preview at the camera's rate
    instead of stuttering along at the model's."""
    assert ld.should_infer(busy=False, since_last_s=1.0) is True
    assert ld.should_infer(busy=True, since_last_s=1.0) is False


def test_inference_does_not_start_more_often_than_the_floor():
    assert ld.should_infer(False, ld.MIN_INTERVAL_S - 0.01) is False
    assert ld.should_infer(False, ld.MIN_INTERVAL_S) is True


def test_the_rate_counts_skipped_frames_rather_than_inverting_the_latency():
    """They differ whenever frames are skipped, and the honest number counts them."""
    rolling = ld.Rolling()
    for i in range(5):
        rolling.record(0.01, now=100 + i * 0.5)   # 10 ms of work, twice a second
    assert rolling.mean_ms == pytest.approx(10.0)
    assert rolling.rate == pytest.approx(2.0)


def test_an_empty_window_reports_zero_rather_than_dividing_by_nothing():
    rolling = ld.Rolling()
    assert rolling.mean_ms == 0.0 and rolling.rate == 0.0


# --- the capture gate ------------------------------------------------------

def test_a_frame_the_model_handles_is_not_kept():
    keep, why = ld.CaptureGate().consider(0.5, now=100)
    assert keep is False and "handling this frame" in why


def test_a_frame_the_model_struggles_with_is_kept():
    keep, why = ld.CaptureGate().consider(9.0, now=100)
    assert keep is True and "struggled" in why


def test_one_hard_battery_cannot_produce_hundreds_of_frames():
    """They would all say the same thing and each would cost a review."""
    gate = ld.CaptureGate()
    assert gate.consider(9.0, now=100)[0] is True
    gate.mark(now=100)
    assert gate.consider(9.0, now=100.5)[0] is False
    assert gate.consider(9.0, now=101.0)[0] is False
    assert gate.consider(9.0, now=100 + gate.cooldown_s)[0] is True


def test_the_cooldown_says_how_long_is_left():
    gate = ld.CaptureGate()
    gate.mark(now=100)
    _keep, why = gate.consider(9.0, now=101)
    assert "cooling down" in why and "s)" in why


def test_walking_away_from_an_armed_view_cannot_fill_a_disk():
    gate = ld.CaptureGate(limit=3, cooldown_s=0.0)
    for i in range(3):
        assert gate.consider(9.0, now=100 + i)[0] is True
        gate.mark(now=100 + i)
    keep, why = gate.consider(9.0, now=200)
    assert keep is False and "session limit" in why


def test_resetting_the_gate_starts_a_fresh_session():
    gate = ld.CaptureGate(limit=1, cooldown_s=0.0)
    gate.mark(now=100)
    assert gate.consider(9.0, now=200)[0] is False
    gate.reset()
    assert gate.consider(9.0, now=200)[0] is True


# --- the readout -----------------------------------------------------------

def test_the_summary_leads_with_the_active_label_when_it_is_there():
    rolling = ld.Rolling()
    rolling.record(0.02, now=100)
    rolling.record(0.02, now=100.2)
    found = ld.frame_summary({"sp_g31": 1, "warn_g31": 2}, "sp_g31", rolling)
    assert "sp_g31: 1 found" in found
    assert "warn_g31: 2" in found


def test_the_summary_says_nothing_about_a_label_that_is_not_presented():
    """The label open in the labeling tab is not necessarily the one under the
    camera. Saying it is missing complains about the wrong thing on every
    frame of every other label."""
    rolling = ld.Rolling()
    rolling.record(0.02, now=100)
    missed = ld.frame_summary({"warn_g31": 1}, "sp_g31", rolling)
    assert "sp_g31" not in missed
    assert "warn_g31: 1" in missed


def test_the_readout_names_the_id_the_recipe_is_written_in():
    """The front end counts label ids, so a readout that cannot name one is
    not showing the operator the same thing the line will judge."""
    rolling = ld.Rolling()
    rolling.record(0.02, now=100)

    found = ld.frame_summary({"2220-9199": 1}, "2220-9199", rolling)
    assert "2220-9199: 1 found" in found

    book = ld.TrackBook()
    book.update([(1, "2220-9199", 0.94)], now=0.0)
    tracked = ld.track_summary(book, "2220-9199", rolling)
    assert "2220-9199 0.94" in tracked
    assert "this label" in tracked


def test_other_labels_on_the_battery_stay_visible_rather_than_filtered():
    """The recipe counts every label on the side, not just the one being
    trained, so hiding the rest would hide most of the answer."""
    rolling = ld.Rolling()
    rolling.record(0.02, now=100)
    book = ld.TrackBook()
    book.update([(1, "2220-9199", 0.94), (2, "2220-9200", 0.81)], now=0.0)
    text = ld.track_summary(book, "2220-9199", rolling)
    assert "2220-9199" in text and "2220-9200" in text


def test_the_armed_note_never_looks_like_it_hung():
    assert "0/50 kept" in ld.capture_note(0, 50, "")
    assert "Last: cooling down" in ld.capture_note(3, 50, "cooling down (1.2s)")


# --- the UI wiring ---------------------------------------------------------

try:
    import cv2
    import numpy as np
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception:  # pragma: no cover - depends on the environment
    HAVE_QT = False

ui = pytest.mark.skipif(not HAVE_QT, reason="PySide6/cv2 not available")

_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


class _FakeOpenCamera:
    """Open, threaded, and never asked for a frame -- enough for _live_running."""
    threaded = True

    def is_open(self):
        return True

    def read_fps(self):
        return 17.0


def _watch_infer(win, sink: list):
    """Collect the frames the pump submits.

    submit() is an ordinary call again: the worker runs on a plain thread with
    a queue, not a QThread with an event loop, so there is no signal in the
    path to watch.
    """
    win._live_worker = type(
        "StubWorker", (), {"submit": lambda _s, f: sink.append(f)})()


def _label(win, label_id="live_sp", ):
    from label_detections.core import persistence
    from label_detections.core.labels import LabelDef

    from conftest import reference_for

    library = persistence.load_library()
    library.add(LabelDef(label_id=label_id,
                         reference_images=[reference_for(label_id)]),
                replace=True)
    persistence.save_library(library)
    win.library = persistence.load_library()
    win.set_active_label(label_id)
    return label_id


class _Results:
    """The shape of one Ultralytics result, enough for the overlay code.

    ``ids`` present is what a tracked result looks like; absent is a plain
    predict, where ``boxes.id`` is None.
    """

    def __init__(self, boxes, names, confs, classes, ids=None):
        self.names = names
        self.obb = None

        class _Boxes:
            def __init__(self):
                self.xyxy = np.array(boxes, dtype=np.float32)
                self.conf = np.array(confs, dtype=np.float32)
                self.cls = np.array(classes, dtype=np.float32)
                self.id = np.array(ids, dtype=np.float32) if ids is not None else None

        self.boxes = _Boxes()


def _items(results):
    """What the worker now emits: plain dicts, via the real extractor."""
    from label_detections.ui.live_detect import extract_items
    return extract_items(results)


@ui
def test_overlays_are_scaled_from_the_inferred_frame_to_the_preview():
    """Inference runs at full resolution -- what production hands the model --
    while the canvas shows a downscaled preview. Skip the scaling and every box
    lands in the wrong place."""
    win = _window()
    items = [{"type": "other_box", "xyxy": [100.0, 200.0, 300.0, 400.0],
              "cx": 200.0, "cy": 300.0, "points": [[100.0, 200.0], [300.0, 200.0],
                                                   [300.0, 400.0], [100.0, 400.0]]}]
    scaled = win._scaled_overlay_items(items, (0.5, 0.5))
    assert scaled[0]["xyxy"] == [50.0, 100.0, 150.0, 200.0]
    assert scaled[0]["cx"] == 100.0 and scaled[0]["cy"] == 150.0
    assert scaled[0]["points"][2] == [150.0, 200.0]
    # The originals are not mutated: the same items feed the readout.
    assert items[0]["xyxy"] == [100.0, 200.0, 300.0, 400.0]


@ui
def test_a_one_to_one_preview_skips_the_scaling_entirely():
    win = _window()
    items = [{"type": "other_box", "xyxy": [1.0, 2.0, 3.0, 4.0]}]
    assert win._scaled_overlay_items(items, (1.0, 1.0)) is items


@ui
def test_nothing_is_handed_to_the_model_while_it_is_busy():
    win = _window()
    handed = []
    win._live_thread = object()          # pretend it is running
    win._live_loaded = True              # ...and that the model finished loading
    _watch_infer(win, handed)
    win._live_busy = True
    # Recent enough that the lost-result watchdog stays out of it, old enough
    # that the minimum-interval floor is satisfied -- so this tests the busy
    # flag and nothing else.
    win._live_last_started = time.monotonic() - 1.0
    try:
        win._pump_live_detect(np.zeros((10, 10, 3), np.uint8))
        assert handed == []
        win._live_busy = False
        win._pump_live_detect(np.zeros((10, 10, 3), np.uint8))
        assert len(handed) == 1
    finally:
        win._live_thread = None
        win._live_worker = None
        win._live_busy = False
        win._live_loaded = False


@ui
def test_keeping_a_frame_puts_it_in_the_active_labels_dataset():
    from label_detections.core import storage

    win = _window()
    label_id = _label(win, "live_keep")
    win._live_frame = np.zeros((60, 80, 3), np.uint8)
    before = len(storage.list_images(label_id))
    win.keep_live_frame()
    assert len(storage.list_images(label_id)) == before + 1


@ui
def test_a_kept_frame_is_not_reviewed_because_nobody_has_labeled_it():
    from label_detections.core import persistence, review, storage

    win = _window()
    label_id = _label(win, "live_unreviewed")
    win._live_frame = np.zeros((60, 80, 3), np.uint8)
    win.keep_live_frame()
    newest = storage.list_images(label_id)[-1]
    data = persistence.load_annotation(label_id, newest.name)
    assert review.annotation_status(data, label_id) == "unlabeled"


@ui
def test_keeping_with_no_label_open_says_so_rather_than_guessing():
    win = _window()
    win.label_id = ""
    win._live_frame = np.zeros((10, 10, 3), np.uint8)

    shown = {}
    import label_detections.ui.main_window as mw_mod
    original = mw_mod.QMessageBox.information
    mw_mod.QMessageBox.information = lambda parent, title, text, *a, **k: shown.update(
        {"text": text})
    try:
        win.keep_live_frame()
    finally:
        mw_mod.QMessageBox.information = original
    assert "Open a label first" in shown.get("text", "")


@ui
def test_an_armed_view_keeps_a_frame_the_model_missed():
    from label_detections.core import storage

    win = _window()
    label_id = _label(win, "live_armed")
    win.live_auto_check.setChecked(True)
    win._live_thread = object()
    win._live_frame = np.zeros((60, 80, 3), np.uint8)
    win._live_overlay_scale = (1.0, 1.0)
    win._live_gate = ld.CaptureGate(cooldown_s=0.0)
    before = len(storage.list_images(label_id))
    try:
        # Nothing detected at all: the strongest signal the image is worth having.
        win._on_live_result(_items([_Results([], {}, [], [])]), 0.02)
        assert len(storage.list_images(label_id)) == before + 1
        assert "kept" in win.live_capture_label.text()
    finally:
        win.live_auto_check.setChecked(False)
        win._live_thread = None


@ui
def test_an_armed_view_leaves_a_frame_the_model_handled():
    from label_detections.core import storage

    win = _window()
    label_id = _label(win, "live_handled")
    win.live_auto_check.setChecked(True)
    win._live_thread = object()
    win._live_frame = np.zeros((60, 80, 3), np.uint8)
    win._live_overlay_scale = (1.0, 1.0)
    win._live_gate = ld.CaptureGate(cooldown_s=0.0)
    before = len(storage.list_images(label_id))
    try:
        confident = _items([_Results([[10, 10, 40, 40]], {0: win.label_id}, [0.97], [0])])
        win._on_live_result(confident, 0.02)
        assert len(storage.list_images(label_id)) == before
    finally:
        win.live_auto_check.setChecked(False)
        win._live_thread = None


@ui
def test_the_untracked_readout_reports_what_the_model_saw():
    win = _window()
    _label(win, "live_readout")
    win._live_thread = object()
    win._live_tracking = False
    win._live_overlay_scale = (1.0, 1.0)
    try:
        win._on_live_result(
            _items([_Results([[10, 10, 40, 40]], {0: win.label_id}, [0.9], [0])]), 0.02)
        text = win.live_readout.toPlainText()
        assert "1 detection(s)" in text
        assert f"{win.label_id}: 1 found" in text
    finally:
        win._live_thread = None
        win._live_tracking = True


@ui
def test_the_tracked_readout_reports_a_held_average_not_a_flicker():
    """One frame's confidence says almost nothing; a held average says whether
    the model actually has the object."""
    win = _window()
    _label(win, "live_tracked")
    win._live_thread = object()
    win._live_tracking = True
    win._live_tracks = ld.TrackBook()
    win._live_overlay_scale = (1.0, 1.0)
    try:
        # Chosen so the mean (0.90) and the last frame (1.00) differ: with
        # 0.88/0.94/0.91 they are both 0.91 and the assertion proves nothing.
        for conf in (0.80, 0.90, 1.00):
            win._on_live_result(_items(
                [_Results([[10, 10, 40, 40]], {0: win.label_id}, [conf], [0],
                          ids=[7])]), 0.02)
        text = win.live_readout.toPlainText()
        assert "1 tracked" in text
        assert f"{win.label_id} 0.90" in text, "showed the latest frame, not the mean"
        assert "frames" not in text
    finally:
        win._live_thread = None


@ui
def test_stopping_the_camera_stops_inference_too():
    """A worker left running holds a model and a thread for no reason."""
    win = _window()
    stopped = []
    original = win.stop_live_detect
    win.stop_live_detect = lambda: stopped.append(True)
    win._live_thread = object()
    try:
        win.close_camera()
        assert stopped == [True]
    finally:
        win.stop_live_detect = original
        win._live_thread = None


@ui
def test_it_refuses_to_start_without_a_model_chosen():
    win = _window()
    win.test_model_edit.setText("")
    shown = {}
    import label_detections.ui.main_window as mw_mod
    original = mw_mod.QMessageBox.information
    mw_mod.QMessageBox.information = lambda parent, title, text, *a, **k: shown.update(
        {"text": text})
    try:
        win.start_live_detect()
    finally:
        mw_mod.QMessageBox.information = original
    assert "Test Models tab" in shown.get("text", "")
    assert win._live_running() is False


# --- tracking --------------------------------------------------------------

def test_a_track_accumulates_rather_than_reporting_one_frame():
    book = ld.TrackBook()
    for i, conf in enumerate((0.90, 0.94, 0.86)):
        book.update([(1, "spec_plate", conf)], now=100 + i * 0.1)
    track = book.rows()[0]
    assert track.frames == 3
    assert track.last_conf == pytest.approx(0.86)
    assert track.mean_conf == pytest.approx(0.90)
    assert (track.min_conf, track.max_conf) == pytest.approx((0.86, 0.94))


def test_the_longest_held_track_is_listed_first():
    book = ld.TrackBook()
    for i in range(5):
        book.update([(1, "spec_plate", 0.9)], now=100 + i * 0.1)
    book.update([(2, "cert_mark", 0.9)], now=100.5)
    assert [t.track_id for t in book.rows()] == [1, 2]


def test_an_object_that_leaves_stops_being_listed():
    book = ld.TrackBook(ttl_s=1.0)
    book.update([(1, "spec_plate", 0.9), (2, "cert_mark", 0.7)], now=100)
    book.update([(1, "spec_plate", 0.9)], now=102)
    assert [t.track_id for t in book.rows()] == [1]


def test_a_track_whose_class_changes_restarts_rather_than_averaging_two():
    """The tracker kept the id but the classifier changed its mind. Averaging
    those together would report a confident detection of neither."""
    book = ld.TrackBook()
    for i in range(4):
        book.update([(1, "spec_plate", 0.95)], now=100 + i * 0.1)
    book.update([(1, "warning_label", 0.60)], now=100.5)
    track = book.rows()[0]
    assert track.name == "warning_label"
    assert track.frames == 1
    assert book.reacquired == 1


def test_detections_with_no_id_are_ignored_by_the_book():
    """A plain predict has no ids; the book must not invent them."""
    book = ld.TrackBook()
    book.update([(None, "spec_plate", 0.9)], now=100)
    assert book.rows() == []


def test_the_track_summary_reports_what_is_there_not_what_is_open():
    """"NOT TRACKED" named the wrong mechanism -- nothing had lost a track --
    and it fired whenever the operator presented any label but the open one."""
    book = ld.TrackBook()
    book.update([(1, "cert_mark", 0.9)], now=100)
    rolling = ld.Rolling()
    rolling.record(0.01, now=100)
    text = ld.track_summary(book, "sp_g31", rolling)
    assert "sp_g31" not in text
    assert "cert_mark 0.90" in text


def test_the_open_label_is_marked_when_it_is_one_of_the_tracks():
    """Silent when absent, but still pointed out when present -- that costs no
    line and answers "which of these is the one I am labeling"."""
    book = ld.TrackBook()
    book.update([(1, "cert_mark", 0.9), (2, "sp_g31", 0.8)], now=100)
    rolling = ld.Rolling()
    rolling.record(0.01, now=100)
    text = ld.track_summary(book, "sp_g31", rolling)
    assert "sp_g31 0.80  <-- this label" in text
    assert "<-- this label" not in text.split("\n")[1]


def test_an_empty_book_reads_as_empty_rather_than_blank():
    assert ld.TrackBook().text() == "No tracked objects."


# --- the two confidences ----------------------------------------------------
#
# The drawn plate carries stage 2's number and the readout used to carry stage
# 1's, with nothing on screen saying so. The same battery read 1.00 on the box
# and 0.91 in the pane, which looks exactly like a bug and is not one: a
# converged classifier over a handful of classes reports its top-1 at 1.00
# almost always, while the detector's box confidence is the one that moves.

def test_a_track_keeps_the_classifier_confidence_apart_from_the_box_one():
    """Averaged together they would be one number that answers neither
    question -- is there a label here, and which label is it."""
    book = ld.TrackBook()
    for i, (box, ident) in enumerate(((0.90, 1.00), (0.94, 0.98), (0.86, 1.00))):
        book.update([(1, "spec_plate", box, ident)], now=100 + i * 0.1)
    track = book.rows()[0]
    assert track.mean_conf == pytest.approx(0.90)
    assert track.mean_identity == pytest.approx(0.9933, abs=1e-3)
    assert track.has_identity


def test_the_box_number_is_named_when_it_is_shown_at_all():
    """Two bare numbers that disagree read as a bug -- the same battery saying
    1.00 on the plate and 0.91 in the readout. They measure different things,
    so when the box number is shown it is shown with its name on it.

    Shown only on request: it is stage 1's answer to "is there a label here",
    which is a question asked while tuning, not while parts go past."""
    plain = ld.apply_identities([{"name": "label", "conf": 0.91}],
                                [("sp_g31", 1.00)])
    assert plain[0]["label"] == "sp_g31 1.00"

    detailed = ld.apply_identities([{"name": "label", "conf": 0.91}],
                                   [("sp_g31", 1.00)], detail=True)
    assert detailed[0]["label"] == "sp_g31 1.00  box 0.91"


def test_the_readout_stays_one_id_and_one_held_average():
    """The pair is on the plate; repeating it here spends a line on something
    already in view. What stays is the number that moves."""
    book = ld.TrackBook()
    book.update([(1, "sp_g31", 0.91, 1.00)], now=100)
    assert ld.track_line(book.rows()[0]) == "sp_g31 0.91"
    book2 = ld.TrackBook()
    book2.update([(1, "label", 0.91)], now=100)
    assert ld.track_line(book2.rows()[0]) == "label 0.91"


def test_frames_stage_two_missed_do_not_drag_its_average_down():
    """A crop that failed contributes no identity rather than a zero: an
    average pulled to 0.50 by two missing frames reads as an unsure classifier
    when the classifier was never asked."""
    book = ld.TrackBook()
    book.update([(1, "sp_g31", 0.90, 1.00)], now=100)
    book.update([(1, "sp_g31", 0.90, None)], now=100.1)
    book.update([(1, "sp_g31", 0.90, None)], now=100.2)
    track = book.rows()[0]
    assert track.frames == 3
    assert track.mean_identity == pytest.approx(1.00)


@ui
def test_the_readout_and_the_plate_report_the_same_battery_the_same_way():
    """The end of the wire: stage 2's confidence has to reach the book, not
    just the drawn label. It was being computed and dropped."""
    win = _window()
    _label(win, "live_agree")
    win._live_thread = object()
    win._live_tracking = True
    win._live_tracks = ld.TrackBook()
    win._live_overlay_scale = (1.0, 1.0)
    try:
        items = _items([_Results([[10, 10, 40, 40]], {0: "label"}, [0.91], [0],
                                 ids=[7])])
        items = ld.apply_identities(items, [(win.label_id, 1.0)])
        win._on_live_result(items, 0.02)
        # The plate the operator reads carries the identity and how sure of it.
        assert items[0]["label"] == f"{win.label_id} 1.00"
        # And stage 2's confidence still reaches the book, which is what makes
        # a held identity average possible at all.
        assert win._live_tracks.rows()[0].mean_identity == pytest.approx(1.0)
        assert f"{win.label_id} 0.91" in win.live_readout.toPlainText()
    finally:
        win._live_thread = None


# --- proposals: keeping a frame with what the model found -------------------

def _item(name, conf=0.9, track_id=None, xyxy=None, points=None):
    item = {"name": name, "conf": conf}
    if track_id is not None:
        item["track_id"] = track_id
    if xyxy is not None:
        item["xyxy"] = xyxy
    if points is not None:
        item["points"] = points
    return item


def test_a_proposal_carries_the_id_the_detector_returned():
    boxes = ld.proposed_boxes([_item("2220-9199", xyxy=[0, 0, 4, 4])],
                              known_ids=["2220-9199"])
    assert boxes[0]["label"] == "2220-9199"
    assert boxes[0]["label_id"] == "2220-9199"


def test_an_axis_aligned_detection_becomes_four_corners():
    boxes = ld.proposed_boxes([_item("2220-9199", xyxy=[10, 20, 30, 50])])
    assert boxes[0]["points"] == [[10.0, 20.0], [30.0, 20.0],
                                  [30.0, 50.0], [10.0, 50.0]]


def test_an_oriented_detection_keeps_its_own_corners():
    pts = [[0, 0], [10, 2], [9, 12], [-1, 10]]
    boxes = ld.proposed_boxes([_item("spec_plate", points=pts)])
    assert boxes[0]["points"] == [[0.0, 0.0], [10.0, 2.0], [9.0, 12.0], [-1.0, 10.0]]


def test_confidence_and_track_id_survive_into_the_sidecar():
    boxes = ld.proposed_boxes(
        [_item("spec_plate", conf=0.83, track_id=4, xyxy=[0, 0, 4, 4])])
    assert boxes[0]["confidence"] == pytest.approx(0.83)
    assert boxes[0]["track_id"] == 4


def test_a_detection_with_no_geometry_is_dropped_rather_than_written_empty():
    assert ld.proposed_boxes([_item("spec_plate")]) == []
    assert ld.proposed_boxes([_item("", xyxy=[0, 0, 1, 1])]) == []


def test_every_proposed_box_is_marked_as_the_machines_work():
    boxes = ld.proposed_boxes([_item("2220-9199", xyxy=[0, 0, 4, 4])])
    assert boxes[0]["proposed_by"] == ld.PROPOSED_BY


def test_a_proposed_sidecar_is_never_review_marked():
    """The one thing that must not happen: a machine proposal that reads as an
    operator's approval and exports as truth."""
    from label_detections.core import review

    data = ld.proposed_annotation(
        "f.jpg", "sp",
        [_item("spec_plate", xyxy=[0, 0, 4, 4])], 640, 480)
    assert review.annotation_reviewed(data) is False
    assert review.annotation_status(data, "sp") == "needs_review"
    assert review.export_ready(review.annotation_status(data, "sp")) is False


def test_a_proposed_sidecar_records_the_frame_it_describes():
    data = ld.proposed_annotation(
        "f.jpg", "sp",
        [_item("spec_plate", xyxy=[0, 0, 4, 4])], 640, 480)
    assert data["image"] == "f.jpg"
    assert data["label_id"] == "sp"
    assert (data["width"], data["height"]) == (640, 480)
    assert data["proposed_by"] == ld.PROPOSED_BY


def test_one_live_run_is_one_capture_group_so_near_duplicates_do_not_split():
    """Frames from a single run are seconds apart under one lens. Letting them
    straddle train/val validates the model against its own training images."""
    from label_detections.core import dataset

    session = ld.proposal_session(0.0)
    entries = [
        dataset.entry_from_annotation(
            "sp", f"f{i}.jpg",
            ld.proposed_annotation(f"f{i}.jpg", "sp",
                                   [_item("spec_plate", xyxy=[0, 0, 4, 4])],
                                   session=session))
        for i in range(4)
    ]
    assert len({e.group_key() for e in entries}) == 1


def test_two_runs_are_two_groups_rather_than_one_lump():
    """A constant provenance string would collapse every live capture ever
    taken into a single group, which is the same bug wearing a hat."""
    from label_detections.core import dataset

    a = ld.proposal_session(0.0)
    b = ld.proposal_session(3600.0)
    assert a != b
    entries = [
        dataset.entry_from_annotation(
            "sp", f"{s}.jpg",
            ld.proposed_annotation(f"{s}.jpg", "sp",
                                   [_item("spec_plate", xyxy=[0, 0, 4, 4])],
                                   session=s))
        for s in (a, b)
    ]
    assert len({e.group_key() for e in entries}) == 2


# --- the two keep buttons ---------------------------------------------------

@ui
def test_keeping_image_only_writes_no_sidecar_at_all():
    """The whole point of the plain button: nothing pre-drawn to anchor on."""
    from label_detections.core import persistence, storage

    win = _window()
    label_id = _label(win, "live_plain")
    win._live_frame = np.zeros((60, 80, 3), np.uint8)
    win.keep_live_frame()
    name = sorted(p.name for p in storage.list_images(label_id))[-1]
    assert persistence.load_annotation(label_id, name) is None


@ui
def test_keeping_with_detections_writes_the_boxes_the_model_found():
    from label_detections.core import persistence, storage

    win = _window()
    label_id = _label(win, "live_json", )
    win._live_result_frame = np.zeros((60, 80, 3), np.uint8)
    win._live_result_items = [
        {"name": label_id, "conf": 0.91, "xyxy": [4.0, 6.0, 40.0, 30.0]}]
    win._live_session = "live_20260827_120000"
    win.keep_live_frame_with_detections()

    name = sorted(p.name for p in storage.list_images(label_id))[-1]
    data = persistence.load_annotation(label_id, name)
    assert data is not None
    assert len(data["boxes"]) == 1
    assert data["boxes"][0]["label"] == label_id
    assert data["boxes"][0]["label_id"] == label_id
    # The frame's own dimensions, not the preview's.
    assert (data["width"], data["height"]) == (80, 60)
    assert data["session"] == "live_20260827_120000"


@ui
def test_a_kept_proposal_still_has_to_be_reviewed_by_a_person():
    from label_detections.core import persistence, review, storage

    win = _window()
    label_id = _label(win, "live_json_unrev", )
    win._live_result_frame = np.zeros((60, 80, 3), np.uint8)
    win._live_result_items = [
        {"name": label_id, "conf": 0.91, "xyxy": [4.0, 6.0, 40.0, 30.0]}]
    win.keep_live_frame_with_detections()

    name = sorted(p.name for p in storage.list_images(label_id))[-1]
    data = persistence.load_annotation(label_id, name)
    assert review.annotation_status(data, label_id) == "needs_review"


@ui
def test_the_saved_image_is_the_one_the_boxes_were_computed_on():
    """Inference lags the preview by a frame or two. Save the newest image
    against the last result's boxes and every box is in the wrong place."""
    from label_detections.core import persistence, storage

    win = _window()
    label_id = _label(win, "live_pair", )

    inferred = np.zeros((60, 80, 3), np.uint8)      # what the model saw
    inferred[:] = 40
    win._live_result_frame = inferred
    win._live_result_items = [
        {"name": label_id, "conf": 0.9, "xyxy": [4.0, 6.0, 40.0, 30.0]}]
    win._live_frame = np.full((60, 80, 3), 200, np.uint8)   # newer, unmodelled

    win.keep_live_frame_with_detections()
    name = sorted(p.name for p in storage.list_images(label_id))[-1]
    saved = cv2.imread(str(storage.dataset_folder(label_id) / name))
    assert saved is not None
    assert int(saved.mean()) < 128, "saved the preview frame, not the inferred one"
    assert persistence.load_annotation(label_id, name)["boxes"]


@ui
def test_with_no_detections_it_keeps_the_image_instead_of_refusing():
    """The operator asked for this frame. An empty sidecar would read as
    'labeled, nothing on it', which is worse than no sidecar."""
    from label_detections.core import persistence, storage

    win = _window()
    label_id = _label(win, "live_empty", )
    win._live_result_frame = np.zeros((60, 80, 3), np.uint8)
    win._live_result_items = []
    before = len(storage.list_images(label_id))
    win.keep_live_frame_with_detections()
    assert len(storage.list_images(label_id)) == before + 1
    name = sorted(p.name for p in storage.list_images(label_id))[-1]
    assert persistence.load_annotation(label_id, name) is None


@ui
def test_both_buttons_are_actually_connected_to_their_handlers():
    """Clicked, not introspected: a button wired to nothing looks perfect
    from the outside and does nothing at all under a finger."""
    from label_detections.core import persistence, storage

    win = _window()
    label_id = _label(win, "live_clicks", )
    frame = np.zeros((60, 80, 3), np.uint8)
    win._live_frame = frame
    win._live_result_frame = frame
    win._live_result_items = [
        {"name": label_id, "conf": 0.9, "xyxy": [4.0, 6.0, 40.0, 30.0]}]

    win.live_keep_btn.click()
    plain = sorted(p.name for p in storage.list_images(label_id))[-1]
    assert persistence.load_annotation(label_id, plain) is None

    win.live_keep_json_btn.click()
    names = sorted(p.name for p in storage.list_images(label_id))
    assert len(names) == 2
    proposed = [n for n in names if n != plain][0]
    assert persistence.load_annotation(label_id, proposed)["boxes"]
    assert win.live_keep_btn.text() != win.live_keep_json_btn.text()


# --- stage 2: naming what the detector found --------------------------------

def test_an_unsure_classifier_says_unknown_rather_than_its_best_guess():
    """A classifier always returns something. Without a floor, a label it was
    never trained on comes back as whichever known label it least resembles --
    confidently. On a line counting ids against a recipe, a confident wrong id
    is worse than an honest blank."""
    assert ld.identify("2220-9199", 0.91)[0] == "2220-9199"
    assert ld.identify("2220-9199", 0.40)[0] == ld.UNKNOWN
    assert ld.identify("", 0.99)[0] == ld.UNKNOWN


def test_identities_land_on_the_boxes_they_belong_to():
    items = [{"name": "label", "track_id": 3, "conf": 0.9},
             {"name": "label", "track_id": 7, "conf": 0.8}]
    out = ld.apply_identities(items, [("2220-9199", 0.97), ("warn-g31-en", 0.88)])
    assert out[0]["name"] == "2220-9199" and out[1]["name"] == "warn-g31-en"
    # The plate is the id and how sure of it, nothing else -- the track id is
    # still on the item for the readout to group by, just not drawn on it.
    assert out[0]["label"] == "2220-9199 0.97"
    assert out[0]["track_id"] == 3
    # The detector's own answer is kept, so a disagreement stays visible.
    assert out[0]["detector_name"] == "label"


def test_a_length_mismatch_drops_identities_instead_of_shifting_them():
    """The failure has to degrade toward visibly incomplete, never toward
    invisibly false: a shifted identity puts a real label id on the wrong box,
    which nothing downstream can distinguish from a correct read."""
    items = [{"name": "label", "conf": 0.9}, {"name": "label", "conf": 0.8}]
    out = ld.apply_identities(items, [("2220-9199", 0.97)])
    assert [i["name"] for i in out] == ["label", "label"]


@ui
def test_stage_two_crops_in_the_same_order_the_overlay_draws():
    """The coupling the whole thing rests on. Identities are matched to boxes
    by position, so the worker's crop order must mirror the overlay's item
    order exactly -- for oriented and axis-aligned results alike."""
    from label_detections.ui.live_detect import InferenceWorker

    win = _window()
    worker = InferenceWorker("unused.pt", 640, 0.4, None, track=False)
    results = [_Results([[10, 10, 40, 40], [50, 50, 90, 90], [5, 60, 25, 80]],
                        {0: "label"}, [0.9, 0.8, 0.7], [0, 0, 0])]
    items, _ = win._detection_overlay_items(results)
    quads = worker._detection_quads(results)
    assert len(quads) == len(items) == 3
    for item, quad in zip(items, quads):
        x1, y1, x2, y2 = item["xyxy"]
        assert quad[0] == [x1, y1] and quad[2] == [x2, y2]


@ui
def test_the_readout_counts_label_ids_once_stage_two_has_spoken():
    win = _window()
    _label(win, "s2_readout")
    win._live_thread = object()
    win._live_tracking = False
    win._live_overlay_scale = (1.0, 1.0)
    try:
        win._on_live_result(
            ld.apply_identities(
                _items([_Results([[10, 10, 40, 40]], {0: "label"}, [0.9], [0])]),
                [("s2_readout", 0.96)]), 0.02)
        text = win.live_readout.toPlainText()
        assert "s2_readout: 1 found" in text
        assert "label:" not in text, "reported the detector's placeholder class"
    finally:
        win._live_thread = None


@ui
def test_without_a_classifier_the_boxes_keep_the_detectors_own_class():
    win = _window()
    _label(win, "s2_none")
    win._live_thread = object()
    win._live_tracking = False
    win._live_overlay_scale = (1.0, 1.0)
    try:
        win._on_live_result(
            _items([_Results([[10, 10, 40, 40]], {0: "label"}, [0.9], [0])]), 0.02)
        assert "label" in win.live_readout.toPlainText()
    finally:
        win._live_thread = None


@ui
def test_the_crop_size_comes_from_the_export_that_trained_the_classifier():
    """A classifier fed a size it did not train at loses accuracy silently, and
    the export wrote the answer next to the weights."""
    import tempfile
    from pathlib import Path

    win = _window()
    root = Path(tempfile.mkdtemp())
    (root / "train" / "a").mkdir(parents=True)
    (root / "classes.txt").write_text("a\n")
    cv2.imwrite(str(root / "train" / "a" / "c.jpg"), np.zeros((384, 384, 3), np.uint8))
    weights = root / "runs" / "w" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"not a real model")
    assert win._live_crop_px(str(weights)) == 384
    assert win._live_crop_px("") == 224


def test_a_generic_detector_does_not_invent_an_identity():
    """Under a localise-only detector the class is "label", which is not an
    identity. Stamping it would put label_id="label" on every box -- a value
    no recipe contains and no library row matches."""
    item = {"name": "label", "conf": 0.9, "xyxy": [0, 0, 4, 4]}
    box = ld.proposed_boxes([item], known_ids=["2220-9199"])[0]
    assert "label_id" not in box
    assert box["label"] == "label", "the detector's own answer still records"


def test_a_known_class_still_stamps_its_identity():
    item = {"name": "2220-9199", "conf": 0.9, "xyxy": [0, 0, 4, 4]}
    box = ld.proposed_boxes([item], known_ids=["2220-9199"])[0]
    assert box["label_id"] == "2220-9199"


def test_an_unknown_from_the_classifier_stamps_nothing():
    """Below its floor stage 2 returns UNKNOWN, which must not become an id."""
    item = {"name": ld.UNKNOWN, "conf": 0.4, "xyxy": [0, 0, 4, 4]}
    assert "label_id" not in ld.proposed_boxes([item], known_ids=["a"])[0]


def test_with_no_library_given_nothing_is_stamped():
    """A missing identity is visible; an invented one is not. Default to the
    visible failure."""
    item = {"name": "2220-9199", "conf": 0.9, "xyxy": [0, 0, 4, 4]}
    assert "label_id" not in ld.proposed_boxes([item])[0]


def test_a_classifier_run_never_lands_in_the_detector_field():
    """The bug that made live detect go silent. The routing test asked whether
    the path contained "classify" -- and the default run name is "classifier",
    which does not contain it. So the classifier went into the detector field,
    where a classification model returns probabilities and no boxes."""
    assert "classify" not in "classifier", "the substring test that failed"


@ui
def test_weights_are_routed_by_the_stage_that_made_them():
    from pathlib import Path

    win = _window()
    win.test_model_edit.setText("")
    win.live_classifier_edit.setText("")

    win._use_trained_as_active(Path("/runs/classifier/weights/best.pt"), "classifier")
    assert win.live_classifier_edit.text().endswith("best.pt")
    assert win.test_model_edit.text() == "", "a classifier reached the detector field"

    win._use_trained_as_active(Path("/runs/detector/weights/best.pt"), "detector")
    assert win.test_model_edit.text().endswith("best.pt")


def test_a_silent_view_says_why_rather_than_showing_nothing():
    """A live view finding nothing looks the same whether the camera is on a
    wall, the threshold is too high, or the wrong model is loaded.

    One line, though. Listing all four causes every time turned a quiet camera
    into the biggest thing in the pane; the full list is a hover away."""
    assert ld.quiet_hint(3, 0.45, 1024, False) == ""

    no_stage2 = ld.quiet_hint(ld.QUIET_FRAMES + 5, 0.45, 1024, False)
    assert "no stage 2 classifier" in no_stage2
    assert len(no_stage2.strip().splitlines()) == 1

    # With stage 2 configured, the threshold is the next thing to look at.
    assert "0.45" in ld.quiet_hint(20, 0.45, 1024, True)
    # And with a sane threshold, the size the model trained at.
    assert "1024" in ld.quiet_hint(20, 0.20, 1024, True)

    # Every cause is still written down, for the readout to carry as a tooltip.
    for cause in ("Confidence", "Image size", "detector", "stage 2"):
        assert cause in ld.QUIET_CAUSES


def test_the_start_floor_is_adjustable_rather_than_fixed():
    """It is a throughput ceiling -- 0.15 caps any hardware at 6.7/s -- but
    lowering it for everyone destabilised a camera that had been fine, because
    it also unblocked the display tick and drove the whole capture path nine
    times harder. So it is conservative by default and adjustable, instead of
    being chosen for the user by someone with no access to the hardware."""
    assert ld.MIN_INTERVAL_S > 0
    # should_infer takes it as an argument: the default is a default, not a law.
    assert ld.should_infer(False, 0.02, min_interval_s=0.01) is True
    assert ld.should_infer(False, 0.02, min_interval_s=0.5) is False


def test_a_throttled_rate_is_called_out_next_to_the_latency():
    """8 ms per inference alongside 6/s is not a slow model, it is a throttled
    one, and neither number says so alone."""
    r = ld.Rolling()
    for i in range(10):
        r.record(0.008, now=100 + i * 0.15)
    note = ld.throughput_note(r, interval_s=0.15)
    assert "could run" in note and "not the GPU" in note

    fast = ld.Rolling()
    for i in range(10):
        fast.record(0.008, now=100 + i * 0.009)
    assert ld.throughput_note(fast, interval_s=0.01) == "", \
        "no note when the rate matches latency"


def test_the_note_names_the_ceiling_actually_in_force():
    """It quoted MIN_INTERVAL_S, the module default, rather than the interval
    in force -- so with the rate set to 30/s it still reported a "150 ms start
    floor" that had not applied for some time. And it offered the camera and
    the floor as alternatives without saying which, when the numbers to tell
    them apart are right there."""
    r = ld.Rolling()
    for i in range(10):
        r.record(0.020, now=100 + i * 0.058)      # 20 ms model, ~17/s achieved

    # Rate raised well above the camera: the camera is the limit.
    camera_bound = ld.throughput_note(r, interval_s=1 / 30, camera_fps=17.0)
    assert "the camera, at 17/s" in camera_bound
    assert "150 ms" not in camera_bound

    # Rate left low: the floor is the limit, and it is quoted correctly.
    floor_bound = ld.throughput_note(r, interval_s=0.15, camera_fps=17.0)
    assert "150 ms start floor" in floor_bound
    assert "camera" not in floor_bound


def test_the_three_rates_are_reported_separately():
    """Camera, display and inference are three different numbers that were
    being read as one -- "6 fps" meant the third while sounding like the
    first."""
    r = ld.Rolling()
    for i in range(10):
        r.record(0.130, now=100 + i * 0.14)
    line = ld.rate_line(58.0, 30.0, r)
    assert "camera 30/s" in line
    assert "display 58/s" in line
    assert "inference 7.1/s" in line and "130 ms" in line


def test_slow_inference_on_cpu_says_so_and_stops_there():
    r = ld.Rolling()
    for i in range(5):
        r.record(0.130, now=100 + i * 0.14)
    cpu = ld.slow_hint(r, "Device: CPU -- torch reports NO CUDA.")
    assert "running on the CPU" in cpu
    assert "TensorRT" not in cpu, "no point suggesting TensorRT to a CPU"


def test_a_slow_model_is_told_it_is_the_model():
    """Only when the breakdown says so. This advice acts on the detector's own
    forward pass and on nothing else."""
    r = ld.Rolling()
    for i in range(5):
        r.record(0.130, now=100 + i * 0.14)
    gpu = ld.slow_hint(r, "Device: CUDA available (RTX 5090); model on cuda:0",
                       {"preprocess": 4.0, "inference": 120.0, "postprocess": 3.0})
    assert "TensorRT" in gpu and "image size" in gpu


def test_a_fast_detector_inside_a_slow_call_is_not_blamed():
    """The screenshot that prompted this: preprocess 3, inference 7,
    postprocess 2 -- 12 ms of a 71 ms call -- and the readout recited "check
    the image size, export to TensorRT" at the one part already fast."""
    r = ld.Rolling()
    for i in range(5):
        r.record(0.071, now=100 + i * 0.10)
    hint = ld.slow_hint(r, "model on cuda:0",
                        {"preprocess": 3.0, "inference": 7.0, "postprocess": 2.0,
                         "stage2": 50.0, "readout": 4.0})
    # It may still mention TensorRT for the *classifier* -- that is where the
    # time is. What must not appear is the detector-side advice.
    assert "Cost rises with the square" not in hint, "advised on the fast part"
    assert "Most of it is the model itself" not in hint
    assert "detector is only 12 ms" in hint
    assert "Stage 2 is 50 ms" in hint


def test_time_outside_every_phase_is_named():
    """A breakdown that adds to 12 next to a total of 71 is the most useful
    thing on the readout, and it used to be invisible."""
    r = ld.Rolling()
    for i in range(5):
        r.record(0.071, now=100 + i * 0.10)
    hint = ld.slow_hint(r, "model on cuda:0",
                        {"preprocess": 3.0, "inference": 7.0, "postprocess": 2.0,
                         "stage2": 1.0, "readout": 1.0})
    assert "outside every phase measured" in hint


def test_the_phase_line_shows_what_it_cannot_account_for():
    line = ld.phase_line(
        {"preprocess": 3.0, "inference": 7.0, "postprocess": 2.0,
         "stage2": 50.0, "readout": 4.0}, total_ms=71.0)
    assert "stage2 50" in line and "readout 4" in line
    assert "other 5" in line


def test_the_phase_line_says_nothing_extra_when_it_adds_up():
    line = ld.phase_line({"preprocess": 3.0, "inference": 7.0,
                          "postprocess": 2.0}, total_ms=12.0)
    assert "other" not in line


def test_a_fast_model_is_not_lectured():
    r = ld.Rolling()
    for i in range(5):
        r.record(0.006, now=100 + i * 0.01)
    assert ld.slow_hint(r, "Device: CUDA available") == ""


def test_a_bare_zero_becomes_cuda_zero_not_cpu_zero():
    """The inversion that put the model on the CPU while the field said 0:
    torch.Module.to(0) means CPU device 0, not GPU 0."""
    from label_detections.ui.live_detect import InferenceWorker

    def resolve(dev):
        w = InferenceWorker.__new__(InferenceWorker)
        w._device = dev
        return InferenceWorker._torch_device(w)

    assert resolve(0) == "cuda:0"
    assert resolve("0") == "cuda:0"
    assert resolve("cpu") == "cpu"
    assert resolve("cuda:1") == "cuda:1"
    assert resolve(None) == "" and resolve("") == ""


@ui
def test_one_device_field_covers_both_stages():
    """There is no separate classifier device on purpose -- two halves of one
    pipeline on two devices would pay a host round-trip per crop."""
    win = _window()
    tip = win.test_device_edit.toolTip()
    assert "BOTH stages" in tip
    assert "classifier" in tip


@ui
def test_both_model_paths_survive_a_relaunch():
    """They did not. editingFinished fires only when a human types into a
    field and leaves it, so a path from Browse or from "Use as active model"
    was never written -- the field looked right for the session and came back
    empty, which is why Live Detect had to be pointed at a model by hand every
    launch."""
    from label_detections.core.storage import load_test_settings

    win = _window()
    win.test_model_edit.setText("/runs/detector/weights/best.pt")
    win.live_classifier_edit.setText("/runs/classifier/weights/best.pt")

    saved = load_test_settings() or {}
    assert saved.get("model") == "/runs/detector/weights/best.pt"
    assert saved.get("classifier") == "/runs/classifier/weights/best.pt"


@ui
def test_start_uses_the_field_without_a_test_run_first(monkeypatch):
    """Starting had to be preceded by a run on the Test Models tab. That was
    the unsaved path, not a load order -- with a path present, Start builds the
    worker straight away."""
    win = _window()
    win.test_model_edit.setText("/nonexistent/detector.pt")
    win.live_classifier_edit.setText("")

    built = {}

    class FakeWorker:
        loaded = failed = result = None

        def __init__(self, path, *a, **k):
            built["path"] = path
            raise RuntimeError("stop before threading")

    import label_detections.ui.live_detect as ld_ui
    monkeypatch.setattr(ld_ui, "InferenceWorker", FakeWorker)
    monkeypatch.setattr(win, "_camera_is_live", lambda: True)
    try:
        win.start_live_detect()
    except RuntimeError:
        pass
    finally:
        win._live_thread = None
    assert built.get("path") == "/nonexistent/detector.pt"


# --- the two thresholds -----------------------------------------------------
#
# There are two, they gate different stages, and only one of them was on
# screen. The visible one -- Confidence, on the Test Models tab -- is stage 1:
# how sure the detector is that there is a label there. Stage 2's, how sure the
# classifier is about which label it is, sat at 0.55 in the source with nothing
# naming it, so raising the visible one to 0.90 changed a stage nobody meant to
# change and left the other exactly where it was.

@ui
def test_the_stage_two_floor_reaches_the_worker(monkeypatch):
    """It was never passed at all: the worker took an identity_floor argument
    and the one caller never supplied one, so every run used the default."""
    win = _window()
    win.test_model_edit.setText("/nonexistent/detector.pt")
    win.live_classifier_edit.setText("/nonexistent/classifier.pt")
    win.live_floor_spin.setValue(0.80)

    got = {}

    class FakeWorker:
        loaded = failed = result = None

        def __init__(self, *a, **k):
            got.update(k)
            raise RuntimeError("stop before threading")

    import label_detections.ui.live_detect as ld_ui
    monkeypatch.setattr(ld_ui, "InferenceWorker", FakeWorker)
    monkeypatch.setattr(win, "_camera_is_live", lambda: True)
    try:
        win.start_live_detect()
    except RuntimeError:
        pass
    finally:
        win._live_thread = None
    assert got.get("identity_floor") == pytest.approx(0.80)


@ui
def test_the_two_thresholds_are_separate_settings():
    """Moving one must not move the other: they gate different stages, and the
    whole reason this is on screen is that they were being confused."""
    win = _window()
    win.test_conf_spin.setValue(0.90)
    win.live_floor_spin.setValue(0.55)
    assert win.test_conf_spin.value() == pytest.approx(0.90)
    assert win.live_floor_spin.value() == pytest.approx(0.55)


@ui
def test_the_stage_two_floor_is_remembered_between_launches():
    win = _window()
    win.live_floor_spin.setValue(0.75)
    win._save_test_settings()
    from label_detections.core.storage import load_test_settings
    assert load_test_settings().get("identity_floor") == pytest.approx(0.75)


@ui
def test_the_floor_starts_where_it_has_always_silently_been():
    """Exposing a hidden setting must not also change it. Every install that
    upgrades into this has no saved value, and it has to keep behaving exactly
    as it did rather than quietly moving to whatever looks like a nice default.

    A fresh window rather than the shared one, and the key removed first: this
    is about what happens with nothing saved, which the shared window stopped
    being able to show the moment another test saved something.
    """
    from label_detections.core.storage import load_test_settings, save_test_settings
    from label_detections.ui.main_window import MainWindow

    settings = dict(load_test_settings() or {})
    settings.pop("identity_floor", None)
    save_test_settings(settings)
    assert MainWindow().live_floor_spin.value() == pytest.approx(
        ld.DEFAULT_IDENTITY_FLOOR)


@ui
def test_the_live_tab_names_the_detector_it_will_use():
    """A required setting on another tab, with nothing on this one naming it,
    reads as Start being broken rather than as a field being empty."""
    win = _window()
    win.test_model_edit.setText("/runs/detector/weights/best.pt")
    assert "best.pt" in win.live_detector_label.text()

    win.test_model_edit.setText("")
    assert "NOT SET" in win.live_detector_label.text()
    assert "Test Models" in win.live_detector_label.text()


@ui
def test_stage_two_handles_an_obb_detector():
    """The pipeline these settings actually produce is OBB, and stage 2 had
    only ever been exercised on axis-aligned boxes."""
    from label_detections.ui.live_detect import InferenceWorker

    class _OBB:
        xyxyxyxy = np.array([[[100, 200], [400, 210], [395, 330], [95, 320]]],
                            np.float32)

    class OBBResult:
        names = {0: "label"}
        obb = _OBB()
        boxes = None

    worker = InferenceWorker("d.pt", 640, 0.25, 0, track=True, crop_px=224)
    quads = worker._detection_quads([OBBResult()])
    assert len(quads) == 1, "an OBB detection produced no crop"
    assert quads[0][0] == [100.0, 200.0]


@ui
def test_a_plain_array_attribute_is_not_silently_dropped():
    """Demanding .cpu() means anything already array-like returns [] -- which
    does not error, it identifies nothing, forever. The same bug was fixed on
    the overlay path and then rewritten here."""
    from label_detections.ui.live_detect import InferenceWorker

    class Boxes:
        xyxy = np.array([[10.0, 20.0, 30.0, 40.0]], np.float32)   # no .cpu()

    class R:
        names = {0: "label"}
        obb = None
        boxes = Boxes()

    worker = InferenceWorker("d.pt", 640, 0.25, 0, track=False)
    assert len(worker._detection_quads([R()])) == 1


@ui
def test_a_failing_stage_two_does_not_take_stage_one_with_it():
    """It was called bare, so one exception propagated out of the slot, Qt
    swallowed it, and no result was emitted at all -- a loaded model sitting on
    its placeholder with nothing to say."""
    from label_detections.ui.live_detect import InferenceWorker

    class Boxes:
        xyxy = np.array([[10.0, 20.0, 30.0, 40.0]], np.float32)
        conf = np.array([0.9], np.float32)
        cls = np.array([0.0], np.float32)
        id = None

    class R:
        names = {0: "label"}
        obb = None
        boxes = Boxes()

    class Exploding:
        def predict(self, *a, **k):
            raise RuntimeError("classifier blew up")

    worker = InferenceWorker("d.pt", 640, 0.25, 0, track=False)
    worker._model = type("M", (), {"predict": lambda s, f, **k: [R()]})()
    worker._classifier = Exploding()

    emitted = []
    worker.result.connect(lambda *a: emitted.append(a))
    worker.infer(np.zeros((100, 200, 3), np.uint8))
    assert emitted, "stage 1 result was lost to a stage 2 failure"
    items = emitted[0][0]
    assert len(items) == 1, "the detection itself must survive"
    # No identity attached, which is the correct degradation.
    assert items[0]["name"] == "label"
    assert "identity_conf" not in items[0]


@ui
def test_no_frame_is_pumped_before_the_model_has_loaded():
    """The wedge. infer() returns immediately when the model is not loaded yet,
    and nothing on that path clears the busy flag -- so the one frame that
    arrived during loading blocked every frame after it, forever. Moving the
    model to CUDA at load made loading slow enough that this race went from
    usually won to always lost."""
    win = _window()
    handed = []
    _watch_infer(win, handed)
    win._live_thread = object()
    win._live_loaded = False
    win._live_busy = False
    win._live_last_started = 0.0
    try:
        win._pump_live_detect(np.zeros((10, 10, 3), np.uint8))
        assert handed == [], "handed a frame to a model that does not exist yet"
        assert win._live_busy is False, "busy latched with nothing able to clear it"

        win._live_loaded = True
        win._pump_live_detect(np.zeros((10, 10, 3), np.uint8))
        assert len(handed) == 1
    finally:
        win._live_thread = None
        win._live_busy = False
        win._live_loaded = False


@ui
def test_a_result_that_never_arrives_does_not_freeze_the_view():
    """Any lost result -- an exception Qt swallowed, a signal dropped on
    shutdown -- would otherwise leave the view frozen with no error and no way
    back but a restart."""
    win = _window()
    handed = []
    _watch_infer(win, handed)
    win._live_thread = object()
    win._live_loaded = True
    win._live_busy = True
    win._live_last_started = time.monotonic() - (ld.BUSY_TIMEOUT_S + 1)
    try:
        win._pump_live_detect(np.zeros((10, 10, 3), np.uint8))
        assert len(handed) == 1, "never recovered from a lost result"
    finally:
        win._live_thread = None
        win._live_busy = False
        win._live_loaded = False


@ui
def test_the_stage_two_size_is_settable_and_persisted():
    """It was only guessable -- by looking for the export's classes.txt near
    the weights. Weights live under runs/, the dataset does not, so the guess
    failed and fell back to 224 with nothing said."""
    from label_detections.core.storage import load_test_settings

    win = _window()
    win.live_crop_spin.setValue(320)
    assert (load_test_settings() or {}).get("crop_px") == 320


@ui
def test_the_size_is_only_auto_filled_when_the_crops_are_really_there():
    """Overwriting a hand-set value with a fallback guess is how the wrong size
    gets in without anyone choosing it."""
    import tempfile
    from pathlib import Path

    win = _window()
    win.live_crop_spin.setValue(320)

    # Weights with no dataset beside them: leave the operator's value alone.
    orphan = Path(tempfile.mkdtemp()) / "runs" / "classify" / "weights"
    orphan.mkdir(parents=True)
    (orphan / "best.pt").write_bytes(b"x")
    assert win._live_crop_px_or_none(str(orphan / "best.pt")) is None
    win._sync_live_crop_size(str(orphan / "best.pt"))
    assert win.live_crop_spin.value() == 320, "a guess overwrote a real setting"

    # Weights inside an export: take the real size.
    root = Path(tempfile.mkdtemp())
    (root / "train" / "a").mkdir(parents=True)
    (root / "classes.txt").write_text("a\n")
    cv2.imwrite(str(root / "train" / "a" / "c.jpg"), np.zeros((448, 448, 3), np.uint8))
    inside = root / "w"
    inside.mkdir()
    (inside / "best.pt").write_bytes(b"x")
    win._sync_live_crop_size(str(inside / "best.pt"))
    assert win.live_crop_spin.value() == 448


@ui
def test_no_tensor_shaped_object_crosses_back_to_the_gui():
    """The crash was on the first inference. The worker emitted the Ultralytics
    Results object, so CUDA tensors travelled a Qt queued signal and were
    dereferenced on the thread that paints the window -- GPU work on the GUI
    thread, and torch objects crossing a boundary they were never promised to
    cross."""
    import inspect
    from label_detections.ui.live_detect import InferenceWorker, extract_items

    src = inspect.getsource(InferenceWorker.infer)
    assert "self.result.emit(items," in src, "still emitting the raw results"
    assert "extract_items(results)" in src, "conversion must happen worker-side"

    items = extract_items([_Results([[1, 2, 3, 4]], {0: "label"}, [0.9], [0])])
    assert items and isinstance(items[0]["conf"], float)
    for value in items[0].values():
        assert not hasattr(value, "cpu"), "a tensor survived extraction"


def test_the_crash_handler_is_armed_at_startup():
    """A native crash ends the process with no Python traceback, which from
    outside is indistinguishable from a clean exit -- "it just closes" was the
    whole of what could be reported."""
    from pathlib import Path

    src = Path("main.py").read_text()
    assert "faulthandler.enable()" in src
    assert "all_threads=True" in src, "the crashing thread may not be the main one"
    assert "labelvision_crash.log" in src, "a console from a shortcut disappears"


@ui
def test_the_inference_rate_is_a_setting_and_governs_the_pump():
    """Raising it drives the whole capture path harder, not just the GPU: with
    inference off the GUI thread, the display tick is free to come round again
    as soon as it is dispatched. 6.7 -> 100 took that tick from ~7 to ~60 a
    second and destabilised a camera that had been fine."""
    win = _window()
    handed = []
    _watch_infer(win, handed)
    win._live_thread = object()
    win._live_loaded = True
    win._live_busy = False
    try:
        win.live_rate_spin.setValue(2.0)          # one inference per 500 ms
        win._live_last_started = time.monotonic() - 0.1
        win._pump_live_detect(np.zeros((8, 8, 3), np.uint8))
        assert handed == [], "ignored the configured rate"

        win._live_last_started = time.monotonic() - 0.6
        win._pump_live_detect(np.zeros((8, 8, 3), np.uint8))
        assert len(handed) == 1
    finally:
        win.live_rate_spin.setValue(1.0 / ld.MIN_INTERVAL_S)   # shared window
        win._live_thread = None
        win._live_busy = False
        win._live_loaded = False


def test_the_default_rate_is_the_one_that_ran_stably():
    """Conservative by default. Raising it drives the capture path harder, so
    it is a step someone takes and can walk back -- not one taken for them."""
    assert abs(1.0 / ld.MIN_INTERVAL_S - 6.7) < 0.2


def test_the_phase_breakdown_separates_the_gpu_from_the_resize():
    """A single latency figure cannot tell a slow model from a slow resize in
    front of it, and on a 20 MP source those look identical from outside."""
    slow_cpu_work = ld.phase_line({"preprocess": 95.0, "inference": 8.0,
                                   "postprocess": 3.0})
    assert "preprocess 95" in slow_cpu_work
    assert "not the model" in slow_cpu_work
    assert "TensorRT" not in slow_cpu_work, "wrong advice for a preprocess cost"

    slow_model = ld.phase_line({"preprocess": 4.0, "inference": 110.0,
                                "postprocess": 6.0})
    assert "Most of it is the model" in slow_model
    assert "TensorRT" in slow_model

    assert ld.phase_line({}) == ""


@ui
def test_the_predictor_device_is_checked_not_assumed():
    """Ultralytics builds its own predictor with its own device on the first
    call. The model object reporting cuda:0 says nothing about where the
    predictor ended up -- and 120 ms for an OBB at 640 is CPU-speed."""
    import inspect
    from label_detections.ui.live_detect import InferenceWorker

    body = inspect.getsource(InferenceWorker.infer)
    assert 'getattr(self._model, "predictor", None)' in body
    assert "_device_checked" in body, "must report once, not every frame"


def test_tracking_uses_bytetrack_not_the_default_botsort():
    """Turning tracking on took inference from single-digit milliseconds to
    120, and all of it was BoT-SORT's global motion compensation: sparse
    optical flow over the whole frame, every frame, to cancel camera movement.
    On a 20 MP image that costs more than the model. The camera is bolted
    down, so it cancels nothing."""
    import inspect
    from label_detections.ui.live_detect import InferenceWorker

    worker = InferenceWorker("d.pt", 640, 0.25, 0, track=True)
    assert worker._tracker == "bytetrack.yaml"
    assert "tracker=self._tracker" in inspect.getsource(InferenceWorker.infer)


def test_the_tracker_stays_overridable():
    """A camera that does move would want BoT-SORT back."""
    from label_detections.ui.live_detect import InferenceWorker

    worker = InferenceWorker("d.pt", 640, 0.25, 0, track=True,
                             tracker="botsort.yaml")
    assert worker._tracker == "botsort.yaml"


def test_the_readout_reports_what_the_gui_thread_itself_costs():
    """Inference runs on the GUI thread, so the window cannot repaint or
    respond while it is in there. That cost was invisible -- the readout showed
    the model's time and the frame rates, neither of which says the UI is
    blocked."""
    r = ld.Rolling()
    r.record(0.005, now=100)
    assert "gui 40 ms" in ld.rate_line(17.0, 17.0, r, gui_ms=40.0)
    assert "gui" not in ld.rate_line(17.0, 17.0, r)


@ui
def test_a_threaded_camera_frame_is_not_copied_twice():
    """CameraSource.read() already copies out of _latest_frame under the lock,
    so copying again is a second 60 MB memcpy of the same pixels on the thread
    that draws the preview -- more time than the model now takes."""
    win = _window()
    handed = []
    _watch_infer(win, handed)
    win._live_thread = object()
    win._live_loaded = True
    win._live_busy = False
    win._live_last_started = 0.0

    class Threaded:
        threaded = True

    real_camera = win.camera
    win.camera = Threaded()
    frame = np.zeros((8, 8, 3), np.uint8)
    try:
        win._pump_live_detect(frame)
        assert handed and handed[0] is frame, "copied a frame that was already private"
    finally:
        win.camera = real_camera
        win._live_thread = None
        win._live_busy = False
        win._live_loaded = False


@ui
def test_starting_a_run_clears_the_previous_run_s_boxes(monkeypatch):
    """Model loading takes seconds, and boxes from the last run sitting there
    through it look exactly like boxes from this one."""
    win = _window()
    win.test_model_edit.setText("/nonexistent/detector.pt")
    win.canvas.set_model_test_overlays(
        [{"points": [[0, 0], [9, 0], [9, 9], [0, 9]], "name": "stale", "conf": 0.9}])

    class FakeWorker:
        def __init__(self, *a, **k):
            raise RuntimeError("stop before threading")

    import label_detections.ui.live_detect as ld_ui
    monkeypatch.setattr(ld_ui, "InferenceWorker", FakeWorker)
    monkeypatch.setattr(win, "_camera_is_live", lambda: True)
    try:
        win.start_live_detect()
    except RuntimeError:
        pass
    finally:
        win._live_thread = None
    assert win.canvas.model_test_overlays == []


def test_the_plate_carries_the_id_and_the_confidences_and_nothing_else():
    """Read at a glance while parts move past: every extra character on the box
    competes with the ones that matter. The track id stays on the item, because
    the readout groups by it -- it is not drawn."""
    results = [_Results([[10, 10, 40, 40]], {0: "2220-9199"}, [0.91], [0], ids=[5])]
    items = _items(results)
    # Detector-only: one number, and no `box` prefix naming a distinction that
    # does not exist yet.
    assert items[0]["label"] == "2220-9199 0.91"
    assert items[0]["track_id"] == 5

    identified = ld.apply_identities(items, [("ODX-Long", 0.87)])
    assert identified[0]["label"] == "ODX-Long 0.87"
    assert identified[0]["track_id"] == 5
    assert "#" not in identified[0]["label"] and "5" not in identified[0]["label"]


# --- the readout is readable while it updates -------------------------------

@contextlib.contextmanager
def _laid_out_readout(win):
    """Swap in a readout Qt will actually lay out, and hand back its scrollbar.

    The real one lives inside a tab of a window that is never shown offscreen,
    and Qt gives an unshown widget no scroll range at all -- so the thing under
    test cannot be exercised in place.
    """
    from PySide6.QtWidgets import QTextEdit

    original = win.live_readout
    stand_in = QTextEdit()
    stand_in.setReadOnly(True)
    stand_in.setFixedSize(300, 60)
    stand_in.show()
    QApplication.processEvents()
    win.live_readout = stand_in
    try:
        yield stand_in.verticalScrollBar()
    finally:
        win.live_readout = original
        stand_in.deleteLater()

@ui
def test_scrolling_the_readout_survives_the_next_update():
    """It was rewritten three times per result, each rebuilding the document
    and resetting the scroll position -- thirty times a second, so the
    scrollbar could be dragged but never stayed anywhere."""
    win = _window()
    with _laid_out_readout(win) as bar:
        win._set_live_readout("\n".join(f"line {i}" for i in range(80)))
        QApplication.processEvents()
        assert bar.maximum() > 0, "the readout must be scrollable to test this"

        bar.setValue(20)
        win._set_live_readout("\n".join(f"line {i} updated" for i in range(80)))
        assert bar.value() == 20, "the update yanked the view back"


@ui
def test_a_readout_at_the_bottom_keeps_following():
    """Someone parked at the bottom is watching the newest line, not a fixed
    offset."""
    win = _window()
    with _laid_out_readout(win) as bar:
        win._set_live_readout("\n".join(f"line {i}" for i in range(80)))
        QApplication.processEvents()
        bar.setValue(bar.maximum())
        win._set_live_readout("\n".join(f"line {i}" for i in range(120)))
        assert bar.value() == bar.maximum()


@ui
def test_identical_text_is_not_rewritten():
    """Rebuilding a document to produce the same characters costs the GUI
    thread real time ten times a second."""
    win = _window()
    win._set_live_readout("steady")
    calls = []
    original = win.live_readout.setPlainText
    win.live_readout.setPlainText = lambda t: (calls.append(t), original(t))[1]
    try:
        win._set_live_readout("steady")
        assert calls == []
        win._set_live_readout("changed")
        assert calls == ["changed"]
    finally:
        win.live_readout.setPlainText = original


# --- pacing the display tick ------------------------------------------------

def test_the_tick_follows_the_camera_rather_than_a_fixed_rate():
    """A fixed 16 ms was ~60 firings a second whatever the camera did. That was
    invisible only because inference sat in the tick and made it impossible."""
    assert ld.tick_interval_ms(17.0) == 39      # ~25 ticks/s for a 17/s camera
    assert ld.tick_interval_ms(30.0) == 22      # ~45 ticks/s for a 30/s camera


def test_the_tick_oversamples_so_frames_are_not_shown_late():
    """A tick exactly at the frame interval drifts in and out of phase."""
    for fps in (10.0, 17.0, 30.0, 60.0):
        interval = ld.tick_interval_ms(fps)
        assert interval < 1000.0 / fps, f"{fps}/s tick is not faster than the camera"


def test_an_unknown_rate_keeps_the_interval_it_always_had():
    """True for the first second of every session."""
    assert ld.tick_interval_ms(0.0) == ld.TICK_DEFAULT_MS
    assert ld.tick_interval_ms(-1.0) == ld.TICK_DEFAULT_MS
    assert ld.tick_interval_ms(None) == ld.TICK_DEFAULT_MS


def test_the_interval_is_clamped_at_both_ends():
    """A very fast camera must not spin the GUI, and a very slow one must not
    leave the preview feeling dead."""
    assert ld.tick_interval_ms(500.0) == ld.TICK_FASTEST_MS
    assert ld.tick_interval_ms(0.5) == ld.TICK_SLOWEST_MS


@ui
def test_an_unthreaded_tick_does_not_read_faster_than_the_camera():
    """The hole this closes. An unthreaded backend has no frame counter, so
    every firing reached read() -- which blocks the GUI thread until the camera
    produces something. Invisible while inference sat in the tick and made 60
    firings a second impossible."""
    win = _window()
    reads = []

    class Unthreaded:
        threaded = False

        def is_open(self):
            return True

        def read(self):
            reads.append(time.perf_counter())
            return True, np.zeros((8, 8, 3), np.uint8)

        def read_fps(self):
            return 17.0

        def frame_seq(self):
            return 0

        def drain(self, count=2):
            pass

    real = win.camera
    try:
        win.camera = Unthreaded()
        win._tick_fps = 17.0
        win._last_tick_worked = 0.0
        win.last_raw = None
        win._on_timer()                      # first tick always works
        assert len(reads) == 1
        # Immediately again: the camera cannot have a new frame yet.
        win._on_timer()
        win._on_timer()
        assert len(reads) == 1, "read the camera again inside one frame interval"
        # Once the interval has passed, it works again.
        win._last_tick_worked -= win._tick_floor_s()
        win._on_timer()
        assert len(reads) == 2
    finally:
        win.camera = real
        win.last_raw = None
        win._last_tick_worked = 0.0


@ui
def test_a_threaded_tick_still_paces_on_the_frame_counter():
    """Exact beats estimated: where a counter exists it decides, so a camera
    running faster than the last measurement cannot have frames skipped."""
    win = _window()
    reads = []

    class Threaded:
        threaded = True

        def __init__(self):
            self.seq = 1

        def is_open(self):
            return True

        def read(self):
            reads.append(1)
            return True, np.zeros((8, 8, 3), np.uint8)

        def read_fps(self):
            return 17.0

        def frame_seq(self):
            return self.seq

        def drain(self, count=2):
            pass

    real = win.camera
    cam = Threaded()
    try:
        win.camera = cam
        win.last_raw = None
        win._last_frame_seq = None
        win._on_timer()
        assert len(reads) == 1
        win._on_timer()                      # same frame
        assert len(reads) == 1
        # A new frame arrives inside the estimated interval -- it must still be
        # taken, because the counter says it is genuinely new.
        cam.seq = 2
        win._last_tick_worked = time.perf_counter()
        win._on_timer()
        assert len(reads) == 2, "a real new frame was skipped by the time floor"
    finally:
        win.camera = real
        win.last_raw = None
        win._last_frame_seq = None


@ui
def test_the_timer_retunes_to_the_camera_that_showed_up():
    win = _window()
    win.timer.start(16)
    try:
        win._retune_tick(30.0)
        assert win.timer.interval() == ld.tick_interval_ms(30.0) == 22
        win._retune_tick(17.0)
        assert win.timer.interval() == 39
        # Unknown rate falls back rather than leaving the last camera's pacing.
        win._retune_tick(0.0)
        assert win.timer.interval() == ld.TICK_DEFAULT_MS
    finally:
        win.timer.stop()
# --- inference is off the display tick --------------------------------------

@ui
def test_the_worker_runs_on_a_plain_thread_with_no_qt_event_loop():
    """The whole point of the rework. QThread emits started() before exec(),
    so load() and its warm-up ran outside the event loop and their CUDA calls
    completed, while infer() arrived as a queued slot inside exec() -- where
    QEventDispatcherWin32 owns a hidden window and pumps Windows messages --
    and every CUDA call from there deadlocked on a WDDM GPU.

    A plain thread has no pump and no window, which is why the camera reader
    beside it has never had the problem."""
    from pathlib import Path

    from label_detections.ui.live_detect import InferenceWorker

    source = (Path(__file__).resolve().parent.parent / "label_detections" /
              "ui" / "main_window.py").read_text()
    assert "QThread(self)" not in source, "inference is back on a QThread"
    assert "threading.Thread(\n            target=self._live_worker.run_forever" in source

    # And the worker offers the loop the thread runs, not a slot Qt delivers.
    assert hasattr(InferenceWorker, "run_forever")
    assert not hasattr(InferenceWorker.submit, "__slots__")


@ui
def test_the_queue_keeps_the_newest_frame_not_the_stalest():
    """One deep. If a frame is already waiting, showing the older one would
    put stale boxes on a live view."""
    from label_detections.ui.live_detect import InferenceWorker

    worker = InferenceWorker("d.pt", 640, 0.4, None, track=False)
    worker.submit("first")
    worker.submit("second")
    assert worker._queue.qsize() == 1
    assert worker._queue.get_nowait() == "second"


@ui
def test_submitting_never_blocks_the_caller():
    """It is called from the display tick; blocking there is the thing this
    whole change exists to stop."""
    from label_detections.ui.live_detect import InferenceWorker

    worker = InferenceWorker("d.pt", 640, 0.4, None, track=False)
    started = time.perf_counter()
    for i in range(200):
        worker.submit(i)
    assert (time.perf_counter() - started) < 0.5


@ui
def test_the_result_uses_the_frame_and_scale_it_was_submitted_with():
    """With inference on its own thread the live view has usually moved on by
    the time a result lands. Scaling by the current factor, or pairing the
    boxes with the current frame, is silently wrong in both directions."""
    win = _window()
    _label(win, "inflight")
    win._live_thread = object()
    win._live_tracking = False
    submitted = np.zeros((8, 8, 3), np.uint8)
    win._inflight_frame = submitted
    win._inflight_scale = (0.25, 0.25)
    # The live view has already moved on to a different frame and scale.
    win._live_frame = np.ones((8, 8, 3), np.uint8)
    win._live_overlay_scale = (1.0, 1.0)
    try:
        win._on_live_result(
            [{"type": "other_box", "xyxy": [100.0, 200.0, 300.0, 400.0],
              "cx": 200.0, "cy": 300.0, "name": "inflight", "conf": 0.9,
              "points": [[100.0, 200.0], [300.0, 200.0],
                         [300.0, 400.0], [100.0, 400.0]]}],
            0.02, {})
        assert win._live_result_frame is submitted, "paired with a newer frame"
        drawn = win.canvas.model_test_overlays[0]
        assert drawn["xyxy"] == [25.0, 50.0, 75.0, 100.0], (
            "scaled by the live view's factor, not the submitted one")
    finally:
        win._live_thread = None


@ui
def test_stopping_wakes_the_worker_out_of_its_wait():
    """The loop blocks on a queue. Without something to wake it, Stop would
    wait out the timeout on every stop."""
    from label_detections.ui.live_detect import InferenceWorker

    worker = InferenceWorker("d.pt", 640, 0.4, None, track=False)
    worker.stop()
    assert worker._stopping is True
    # A sentinel is waiting, so the loop returns at once rather than sitting
    # out its poll interval.
    assert worker._queue.get_nowait() is None


# --- the predictor is built at load, not on the first frame -----------------

@ui
def test_loading_builds_the_predictor_rather_than_leaving_it_to_frame_one():
    """A py-spy dump of a frozen session showed the worker stopped inside
    torch.cuda.set_device, reached through Ultralytics' setup_model on the
    first track() call. Called directly that ran on the GUI thread and was
    merely a slow first frame; moved to the worker thread it deadlocked."""
    from label_detections.ui.live_detect import InferenceWorker

    calls = []

    class FakeModel:
        task = "obb"

        def to(self, target):
            calls.append(("to", target))
            return self

        def track(self, frame, **kw):
            calls.append(("track", frame.shape, kw.get("persist")))
            return []

        def predict(self, frame, **kw):
            calls.append(("predict", getattr(frame, "shape", None)))
            return []

    worker = InferenceWorker("d.pt", 640, 0.4, None, track=True,
                             warm_shape=(3672, 5496, 3))
    worker._model = FakeModel()
    worker._warm_up()

    assert calls, "load left the predictor to be built by the first real frame"
    kind, shape, persist = calls[0]
    assert kind == "track", "warmed through a different call than the live path uses"
    # The camera's real frame size and the live path's persist, so that a call
    # which works here and hangs there differs in nothing that was chosen.
    assert shape == (3672, 5496, 3), "warmed on a convenient small square"
    assert persist is True, "warmed with a different persist than the live call"


@ui
def test_the_warmup_falls_back_to_a_square_when_no_frame_has_arrived():
    from label_detections.ui.live_detect import InferenceWorker

    shapes = []

    class FakeModel:
        task = "obb"

        def predict(self, frame, **kw):
            shapes.append(frame.shape)
            return []

    worker = InferenceWorker("d.pt", 640, 0.4, None, track=False)
    worker._model = FakeModel()
    worker._warm_up()
    assert shapes == [(640, 640, 3)]


@ui
def test_a_warmup_failure_is_reported_and_does_not_take_the_load_down():
    from label_detections.ui.live_detect import InferenceWorker

    class Broken:
        task = "obb"

        def predict(self, *a, **k):
            raise RuntimeError("no CUDA kernels for this card")

    worker = InferenceWorker("d.pt", 640, 0.4, None, track=False)
    worker._model = Broken()
    said = []
    worker.failed.connect(said.append)
    worker._warm_up()                     # must not raise
    assert said and "first run failed" in said[0]
    assert "no CUDA kernels" in said[0]


# --- the frame handed on must not be SDK memory -----------------------------

def test_a_basler_frame_is_copied_out_before_the_sdk_takes_it_back():
    """pypylon's GetArray() can be a view over the converted image's buffer,
    and that image was a temporary here -- released the moment the expression
    ended, with grab.Release() right behind it. What the UI got was a window
    onto memory the SDK had taken back, which is consistent with the 0xc0000374
    heap corruption this application saw and never explained."""
    import numpy as np

    from label_detections.core.camera import CameraSource

    sdk_buffer = np.zeros((4, 4, 3), dtype=np.uint8)
    released = {"done": False}

    class Converted:
        def GetArray(self):
            return sdk_buffer            # a VIEW, as pypylon may return

    class Converter:
        def Convert(self, _grab):
            return Converted()

    class Grab:
        def GrabSucceeded(self):
            return True

        def Release(self):
            released["done"] = True
            sdk_buffer[:] = 0xEF         # the SDK reuses the buffer

    class Cap:
        def IsGrabbing(self):
            return True

        def RetrieveResult(self, *a, **k):
            return Grab()

    cam = CameraSource()
    cam.cap = Cap()
    cam.converter = Converter()

    import label_detections.core.camera as camera_mod
    real_pylon = camera_mod.pylon
    camera_mod.pylon = type("P", (), {"TimeoutHandling_Return": 0})()
    try:
        ok, frame = cam._read_basler_frame(timeout_ms=10)
    finally:
        camera_mod.pylon = real_pylon

    assert ok and frame is not None
    assert released["done"], "the test did not exercise the release"
    # The frame must be untouched by what happened to the SDK's buffer.
    assert frame.max() == 0, "the frame is a view into memory the SDK reclaimed"
    assert not np.shares_memory(frame, sdk_buffer)


# --- the profiler's synchronize is the frame that hangs ---------------------

@ui
def test_the_profiler_stops_synchronising_the_accelerator():
    """Three py-spy dumps agree on the frame: Profile.time -> synchronize ->
    _exchange_device. That synchronize exists so Ultralytics' milliseconds are
    the GPU's rather than the queue's -- a stopwatch, not the work."""
    from label_detections.ui.live_detect import InferenceWorker

    ops = pytest.importorskip("ultralytics.utils.ops")
    original = ops.Profile.time
    had_flag = getattr(ops.Profile, "_lv_unsynchronised", False)
    try:
        ops.Profile.time = original
        ops.Profile._lv_unsynchronised = False

        synced = []

        class FakeAccelerator:
            def synchronize(self, device):
                synced.append(device)

        assert InferenceWorker.unsynchronised_timing() == "", "did not apply"

        profile = ops.Profile(device="cuda:0")
        profile.accelerator = FakeAccelerator()
        with profile:
            pass
        assert synced == [], "the accelerator was still synchronised for timing"
        assert isinstance(profile.dt, float) and profile.dt >= 0.0, (
            "timing stopped working entirely")
    finally:
        ops.Profile.time = original
        ops.Profile._lv_unsynchronised = had_flag


@ui
def test_it_says_so_rather_than_silently_failing_to_apply():
    """A patch that stops applying because the library moved is worse than one
    that never did: the numbers would quietly go back to being synchronised and
    nobody would know which they were looking at."""
    from label_detections.ui.live_detect import InferenceWorker
    import label_detections.ui.live_detect as mod

    ops = pytest.importorskip("ultralytics.utils.ops")
    real = ops.Profile
    try:
        del ops.Profile
        why = InferenceWorker.unsynchronised_timing()
        assert why and "no longer looks" in why
    finally:
        ops.Profile = real


def test_the_readout_says_the_numbers_are_not_gpu_time():
    line = ld.phase_line({"preprocess": 3.0, "inference": 7.0,
                          "postprocess": 2.0, "_unsynced": 1.0}, total_ms=12.0)
    assert "issued, not GPU" in line

    synced = ld.phase_line({"preprocess": 3.0, "inference": 7.0,
                            "postprocess": 2.0}, total_ms=12.0)
    assert "issued, not GPU" not in synced


# --- a result must not cost a whole camera frame ----------------------------

@ui
def test_a_finished_result_takes_the_newest_frame_immediately():
    """The busy flag clears when the result lands on the GUI thread. If that
    happens between two ticks -- which it usually does -- the tick's pump check
    has already run and found the worker busy, so waiting for the next one
    costs a whole camera interval. That is one inference per two frames: 10/s
    behind a 17/s camera."""
    win = _window()
    _label(win, "beat")
    handed = []
    _watch_infer(win, handed)
    win._live_thread = object()
    win._live_loaded = True
    win._live_tracking = False
    win._live_busy = True
    win._live_last_started = time.monotonic() - 1.0
    submitted = np.zeros((8, 8, 3), np.uint8)
    win._inflight_frame = submitted
    win._inflight_scale = (1.0, 1.0)
    win._live_overlay_scale = (1.0, 1.0)
    # A newer frame has arrived while the result was in flight.
    win.last_raw = np.ones((8, 8, 3), np.uint8)
    real = win.camera
    try:
        win.camera = _FakeOpenCamera()
        win._on_live_result([], 0.02, {})
        assert len(handed) == 1, "the newest frame waited for the next tick"
        assert handed[0] is win.last_raw
    finally:
        win.camera = real
        win._live_thread = None
        win._live_loaded = False
        win._live_busy = False
        win.last_raw = None


@ui
def test_it_does_not_re_run_the_frame_it_just_finished():
    """No tick has happened, so there is nothing new to offer -- running it
    again would burn a GPU pass to redraw the same boxes."""
    win = _window()
    handed = []
    _watch_infer(win, handed)
    win._live_thread = object()
    win._live_loaded = True
    win._live_tracking = False
    win._live_busy = True
    win._live_last_started = time.monotonic() - 1.0
    frame = np.zeros((8, 8, 3), np.uint8)
    win._inflight_frame = frame
    win.last_raw = frame                    # the same object: no new tick
    real = win.camera
    try:
        win.camera = _FakeOpenCamera()
        win._on_live_result([], 0.02, {})
        assert handed == []
    finally:
        win.camera = real
        win._live_thread = None
        win._live_loaded = False
        win._live_busy = False
        win.last_raw = None


def test_the_note_does_not_blame_a_ceiling_the_rate_is_nowhere_near():
    """Naming the camera while running at 10/s behind a 17/s camera was a
    comfortable answer to a question nobody asked."""
    rolling = ld.Rolling()
    for i in range(6):
        rolling.record(0.033, now=100 + i * 0.097)   # 33 ms of work, 10/s
    note = ld.throughput_note(rolling, interval_s=1 / 30.0, camera_fps=17.0)
    assert "below every ceiling" in note
    assert "the limit is the camera" not in note


def test_the_note_still_names_the_camera_when_the_rate_is_reaching_it():
    rolling = ld.Rolling()
    for i in range(6):
        rolling.record(0.010, now=100 + i * (1 / 17.0))   # 10 ms of work, 17/s
    note = ld.throughput_note(rolling, interval_s=1 / 30.0, camera_fps=17.0)
    assert "the limit is the camera, at 17/s" in note
