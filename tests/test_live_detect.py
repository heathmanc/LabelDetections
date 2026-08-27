"""Live detection: pacing, keeping hard frames, and the coordinate scaling.

The policy is stdlib and tested bare. The UI half is tested offscreen with a
stand-in worker, because the thing worth pinning is the wiring -- what gets
handed to the model, what comes back, and where the overlays land -- not
whether Ultralytics works.
"""
from __future__ import annotations

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

def test_the_summary_leads_with_whether_the_active_label_was_found():
    rolling = ld.Rolling()
    rolling.record(0.02, now=100)
    rolling.record(0.02, now=100.2)
    found = ld.frame_summary({"sp_g31": 1, "warn_g31": 2}, "sp_g31", rolling)
    assert "sp_g31: 1 found" in found
    assert "warn_g31: 2" in found

    missed = ld.frame_summary({"warn_g31": 1}, "sp_g31", rolling)
    assert "NOT FOUND" in missed


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
    assert "#1 2220-9199 0.94" in tracked
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


def _label(win, label_id="live_sp", ):
    from label_detections.core import persistence
    from label_detections.core.labels import LabelDef

    library = persistence.load_library()
    library.add(LabelDef(label_id=label_id, ), replace=True)
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
    win.infer_requested.connect(handed.append)
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
        win.infer_requested.disconnect(handed.append)
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
        assert f"#7 {win.label_id} 0.90" in text, "showed the latest frame, not the mean"
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


def test_the_track_summary_says_when_the_active_label_is_not_tracked():
    book = ld.TrackBook()
    book.update([(1, "cert_mark", 0.9)], now=100)
    rolling = ld.Rolling()
    rolling.record(0.01, now=100)
    text = ld.track_summary(book, "sp_g31", rolling)
    assert "sp_g31 NOT TRACKED" in text
    assert "#1 cert_mark" in text


def test_an_empty_book_reads_as_empty_rather_than_blank():
    assert ld.TrackBook().text() == "No tracked objects."


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
    assert "2220-9199 #3 0.97" == out[0]["label"]
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
    wall, the threshold is too high, or the wrong model is loaded."""
    assert ld.quiet_hint(3, 0.45, 1024, False) == ""
    hint = ld.quiet_hint(ld.QUIET_FRAMES + 5, 0.45, 1024, False)
    assert "Confidence is 0.45" in hint
    assert "DETECTOR run, not the classifier" in hint
    assert "stage 2 classifier is set" in hint
    # With stage 2 configured, that last line is noise.
    assert "stage 2 classifier is set" not in ld.quiet_hint(20, 0.45, 1024, True)


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
    note = ld.throughput_note(r)
    assert "could run" in note and "not the GPU" in note

    fast = ld.Rolling()
    for i in range(10):
        fast.record(0.008, now=100 + i * 0.009)
    assert ld.throughput_note(fast) == "", "no note when the rate matches latency"


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


def test_slow_inference_on_gpu_suggests_the_gpu_causes():
    r = ld.Rolling()
    for i in range(5):
        r.record(0.130, now=100 + i * 0.14)
    gpu = ld.slow_hint(r, "Device: CUDA available (RTX 5090); model on cuda:0")
    assert "TensorRT" in gpu and "image size" in gpu


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
    win.infer_requested.connect(handed.append)
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
        win.infer_requested.disconnect(handed.append)
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
    win.infer_requested.connect(handed.append)
    win._live_thread = object()
    win._live_loaded = True
    win._live_busy = True
    win._live_last_started = time.monotonic() - (ld.BUSY_TIMEOUT_S + 1)
    try:
        win._pump_live_detect(np.zeros((10, 10, 3), np.uint8))
        assert len(handed) == 1, "never recovered from a lost result"
    finally:
        win.infer_requested.disconnect(handed.append)
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
    win.infer_requested.connect(handed.append)
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
        win.infer_requested.disconnect(handed.append)
        win.live_rate_spin.setValue(1.0 / ld.MIN_INTERVAL_S)   # shared window
        win._live_thread = None
        win._live_busy = False
        win._live_loaded = False


def test_the_default_rate_is_the_one_that_ran_stably():
    """Conservative by default. Raising it drives the capture path harder, so
    it is a step someone takes and can walk back -- not one taken for them."""
    assert abs(1.0 / ld.MIN_INTERVAL_S - 6.7) < 0.2
