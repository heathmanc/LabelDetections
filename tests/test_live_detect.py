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
    found = ld.frame_summary({"spec_plate": 1, "cert_mark": 2},
                             "spec_plate", "sp_g31", rolling)
    assert "spec_plate: 1 found" in found
    assert "cert_mark: 2" in found

    missed = ld.frame_summary({"cert_mark": 1}, "spec_plate", "sp_g31", rolling)
    assert "NOT FOUND" in missed


def test_the_readout_never_claims_the_model_identified_the_label():
    """The detector returns a family. A line reading "2220-9199: mean 0.94"
    says it returned an identity, and a wrong label would pass live looking
    perfectly healthy."""
    rolling = ld.Rolling()
    rolling.record(0.02, now=100)

    found = ld.frame_summary({"spec_plate": 1}, "spec_plate", "sp_g31", rolling)
    assert not found.splitlines()[1].startswith("sp_g31")
    assert "spec_plate" in found and "the family sp_g31 belongs to" in found

    book = ld.TrackBook()
    book.update([(1, "spec_plate", 0.94)], now=0.0)
    tracked = ld.track_summary(book, "spec_plate", "sp_g31", rolling)
    assert not tracked.splitlines()[1].startswith("sp_g31")
    assert "spec_plate: held" in tracked
    assert "the family sp_g31 belongs to" in tracked


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


def _label(win, label_id="live_sp", family="spec_plate"):
    from label_detections.core import persistence
    from label_detections.core.labels import LabelDef

    library = persistence.load_library()
    library.add(LabelDef(label_id=label_id, family=family), replace=True)
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
    win._live_worker = type("W", (), {"infer": lambda _s, f: handed.append(f)})()
    win._live_busy = True
    win._live_last_started = 0.0
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
        win._on_live_result([_Results([], {}, [], [])], 0.02)
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
        confident = _Results([[10, 10, 40, 40]], {0: "spec_plate"}, [0.97], [0])
        win._on_live_result([confident], 0.02)
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
            [_Results([[10, 10, 40, 40]], {0: "spec_plate"}, [0.9], [0])], 0.02)
        text = win.live_readout.toPlainText()
        assert "1 detection(s)" in text
        assert "spec_plate: 1 found" in text
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
        for conf in (0.88, 0.94, 0.91):
            win._on_live_result(
                [_Results([[10, 10, 40, 40]], {0: "spec_plate"}, [conf], [0],
                          ids=[7])], 0.02)
        text = win.live_readout.toPlainText()
        assert "1 tracked" in text
        assert "spec_plate: held 3 frames" in text
        assert "#7 spec_plate:" in text
        assert "0.91 mean over 3 frames" in text
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
    text = ld.track_summary(book, "spec_plate", "sp_g31", rolling)
    assert "spec_plate: NOT TRACKED" in text
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


def test_a_proposal_carries_the_family_the_detector_actually_returned():
    boxes = ld.proposed_boxes([_item("spec_plate", xyxy=[0, 0, 4, 4])],
                              "spec_plate", "sp_9199")
    assert boxes[0]["label"] == "spec_plate"


def test_only_the_open_labels_family_is_handed_its_identity():
    """A battery_side box is not this label. Filling in label_id anyway would
    put the wrong identity on a box a reviewer is likely to accept."""
    items = [_item("spec_plate", xyxy=[0, 0, 4, 4]),
             _item("battery_side", xyxy=[0, 0, 9, 9])]
    boxes = ld.proposed_boxes(items, "spec_plate", "sp_9199")
    assert boxes[0]["label_id"] == "sp_9199"
    assert "label_id" not in boxes[1]


def test_an_axis_aligned_detection_becomes_four_corners():
    boxes = ld.proposed_boxes([_item("spec_plate", xyxy=[10, 20, 30, 50])],
                              "spec_plate", "sp")
    assert boxes[0]["points"] == [[10.0, 20.0], [30.0, 20.0],
                                  [30.0, 50.0], [10.0, 50.0]]


def test_an_oriented_detection_keeps_its_own_corners():
    pts = [[0, 0], [10, 2], [9, 12], [-1, 10]]
    boxes = ld.proposed_boxes([_item("spec_plate", points=pts)], "spec_plate", "sp")
    assert boxes[0]["points"] == [[0.0, 0.0], [10.0, 2.0], [9.0, 12.0], [-1.0, 10.0]]


def test_confidence_and_track_id_survive_into_the_sidecar():
    boxes = ld.proposed_boxes(
        [_item("spec_plate", conf=0.83, track_id=4, xyxy=[0, 0, 4, 4])],
        "spec_plate", "sp")
    assert boxes[0]["confidence"] == pytest.approx(0.83)
    assert boxes[0]["track_id"] == 4


def test_a_detection_with_no_geometry_is_dropped_rather_than_written_empty():
    assert ld.proposed_boxes([_item("spec_plate")], "spec_plate", "sp") == []
    assert ld.proposed_boxes([_item("", xyxy=[0, 0, 1, 1])], "spec_plate", "sp") == []


def test_every_proposed_box_is_marked_as_the_machines_work():
    boxes = ld.proposed_boxes([_item("spec_plate", xyxy=[0, 0, 4, 4])],
                              "spec_plate", "sp")
    assert boxes[0]["proposed_by"] == ld.PROPOSED_BY


def test_a_proposed_sidecar_is_never_review_marked():
    """The one thing that must not happen: a machine proposal that reads as an
    operator's approval and exports as truth."""
    from label_detections.core import review

    data = ld.proposed_annotation(
        "f.jpg", "sp", "spec_plate",
        [_item("spec_plate", xyxy=[0, 0, 4, 4])], 640, 480)
    assert review.annotation_reviewed(data) is False
    assert review.annotation_status(data, "sp") == "needs_review"
    assert review.export_ready(review.annotation_status(data, "sp")) is False


def test_a_proposed_sidecar_records_the_frame_it_describes():
    data = ld.proposed_annotation(
        "f.jpg", "sp", "spec_plate",
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
            ld.proposed_annotation(f"f{i}.jpg", "sp", "spec_plate",
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
            ld.proposed_annotation(f"{s}.jpg", "sp", "spec_plate",
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
    label_id = _label(win, "live_json", family="spec_plate")
    win._live_result_frame = np.zeros((60, 80, 3), np.uint8)
    win._live_result_items = [
        {"name": "spec_plate", "conf": 0.91, "xyxy": [4.0, 6.0, 40.0, 30.0]}]
    win._live_session = "live_20260827_120000"
    win.keep_live_frame_with_detections()

    name = sorted(p.name for p in storage.list_images(label_id))[-1]
    data = persistence.load_annotation(label_id, name)
    assert data is not None
    assert len(data["boxes"]) == 1
    assert data["boxes"][0]["label"] == "spec_plate"
    assert data["boxes"][0]["label_id"] == label_id
    # The frame's own dimensions, not the preview's.
    assert (data["width"], data["height"]) == (80, 60)
    assert data["session"] == "live_20260827_120000"


@ui
def test_a_kept_proposal_still_has_to_be_reviewed_by_a_person():
    from label_detections.core import persistence, review, storage

    win = _window()
    label_id = _label(win, "live_json_unrev", family="spec_plate")
    win._live_result_frame = np.zeros((60, 80, 3), np.uint8)
    win._live_result_items = [
        {"name": "spec_plate", "conf": 0.91, "xyxy": [4.0, 6.0, 40.0, 30.0]}]
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
    label_id = _label(win, "live_pair", family="spec_plate")

    inferred = np.zeros((60, 80, 3), np.uint8)      # what the model saw
    inferred[:] = 40
    win._live_result_frame = inferred
    win._live_result_items = [
        {"name": "spec_plate", "conf": 0.9, "xyxy": [4.0, 6.0, 40.0, 30.0]}]
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
    label_id = _label(win, "live_empty", family="spec_plate")
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
    label_id = _label(win, "live_clicks", family="spec_plate")
    frame = np.zeros((60, 80, 3), np.uint8)
    win._live_frame = frame
    win._live_result_frame = frame
    win._live_result_items = [
        {"name": "spec_plate", "conf": 0.9, "xyxy": [4.0, 6.0, 40.0, 30.0]}]

    win.live_keep_btn.click()
    plain = sorted(p.name for p in storage.list_images(label_id))[-1]
    assert persistence.load_annotation(label_id, plain) is None

    win.live_keep_json_btn.click()
    names = sorted(p.name for p in storage.list_images(label_id))
    assert len(names) == 2
    proposed = [n for n in names if n != plain][0]
    assert persistence.load_annotation(label_id, proposed)["boxes"]
    assert win.live_keep_btn.text() != win.live_keep_json_btn.text()
