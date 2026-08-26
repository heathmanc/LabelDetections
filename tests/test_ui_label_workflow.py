"""Offscreen tests of the label-centric workflow the fork introduced.

The smoke tests prove the window builds. These prove it does the right thing:
switching labels rescopes the app, saving approves only images that carry the
label they were collected for, and a stale approval is cleared rather than
left behind.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-workflow-"))

import pytest

try:
    from PySide6.QtWidgets import QApplication
    import cv2  # noqa: F401
    import numpy as np
    HAVE_QT = True
except Exception as exc:  # pragma: no cover - depends on the environment
    HAVE_QT = False
    _WHY = exc

pytestmark = pytest.mark.skipif(not HAVE_QT, reason="PySide6/cv2 not available")


def _define(win, label_id, family="spec_plate"):
    """Add a label to the library and make the window aware of it.

    Every test uses its own label id. The whole suite shares one process, and
    storage resolves the data root once at import, so two test modules cannot
    have separate libraries -- isolating by label id is what keeps these tests
    order-independent instead of depending on an empty folder.
    """
    from label_detections.core import persistence
    from label_detections.core.labels import LabelDef

    library = persistence.load_library()
    library.add(LabelDef(label_id=label_id, family=family, size_mm=[90, 60],
                         reference_images=["ref.png"], train_target=10),
                replace=True)
    persistence.save_library(library)
    win.library = persistence.load_library()
    return label_id


_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


def _capture(win, label_id, name="frame_001.jpg"):
    """Write a real image into a label's dataset and open it."""
    from label_detections.core import storage
    folder = storage.dataset_folder(label_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    cv2.imwrite(str(path), np.zeros((200, 400, 3), dtype=np.uint8))
    win._dataset_index_dirty = True
    win._load_image_path(path)
    return path


def _draw(win, family, label_id=""):
    from label_detections.ui.canvas import Box
    box = Box(x=10, y=10, w=100, h=60, class_id=0, label=family, kind="obb",
              points=[[10, 10], [110, 10], [110, 70], [10, 70]])
    if label_id:
        box.label_id = label_id
    win.canvas.boxes.append(box)
    return box


def test_opening_a_label_selects_its_family_for_drawing():
    """The next thing an operator does is draw one; the wrong family is a mislabel."""
    win = _window()
    _define(win, "wf_family", family="cert_mark")
    win.set_active_label("wf_family")
    assert win.class_combo.currentText() == "cert_mark"


def test_the_app_opens_scoped_to_a_label():
    win = _window()
    _define(win, "wf_open")
    win.set_active_label("wf_open")
    assert win.label_id == "wf_open"
    assert win.library.get(win.label_id) is not None


def test_the_label_list_shows_progress_against_each_target():
    win = _window()
    _define(win, "wf_progress")
    win._refresh_labels()
    rows = [win.label_list.item(i).text() for i in range(win.label_list.count())]
    assert any("wf_progress" in r and "/10" in r for r in rows)


def test_switching_label_rescopes_the_whole_app():
    win = _window()
    _define(win, "wf_switch_a")
    _define(win, "wf_switch_b", family="trace_tag")

    win.set_active_label("wf_switch_a")
    _capture(win, "wf_switch_a", "a.jpg")
    win._refresh_images()
    assert win.image_list.count() == 1

    win.set_active_label("wf_switch_b")
    assert win.label_id == "wf_switch_b"
    assert win.image_list.count() == 0          # a different dataset
    assert win.canvas.boxes == []


def test_saving_an_image_that_carries_the_label_approves_it():
    from label_detections.core import review, storage
    win = _window()
    _define(win, "wf_save_good")
    win.set_active_label("wf_save_good")
    path = _capture(win, "wf_save_good", "good.jpg")
    _draw(win, "spec_plate")                     # family only; identity is stamped
    win.save_labels()

    data = storage.load_annotations(path)
    assert data["boxes"][0]["label_id"] == "wf_save_good"
    assert review.annotation_reviewed(data)
    assert review.annotation_status(data, "wf_save_good") == "ready"


def test_saving_an_image_without_the_label_saves_but_does_not_approve():
    from label_detections.core import review, storage
    win = _window()
    _define(win, "wf_save_other")
    win.set_active_label("wf_save_other")
    path = _capture(win, "wf_save_other", "other.jpg")
    _draw(win, "warning_label")                  # a different family entirely
    win.save_labels()

    data = storage.load_annotations(path)
    assert data["boxes"], "geometry is still saved"
    assert not review.annotation_reviewed(data)


def test_editing_an_approved_image_into_an_empty_one_clears_the_approval():
    """Editing is not approving -- the stale-marker bug, in one test."""
    from label_detections.core import review, storage
    win = _window()
    _define(win, "wf_edited")
    win.set_active_label("wf_edited")
    path = _capture(win, "wf_edited", "edited.jpg")
    _draw(win, "spec_plate")
    win.save_labels()
    assert review.annotation_reviewed(storage.load_annotations(path))

    win.canvas.boxes.clear()
    _draw(win, "warning_label")
    win.save_labels()
    assert not review.annotation_reviewed(storage.load_annotations(path))


def test_the_box_counter_reports_the_active_label():
    win = _window()
    _define(win, "wf_counter")
    win.set_active_label("wf_counter")
    win.canvas.boxes.clear()
    _draw(win, "spec_plate", "wf_counter")
    win._update_box_count()
    assert "wf_counter" in win.count_label.text()
    assert "1 drawn" in win.count_label.text()


def test_validation_runs_against_what_is_on_screen():
    win = _window()
    _define(win, "wf_validate")
    win.set_active_label("wf_validate")
    _capture(win, "wf_validate", "validate.jpg")
    win.canvas.boxes.clear()
    _draw(win, "spec_plate", "wf_validate")
    payload = win._current_annotation_for_validation()
    assert payload["label_id"] == "wf_validate"
    assert payload["boxes"][0]["label_id"] == "wf_validate"


def test_family_shortcuts_exist_for_every_configured_family():
    win = _window()
    titles = {a.text() for a in win.actions()}
    for family in ("battery_side", "spec_plate", "cert_mark"):
        assert f"Class: {family}" in titles


def test_no_recipe_authoring_survived_the_fork():
    """Recipes belong to the front end; nothing here should author one."""
    win = _window()
    tabs = [win.tabs.tabText(i) for i in range(win.tabs.count())]
    assert not any("recipe" in t.lower() for t in tabs)
    assert not hasattr(win, "recipe")
    for attribute in ("save_recipe_from_ui", "_refresh_recipes", "run_count_test"):
        assert not hasattr(win, attribute), f"{attribute} should be gone"


# --- read-regions from a captured image ------------------------------------

def test_regions_are_defined_from_a_capture_not_an_artwork_file(monkeypatch):
    """The workflow question: no external file, no measuring, no calibration.

    Capture an image, draw the label's box, and that box -- flattened -- becomes
    the artwork the regions are drawn on.
    """
    from label_detections.core import persistence

    win = _window()
    _define(win, "wf_regions")
    win.set_active_label("wf_regions")
    _capture(win, "wf_regions", "with_label.jpg")
    win.canvas.boxes.clear()
    _draw(win, "spec_plate")

    # Stand in for the operator dragging two regions on the flattened crop.
    import label_detections.ui.region_editor as region_editor

    class FakeDialog:
        def __init__(self, reference, codes, text_fields, anchor, parent=None):
            FakeDialog.reference = reference

        def exec(self):
            return True

        def result_regions(self):
            return {
                "codes": [{"role": "serial", "symbology": "datamatrix",
                           "policy": "must_decode", "region": [0.6, 0.1, 0.3, 0.4]}],
                "text_fields": [{"name": "date_code", "policy": "must_be_present",
                                 "region": [0.1, 0.7, 0.4, 0.2]}],
                "anchor_region": [0.0, 0.0, 0.5, 0.5],
            }

    monkeypatch.setattr(region_editor, "RegionEditorDialog", FakeDialog)
    win.define_read_regions()

    label = persistence.load_library().get("wf_regions")
    assert label.code_by_role("serial").region == [0.6, 0.1, 0.3, 0.4]
    assert label.text_fields[0].name == "date_code"
    # Drawing an anchor is itself the statement that the artwork varies.
    assert label.variable_data is True
    # The flattened crop was saved as this label's artwork -- no file hunting.
    assert label.reference_images and Path(label.reference_images[0]).is_file()
    assert FakeDialog.reference == label.reference_images[0]


def test_defined_regions_then_place_themselves_on_every_other_image():
    """The point of storing fractions: draw once, applies to every image."""
    from label_detections.core import persistence
    from label_detections.core.labels import CodeSpec

    win = _window()
    _define(win, "wf_reuse")
    library = persistence.load_library()
    label = library.get("wf_reuse")
    label.codes = [CodeSpec(role="serial", region=[0.5, 0.25, 0.25, 0.5])]
    library.add(label, replace=True)
    persistence.save_library(library)
    win.library = persistence.load_library()

    win.set_active_label("wf_reuse")
    _capture(win, "wf_reuse", "another.jpg")
    win.canvas.boxes.clear()
    _draw(win, "spec_plate")             # box at (10, 10) 100x60
    win.place_regions_on_canvas()

    box = win.canvas.boxes[0]
    assert box.label_id == "wf_reuse"
    code = next(r for r in box.regions if r.get("code_role") == "serial")
    # 50% across and 25% down a 100x60 box drawn at (10, 10).
    assert code["points"][0] == pytest.approx([60.0, 25.0])


def test_defining_regions_without_a_label_box_says_what_to_do():
    win = _window()
    _define(win, "wf_no_box")
    win.set_active_label("wf_no_box")
    _capture(win, "wf_no_box", "empty.jpg")
    win.canvas.boxes.clear()

    shown = {}
    import label_detections.ui.main_window as mw_mod
    original = mw_mod.QMessageBox.information
    mw_mod.QMessageBox.information = lambda parent, title, text, *a, **k: shown.update(
        {"title": title, "text": text})
    try:
        win.define_read_regions()
    finally:
        mw_mod.QMessageBox.information = original
    assert "Draw the wf_no_box box" in shown.get("text", "")


def test_the_action_is_reachable_without_opening_the_wizard():
    """It was buried on a wizard page, which is why nobody found it."""
    win = _window()
    titles = {a.text() for a in win.actions()}
    assert "Define read-regions from this image" in titles
    assert "Place read-regions" in titles


# --- capturing the reference from the camera --------------------------------

def _fake_editor(monkeypatch, result=None):
    """Stand in for the operator dragging regions on the flattened crop."""
    import label_detections.ui.region_editor as region_editor

    calls = {}

    class FakeDialog:
        def __init__(self, reference, codes, text_fields, anchor, parent=None):
            calls["reference"] = reference
            calls["opened"] = calls.get("opened", 0) + 1

        def exec(self):
            return True

        def result_regions(self):
            return result or {"codes": [], "text_fields": [], "anchor_region": []}

    monkeypatch.setattr(region_editor, "RegionEditorDialog", FakeDialog)
    return calls


def test_capture_reference_saves_the_frame_and_arms_the_follow_through(monkeypatch):
    """A picture of the label is training data whatever else it is used for."""
    import numpy as np
    from label_detections.core import storage

    win = _window()
    _define(win, "wf_capref")
    win.set_active_label("wf_capref")
    win.last_raw = np.zeros((200, 400, 3), dtype=np.uint8)

    before = len(storage.list_images("wf_capref"))
    win.capture_reference()
    assert len(storage.list_images("wf_capref")) == before + 1
    assert win._awaiting_reference_box is True
    assert "draw the wf_capref box" in win.guidance_label.text()


def test_drawing_the_box_after_a_reference_capture_opens_the_editor(monkeypatch):
    """Capture, draw, draw regions -- no menu hunting in between."""
    import numpy as np
    from PySide6.QtWidgets import QApplication

    calls = _fake_editor(monkeypatch)
    win = _window()
    _define(win, "wf_capflow")
    win.set_active_label("wf_capflow")
    win.last_raw = np.zeros((200, 400, 3), dtype=np.uint8)
    win.capture_reference()

    _draw(win, "spec_plate")
    win._update_box_count()
    QApplication.processEvents()          # the deferred open

    assert calls.get("opened") == 1
    assert win._awaiting_reference_box is False


def test_a_box_of_another_family_does_not_trigger_it(monkeypatch):
    import numpy as np
    from PySide6.QtWidgets import QApplication

    calls = _fake_editor(monkeypatch)
    win = _window()
    _define(win, "wf_capwrong", family="cert_mark")
    win.set_active_label("wf_capwrong")
    win.last_raw = np.zeros((200, 400, 3), dtype=np.uint8)
    win.capture_reference()

    _draw(win, "warning_label")
    win._update_box_count()
    QApplication.processEvents()

    assert "opened" not in calls
    assert win._awaiting_reference_box is True


def test_switching_label_cancels_an_armed_reference_capture():
    import numpy as np

    win = _window()
    _define(win, "wf_capcancel_a")
    _define(win, "wf_capcancel_b")
    win.set_active_label("wf_capcancel_a")
    win.last_raw = np.zeros((200, 400, 3), dtype=np.uint8)
    win.capture_reference()

    win.set_active_label("wf_capcancel_b")
    assert win._awaiting_reference_box is False


def test_capture_reference_without_a_frame_says_so():
    win = _window()
    _define(win, "wf_noframe")
    win.set_active_label("wf_noframe")
    win.last_raw = None

    shown = {}
    import label_detections.ui.main_window as mw_mod
    original = mw_mod.QMessageBox.information
    mw_mod.QMessageBox.information = lambda parent, title, text, *a, **k: shown.update(
        {"text": text})
    try:
        win.capture_reference()
    finally:
        mw_mod.QMessageBox.information = original
    assert "live preview" in shown.get("text", "")


# --- live preview is not a drawing surface ----------------------------------

class _FakeCamera:
    def __init__(self, open_):
        self._open = open_

    def is_open(self):
        return self._open

    def close(self):
        self._open = False


def test_drawing_is_blocked_while_a_camera_streams():
    """A box on a frame replaced 30x a second belongs to no image."""
    win = _window()
    real = win.camera
    try:
        win.camera = _FakeCamera(True)
        win._refresh_live_mode()
        assert win.canvas.drawing_enabled is False
        assert "Live" in win.mode_label.text()
        assert "capture a frame before drawing" in win.guidance_label.text()
    finally:
        win.camera = real
        win._refresh_live_mode()


def test_drawing_comes_back_once_the_camera_stops():
    win = _window()
    real = win.camera
    try:
        win.camera = _FakeCamera(True)
        win._refresh_live_mode()
        win.camera.close()
        win._refresh_live_mode()
        assert win.canvas.drawing_enabled is True
        assert "Labeling" in win.mode_label.text()
    finally:
        win.camera = real
        win._refresh_live_mode()


def test_a_blocked_canvas_ignores_a_draw_but_still_pans():
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    win = _window()
    win.canvas.set_drawing_enabled(False)
    try:
        before = len(win.canvas.boxes)
        event = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(40, 40),
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        win.canvas.mousePressEvent(event)
        assert len(win.canvas.boxes) == before
        # Left-drag falls through to panning, so the view is still navigable.
        assert win.canvas.panning is True
    finally:
        win.canvas.panning = False
        win.canvas.set_drawing_enabled(True)


def test_the_canvas_says_why_it_is_not_accepting_boxes():
    win = _window()
    win.canvas.set_drawing_enabled(False)
    try:
        assert win.canvas._blocked() is True
    finally:
        win.canvas.set_drawing_enabled(True)


# --- burst capture, and marking the reference -------------------------------

def test_capturing_does_not_stop_the_preview():
    """Capture is a burst activity: frame, shoot, reposition, shoot again."""
    import numpy as np
    from label_detections.core import storage

    win = _window()
    _define(win, "wf_burst")
    win.set_active_label("wf_burst")
    real = win.camera
    try:
        win.camera = _FakeCamera(True)
        win._refresh_live_mode()
        win.last_raw = np.zeros((120, 200, 3), dtype=np.uint8)

        for _ in range(3):
            win.capture_frame(save_adjusted=False)

        assert win.camera.is_open() is True          # never stopped
        assert win.canvas.drawing_enabled is False   # still a live frame
        assert win._session_captures == 3
        assert len(storage.list_images("wf_burst")) >= 3
        # The canvas is not repointed at a still nobody is looking at.
        assert win.current_image_path is None
    finally:
        win.camera = real
        win._refresh_live_mode()


def test_stopping_the_preview_opens_the_last_capture_ready_to_label():
    import numpy as np

    win = _window()
    _define(win, "wf_burst_end")
    win.set_active_label("wf_burst_end")
    real = win.camera
    try:
        win.camera = _FakeCamera(True)
        win._refresh_live_mode()
        win.last_raw = np.zeros((120, 200, 3), dtype=np.uint8)
        win.capture_frame(save_adjusted=False)
        last = win._last_capture_path

        win.close_camera()
        assert win.current_image_path == last
        assert win.canvas.drawing_enabled is True
        assert win._session_captures == 0            # the count resets per session
    finally:
        win.camera = real
        win._refresh_live_mode()


def test_the_image_the_artwork_came_from_is_marked_in_the_list(monkeypatch):
    """Redefining regions from a different shot silently moves every region."""
    from label_detections.core import persistence

    calls = _fake_editor(monkeypatch, {
        "codes": [], "text_fields": [], "anchor_region": [0.0, 0.0, 0.5, 0.5]})
    win = _window()
    _define(win, "wf_marker")
    win.set_active_label("wf_marker")
    plain = _capture(win, "wf_marker", "plain.jpg")
    source = _capture(win, "wf_marker", "source.jpg")
    win.canvas.boxes.clear()
    _draw(win, "spec_plate")
    win.define_read_regions()
    assert calls.get("opened") == 1

    label = persistence.load_library().get("wf_marker")
    assert Path(label.reference_source) == source

    win.library = persistence.load_library()
    win._image_status_cache.clear()
    assert "◆ REFERENCE" in win._cached_image_status(source)["prefix"]
    assert "◆ REFERENCE" not in win._cached_image_status(plain)["prefix"]


def _with_artwork(win, label_id, monkeypatch, name="first.jpg"):
    """Get a label to the state of having artwork, and return the source image."""
    _define(win, label_id)
    win.set_active_label(label_id)
    source = _capture(win, label_id, name)
    win.canvas.boxes.clear()
    _draw(win, "spec_plate")
    calls = _fake_editor(monkeypatch)
    win.define_read_regions()
    assert calls.get("opened") == 1
    return source, calls


def test_a_second_define_edits_the_existing_artwork_instead_of_making_new(monkeypatch):
    """Artwork is defined once. Regions are fractions of it, so re-flattening a
    different shot moves every one of them against images already reviewed."""
    from label_detections.core import persistence

    win = _window()
    first, calls = _with_artwork(win, "wf_once", monkeypatch)
    artwork = persistence.load_library().get("wf_once").reference_images[0]

    second = _capture(win, "wf_once", "second.jpg")
    win.canvas.boxes.clear(); _draw(win, "spec_plate")
    win.define_read_regions()

    assert calls["opened"] == 2                       # it opened, ...
    assert calls["reference"] == artwork              # ... on the SAME artwork
    label = persistence.load_library().get("wf_once")
    assert Path(label.reference_source) == first      # the source did not move
    assert "◆ REFERENCE" in win._cached_image_status(first)["prefix"]
    assert "◆ REFERENCE" not in win._cached_image_status(second)["prefix"]


def test_replacing_artwork_asks_first_and_says_what_it_costs(monkeypatch):
    win = _window()
    _with_artwork(win, "wf_replace_no", monkeypatch)
    _capture(win, "wf_replace_no", "second.jpg")
    win.canvas.boxes.clear(); _draw(win, "spec_plate")

    asked = {}
    import label_detections.ui.main_window as mw_mod
    original = mw_mod.QMessageBox.question
    mw_mod.QMessageBox.question = lambda parent, title, text, *a, **k: (
        asked.update({"text": text}) or mw_mod.QMessageBox.No)
    try:
        win.replace_label_artwork()
    finally:
        mw_mod.QMessageBox.question = original

    assert "every one of them moves" in asked.get("text", "")
    assert "already reviewed" in asked.get("text", "")


def test_confirming_the_replace_moves_the_artwork_and_the_marker(monkeypatch):
    from label_detections.core import persistence

    win = _window()
    first, calls = _with_artwork(win, "wf_replace_yes", monkeypatch)
    second = _capture(win, "wf_replace_yes", "second.jpg")
    win.canvas.boxes.clear(); _draw(win, "spec_plate")

    import label_detections.ui.main_window as mw_mod
    original = mw_mod.QMessageBox.question
    mw_mod.QMessageBox.question = lambda *a, **k: mw_mod.QMessageBox.Yes
    try:
        win.replace_label_artwork()
    finally:
        mw_mod.QMessageBox.question = original

    label = persistence.load_library().get("wf_replace_yes")
    assert Path(label.reference_source) == second
    # The old marker must not linger: the status cache keys on it for this reason.
    assert "◆ REFERENCE" not in win._cached_image_status(first)["prefix"]
    assert "◆ REFERENCE" in win._cached_image_status(second)["prefix"]


def test_replacing_carries_the_existing_regions_onto_the_new_artwork(monkeypatch):
    """The one thing that silently breaks is an outline drawn differently."""
    from label_detections.core import persistence
    from label_detections.core.labels import CodeSpec

    win = _window()
    _with_artwork(win, "wf_replace_carry", monkeypatch)
    library = persistence.load_library()
    label = library.get("wf_replace_carry")
    label.codes = [CodeSpec(role="serial", region=[0.5, 0.25, 0.2, 0.2])]
    library.add(label, replace=True); persistence.save_library(library)
    win.library = persistence.load_library()

    _capture(win, "wf_replace_carry", "second.jpg")
    win.canvas.boxes.clear(); _draw(win, "spec_plate")

    seen = {}
    import label_detections.ui.region_editor as region_editor

    class Capturing:
        def __init__(self, reference, codes, text_fields, anchor, parent=None):
            seen["codes"] = codes

        def exec(self):
            return False        # cancelled: nothing should be written

    monkeypatch.setattr(region_editor, "RegionEditorDialog", Capturing)
    import label_detections.ui.main_window as mw_mod
    original = mw_mod.QMessageBox.question
    mw_mod.QMessageBox.question = lambda *a, **k: mw_mod.QMessageBox.Yes
    try:
        win.replace_label_artwork()
    finally:
        mw_mod.QMessageBox.question = original

    assert seen["codes"][0]["region"] == [0.5, 0.25, 0.2, 0.2]


def test_capture_reference_refuses_once_a_label_has_artwork(monkeypatch):
    import numpy as np

    win = _window()
    _with_artwork(win, "wf_capref_once", monkeypatch)
    win.last_raw = np.zeros((120, 200, 3), dtype=np.uint8)

    shown = {}
    import label_detections.ui.main_window as mw_mod
    original = mw_mod.QMessageBox.information
    mw_mod.QMessageBox.information = lambda parent, title, text, *a, **k: shown.update(
        {"text": text})
    try:
        win.capture_reference()
    finally:
        mw_mod.QMessageBox.information = original

    assert "already has artwork" in shown.get("text", "")
    assert win._awaiting_reference_box is False


def test_artwork_deleted_from_disk_can_be_defined_again(monkeypatch):
    """Recovering from a missing file is not the same act as replacing artwork."""
    from label_detections.core import persistence

    win = _window()
    _with_artwork(win, "wf_recover", monkeypatch)
    label = persistence.load_library().get("wf_recover")
    Path(label.reference_images[0]).unlink()
    win.library = persistence.load_library()

    assert win._existing_artwork(win.library.get("wf_recover")) is None


# --- the list row must not become a file path -------------------------------

def test_a_stacked_prefix_does_not_leak_into_the_file_name():
    """The bug this pins: "◆ REFERENCE  ✓ REVIEWED OK  x.jpg" split on the FIRST
    double space handed back "✓ REVIEWED OK  x.jpg" as the name, and opening
    that row tried to read a file by that name."""
    win = _window()
    assert win._image_name_from_list_item(
        "◆ REFERENCE  ✓ REVIEWED OK  2220-9199_20260826_161029_729.jpg"
    ) == "2220-9199_20260826_161029_729.jpg"
    assert win._image_name_from_list_item("✓ REVIEWED OK  a.jpg") == "a.jpg"
    assert win._image_name_from_list_item("□ NO JSON  b.jpg") == "b.jpg"
    assert win._image_name_from_list_item("🟡 REVIEW 1x  c.jpg") == "c.jpg"
    assert win._image_name_from_list_item("plain.jpg") == "plain.jpg"


def test_rows_carry_their_file_name_rather_than_it_being_parsed_back(monkeypatch):
    """Prefixes stack and change; the name behind a row must not depend on them."""
    from PySide6.QtCore import Qt

    _fake_editor(monkeypatch)
    win = _window()
    _define(win, "wf_rowname")
    win.set_active_label("wf_rowname")
    source = _capture(win, "wf_rowname", "row_name.jpg")
    win.canvas.boxes.clear(); _draw(win, "spec_plate")
    win.define_read_regions()          # gives this row the REFERENCE prefix too

    win._image_status_cache.clear()
    win._refresh_images(force=True)
    row = next(win.image_list.item(i) for i in range(win.image_list.count())
               if "row_name.jpg" in win.image_list.item(i).text())
    assert "◆ REFERENCE" in row.text()
    assert row.data(Qt.ItemDataRole.UserRole) == "row_name.jpg"
    assert win._image_name_from_list_item(row) == "row_name.jpg"


def test_opening_a_marked_row_loads_the_real_image(monkeypatch):
    """End to end: the failure was an unreadable path, so open one and check."""
    _fake_editor(monkeypatch)
    win = _window()
    _define(win, "wf_openrow")
    win.set_active_label("wf_openrow")
    source = _capture(win, "wf_openrow", "open_me.jpg")
    win.canvas.boxes.clear(); _draw(win, "spec_plate")
    win.define_read_regions()

    win.current_image_path = None
    win._image_status_cache.clear()
    win._refresh_images(force=True)
    row = next(win.image_list.item(i) for i in range(win.image_list.count())
               if "open_me.jpg" in win.image_list.item(i).text())
    win.image_list.setCurrentItem(row)
    win._load_selected_image()

    assert win.current_image_path == source
    assert win.canvas.image_w > 0        # it actually decoded
