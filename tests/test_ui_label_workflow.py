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


def _define(win, label_id):
    """Add a label to the library and make the window aware of it.

    Every test uses its own label id. The whole suite shares one process, and
    storage resolves the data root once at import, so two test modules cannot
    have separate libraries -- isolating by label id is what keeps these tests
    order-independent instead of depending on an empty folder.
    """
    from label_detections.core import persistence
    from label_detections.core.labels import LabelDef

    library = persistence.load_library()
    library.add(LabelDef(label_id=label_id, size_mm=[90, 60],
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


def _draw(win, class_name, label_id=""):
    """Draw a box of one detector class. The class is a label id now, so the
    box carries its own identity unless a test deliberately withholds it."""
    from label_detections.ui.canvas import Box
    box = Box(x=10, y=10, w=100, h=60, class_id=0, label=class_name, kind="obb",
              points=[[10, 10], [110, 10], [110, 70], [10, 70]])
    box.label_id = label_id or (
        "" if class_name in ("battery_side",) else class_name)
    win.canvas.boxes.append(box)
    return box


def test_opening_a_label_selects_it_for_drawing():
    """The next thing an operator does is draw one, and the class is the
    identity now -- so the wrong class is a wrong label id in the dataset."""
    win = _window()
    _define(win, "wf_family")
    win.set_active_label("wf_family")
    assert win.class_combo.currentText() == "wf_family"


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
    _define(win, "wf_switch_b", )

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
    _draw(win, win.label_id)                     # the class carries the identity
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
    _draw(win, "some_other_label")                  # a different label entirely
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
    _draw(win, win.label_id)
    win.save_labels()
    assert review.annotation_reviewed(storage.load_annotations(path))

    win.canvas.boxes.clear()
    _draw(win, "some_other_label")
    win.save_labels()
    assert not review.annotation_reviewed(storage.load_annotations(path))


def test_the_box_counter_reports_the_active_label():
    win = _window()
    _define(win, "wf_counter")
    win.set_active_label("wf_counter")
    win.canvas.boxes.clear()
    _draw(win, "wf_counter")
    win._update_box_count()
    assert "wf_counter" in win.count_label.text()
    assert "1 drawn" in win.count_label.text()


def test_validation_runs_against_what_is_on_screen():
    win = _window()
    _define(win, "wf_validate")
    win.set_active_label("wf_validate")
    _capture(win, "wf_validate", "validate.jpg")
    win.canvas.boxes.clear()
    _draw(win, "wf_validate")
    payload = win._current_annotation_for_validation()
    assert payload["label_id"] == "wf_validate"
    assert payload["boxes"][0]["label_id"] == "wf_validate"


def test_two_class_shortcuts_cover_everything_that_gets_drawn():
    """One shortcut per class stopped being possible when classes became label
    ids. Only two things are ever drawn on an image: this label, or the face."""
    win = _window()
    titles = {a.text() for a in win.actions()}
    assert "Class: battery face" in titles
    assert "Class: the label being trained" in titles


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
    _draw(win, win.label_id)

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
    _draw(win, win.label_id)             # box at (10, 10) 100x60
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

    _draw(win, win.label_id)
    win._update_box_count()
    QApplication.processEvents()          # the deferred open

    assert calls.get("opened") == 1
    assert win._awaiting_reference_box is False


def test_a_box_of_another_family_does_not_trigger_it(monkeypatch):
    import numpy as np
    from PySide6.QtWidgets import QApplication

    calls = _fake_editor(monkeypatch)
    win = _window()
    _define(win, "wf_capwrong", )
    win.set_active_label("wf_capwrong")
    win.last_raw = np.zeros((200, 400, 3), dtype=np.uint8)
    win.capture_reference()

    _draw(win, "some_other_label")
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
    _draw(win, win.label_id)
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
    _draw(win, win.label_id)
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
    win.canvas.boxes.clear(); _draw(win, win.label_id)
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
    win.canvas.boxes.clear(); _draw(win, win.label_id)

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
    win.canvas.boxes.clear(); _draw(win, win.label_id)

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
    win.canvas.boxes.clear(); _draw(win, win.label_id)

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
    win.canvas.boxes.clear(); _draw(win, win.label_id)
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
    win.canvas.boxes.clear(); _draw(win, win.label_id)
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


# --- filtering a large label list -------------------------------------------

def test_typing_narrows_the_label_list_and_says_by_how_much():
    """With hundreds of labels the list is unusable without this."""
    win = _window()
    for label_id in ("flt_warning_g31", "flt_spec_g31", "flt_spec_g27"):
        _define(win, label_id)

    win.label_search_edit.setText("")
    win._refresh_labels()
    everything = win.label_list.count()
    assert "label(s)" in win.label_count_label.text()

    win.label_search_edit.setText("flt_ g31")
    rows = [win.label_list.item(i).text() for i in range(win.label_list.count())]
    assert len(rows) == 2 and win.label_list.count() < everything
    assert all("g31" in r for r in rows)
    assert f"2 of {everything}" in win.label_count_label.text()

    win.label_search_edit.setText("")
    assert win.label_list.count() == everything


def test_every_typed_term_has_to_match_so_terms_narrow_each_other():
    win = _window()
    _define(win, "flt_combo_warn")
    _define(win, "flt_combo_spec")
    try:
        win.label_search_edit.setText("flt_combo")
        assert win.label_list.count() == 2
        win.label_search_edit.setText("flt_combo spec")
        rows = [win.label_list.item(i).text() for i in range(win.label_list.count())]
        assert rows and all("flt_combo_spec" in r for r in rows)
    finally:
        win.label_search_edit.setText("")


def test_a_query_matching_nothing_empties_the_list_rather_than_erroring():
    win = _window()
    _define(win, "flt_empty")
    try:
        win.label_search_edit.setText("no such label anywhere")
        assert win.label_list.count() == 0
        assert "0 of" in win.label_count_label.text()
    finally:
        win.label_search_edit.setText("")


def test_removing_a_label_deletes_its_images_too():
    """Keeping them was worse: the folder outlived the label, kept being
    measured, and kept being exported as a class nothing referenced."""
    from label_detections.core import persistence, storage

    win = _window()
    _define(win, "rm_wipes")
    win.set_active_label("rm_wipes")
    _capture(win, "rm_wipes", "a.jpg")
    _capture(win, "rm_wipes", "b.jpg")
    persistence.save_annotation("rm_wipes", "a.jpg", {"image": "a.jpg", "boxes": []})
    assert len(storage.list_images("rm_wipes")) == 2

    removed = win._delete_label_data("rm_wipes")
    assert removed >= 2
    assert storage.list_images("rm_wipes") == []
    assert persistence.load_annotation("rm_wipes", "a.jpg") is None


def test_a_dataset_with_no_library_row_is_not_exported():
    """A renamed or deleted label leaves a folder behind. Training it makes a
    class no recipe references, quietly competing with its replacement."""
    from label_detections.core import persistence, storage, yolo_export

    win = _window()
    _define(win, "orphan_keeper")
    # A library row alone is not a dataset -- it needs a folder to be listed.
    storage.dataset_folder("orphan_keeper").mkdir(parents=True, exist_ok=True)
    folder = storage.dataset_folder("orphan_ghost")
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "x.jpg"), np.zeros((40, 40, 3), np.uint8))

    datasets, orphans = yolo_export.exportable_datasets(persistence.load_library())
    assert "orphan_ghost" in orphans
    assert "orphan_ghost" not in datasets
    assert "orphan_keeper" in datasets


def test_the_orphan_filter_reads_the_library_from_disk_not_from_memory():
    """A window holds its library in memory and can be a label behind. A
    filter run against a stale copy drops a perfectly valid dataset."""
    from label_detections.core import persistence, storage, yolo_export
    from label_detections.core.labels import LabelDef, LabelLibrary

    win = _window()
    lib = persistence.load_library()
    lib.add(LabelDef(label_id="fresh_on_disk"), replace=True)
    persistence.save_library(lib)
    storage.dataset_folder("fresh_on_disk").mkdir(parents=True, exist_ok=True)

    # Caller passes an empty library, as a stale window would.
    datasets, _ = yolo_export.exportable_datasets(LabelLibrary([]))
    assert "fresh_on_disk" in datasets


def test_a_generic_detection_gets_the_open_labels_id():
    """Under a two-stage detector every box is class `label`, which is not an
    identity -- so pre-label wrote boxes with no label_id at all. The fallback
    to the open label is what this method is named after; it was removed when
    the detector started reporting ids and never restored for the pipeline
    that does not."""
    from label_detections.core import yolo_export

    win = _window()
    _define(win, "gen_fallback")
    win.set_active_label("gen_fallback")

    boxes = [{"label": yolo_export.GENERIC_CLASS,
              "points": [[10, 10], [50, 10], [50, 40], [10, 40]]}]
    out = win._assign_active_label(boxes)
    assert out[0]["label_id"] == "gen_fallback"


def test_a_per_label_detection_keeps_its_own_id():
    win = _window()
    _define(win, "own_id_a")
    _define(win, "own_id_b")
    win.set_active_label("own_id_a")

    boxes = [{"label": "own_id_b",
              "points": [[10, 10], [50, 10], [50, 40], [10, 40]]}]
    out = win._assign_active_label(boxes)
    assert out[0]["label_id"] == "own_id_b", "overwrote a real identity"


def test_a_box_drawn_as_something_else_is_not_relabelled():
    """A box the operator drew as something else is a statement that it IS
    something else. Overwriting it with the open label would make an image that
    does not carry this label look like it does -- and approve it."""
    win = _window()
    _define(win, "not_relabelled")
    win.set_active_label("not_relabelled")

    boxes = [{"label": "some_other_thing",
              "points": [[10, 10], [50, 10], [50, 40], [10, 40]]}]
    out = win._assign_active_label(boxes)
    assert "label_id" not in out[0]


def test_the_battery_face_never_gets_an_identity():
    win = _window()
    _define(win, "face_untouched")
    win.set_active_label("face_untouched")

    boxes = [{"label": "battery_side",
              "points": [[0, 0], [90, 0], [90, 90], [0, 90]]}]
    assert "label_id" not in win._assign_active_label(boxes)[0]


# --- the reference capture sits at the top of the list ---------------------

def test_the_reference_capture_leads_the_list_however_many_come_after_it(monkeypatch):
    """The reference is the shot every read-region on the label is a fraction
    of -- the one to go back to when a region looks wrong. It is also the first
    shot taken, so in a newest-first list it sinks a row further out of reach
    with every capture after it."""
    win = _window()
    source, _ = _with_artwork(win, "wf_ref_top", monkeypatch, name="a_first.jpg")
    later = [_capture(win, "wf_ref_top", f"z_later_{i}.jpg") for i in range(3)]

    from label_detections.core import persistence
    win.library = persistence.load_library()
    win._dataset_index_dirty = True
    order = win._get_dataset_image_paths()

    assert order[0] == source
    assert set(order[1:]) == set(later)


def test_a_label_with_no_artwork_keeps_the_plain_newest_first_order():
    win = _window()
    _define(win, "wf_ref_none")
    win.set_active_label("wf_ref_none")
    for name in ("a.jpg", "b.jpg", "c.jpg"):
        _capture(win, "wf_ref_none", name)

    win._dataset_index_dirty = True
    order = win._get_dataset_image_paths()
    assert [p.name for p in order] == ["c.jpg", "b.jpg", "a.jpg"]


def test_next_image_steps_in_the_order_the_list_shows(monkeypatch):
    """Ordering the visible list alone would leave N/P walking a different
    sequence than the rows under the cursor."""
    win = _window()
    source, _ = _with_artwork(win, "wf_ref_nav", monkeypatch, name="a_first.jpg")
    _capture(win, "wf_ref_nav", "z_later.jpg")

    from PySide6.QtCore import Qt
    from label_detections.core import persistence
    win.library = persistence.load_library()
    win._dataset_index_dirty = True
    win._refresh_images(force=True)
    rows = [win.image_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(win.image_list.count())]
    assert rows[0] == source.name

    win._load_image_path(source)
    win.next_image()
    assert win.current_image_path.name == rows[1]


# --- nothing but inference draws on a live frame ---------------------------

def test_saved_boxes_are_not_painted_over_a_live_frame():
    """Saved boxes are in some still's coordinates. A live frame is not that
    still, so painting them there marks pixels they were never about -- and it
    reads as a detection, because on a live view detections are what boxes are."""
    win = _window()
    _define(win, "wf_live_paint")
    win.set_active_label("wf_live_paint")
    _capture(win, "wf_live_paint", "still.jpg")
    _draw(win, win.label_id)
    assert win.canvas.annotations_painted() is True

    real = win.camera
    try:
        win.camera = _FakeCamera(True)
        win._refresh_live_mode()
        assert win.canvas.annotations_painted() is False
        # Hidden, not deleted: they are still the still's.
        assert len(win.canvas.boxes) == 1
    finally:
        win.camera = real
        win._refresh_live_mode()
    assert win.canvas.annotations_painted() is True


def test_hiding_saved_labels_is_still_honoured_on_a_still():
    """The live rule is on top of the operator's own toggle, not instead of it."""
    win = _window()
    win.canvas.set_annotation_visibility(False)
    try:
        assert win.canvas.annotations_painted() is False
    finally:
        win.canvas.set_annotation_visibility(True)


def test_opening_the_camera_drops_the_still_s_detections():
    """Overlays are a result computed for one image. The still is about to be
    replaced by a stream, and an overlay that outlives its frame reads as a
    detection on the frame it is sitting over."""
    win = _window()
    win.canvas.set_model_test_overlays([{"points": [[0, 0], [9, 0], [9, 9], [0, 9]],
                                         "name": "stale", "conf": 0.9}])
    real = win.camera
    try:
        win.camera = _FakeCamera(False)
        win.camera.open = lambda *a, **k: True
        win.camera.is_open = lambda: True
        win.camera.last_result = type("R", (), {"message": "ok"})()
        win.open_camera()
        assert win.canvas.model_test_overlays == []
    finally:
        win.timer.stop()
        win.camera = real
        win._refresh_live_mode()


# --- the way out of artwork drawn wrong is reachable -----------------------

def test_replacing_artwork_has_a_button_and_it_says_when_it_will_work(monkeypatch):
    """Redrawing badly-drawn artwork was signposted only in a tooltip, and only
    as an entry in a menu bar that is hidden. There was no visible way back."""
    win = _window()
    _define(win, "wf_replace_btn_none")
    win.set_active_label("wf_replace_btn_none")
    assert win.replace_artwork_btn.isEnabled() is False
    assert "no artwork" in win.replace_artwork_btn.toolTip()

    _with_artwork(win, "wf_replace_btn", monkeypatch, name="art.jpg")
    assert win.replace_artwork_btn.isEnabled() is True
    assert "Ctrl+Shift+A" in win.replace_artwork_btn.toolTip()


def test_nothing_tells_the_operator_to_open_a_menu_that_is_hidden():
    """The menu bar is hidden, so a message naming a menu path is a dead end.
    Name the button or the key instead."""
    source = (Path(__file__).resolve().parent.parent
              / "label_detections" / "ui" / "main_window.py").read_text()
    assert "Tools > " not in source


def test_every_action_with_a_shortcut_is_registered_on_the_window():
    """Qt does not dispatch the shortcut of an action that lives only inside a
    hidden menu bar, so an unregistered one is a key that does nothing."""
    from PySide6.QtWidgets import QMenu

    win = _window()
    registered = set(win.actions())
    menus = win.menuBar().findChildren(QMenu)
    assert menus, "no menus found -- the scan would pass vacuously"

    orphans = sorted({
        action.text()
        for menu in menus
        for action in menu.actions()
        if not action.shortcut().isEmpty() and action not in registered
    })
    assert orphans == []


# --- auto-label and the review queue actually finish -----------------------

class _FakeResults:
    """One Ultralytics result, enough for the overlay extractor."""

    def __init__(self, boxes, names, confs, classes):
        self.names = names
        self.obb = None

        class _Boxes:
            def __init__(self):
                self.xyxy = np.array(boxes, dtype=np.float32)
                self.conf = np.array(confs, dtype=np.float32)
                self.cls = np.array(classes, dtype=np.float32)
                self.id = None

        self.boxes = _Boxes()


def _stub_model(win, monkeypatch, label_id, count=2):
    """Make the model return `count` boxes of `label_id` on any image."""
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    results = [_FakeResults(
        [[10 + 40 * i, 10, 50 + 40 * i, 60] for i in range(count)],
        {0: label_id}, [0.9] * count, [0] * count)]
    monkeypatch.setattr(win, "_run_test_model_on_image",
                        lambda path: (frame, results, 0.0, 0.01))
    win.test_model_edit.setText("/some/model.pt")


def test_auto_label_reports_what_it_placed_instead_of_raising(monkeypatch):
    """The summary named three variables from the bung taxonomy that no longer
    existed, so the boxes landed and then the line describing them raised --
    every single run."""
    win = _window()
    _define(win, "wf_autolabel")
    win.set_active_label("wf_autolabel")
    _capture(win, "wf_autolabel", "auto.jpg")
    win.canvas.clear_boxes()
    _stub_model(win, monkeypatch, "wf_autolabel", count=2)

    win.auto_label_current()

    assert len(win.canvas.boxes) == 2
    message = win.status.currentMessage()
    assert "2 wf_autolabel" in message
    assert "bung" not in message.lower()


def test_the_review_queue_scores_the_result_it_was_handed(monkeypatch):
    """It scored on an undefined name, so the queue raised on the first image
    it looked at -- and nothing reached the ranking at all."""
    import label_detections.ui.main_window as mw_mod

    win = _window()
    _define(win, "wf_queue")
    win.set_active_label("wf_queue")
    for name in ("q1.jpg", "q2.jpg"):
        _capture(win, "wf_queue", name)
    win.canvas.clear_boxes()
    _stub_model(win, monkeypatch, "wf_queue", count=1)

    monkeypatch.setattr(mw_mod.QMessageBox, "question",
                        staticmethod(lambda *a, **k: mw_mod.QMessageBox.Yes))
    monkeypatch.setattr(mw_mod.QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(win, "next_in_review_queue", lambda: None)

    win.build_review_queue()
    assert len(win._review_queue) == 2


def test_marking_an_image_background_writes_a_reviewed_negative():
    """Background samples are how the model learns a bare fixture is not a
    label. The stamp they are written with did not exist, so the button raised
    and nothing was ever marked."""
    import json
    from label_detections.core import review as review_logic
    from label_detections.core import storage

    win = _window()
    _define(win, "wf_background")
    win.set_active_label("wf_background")
    image = _capture(win, "wf_background", "empty_fixture.jpg")
    win.canvas.clear_boxes()

    win.mark_current_background()

    data = json.loads(storage.image_label_json_path(image).read_text(encoding="utf-8"))
    assert data["background"] is True
    assert data["boxes"] == []
    # Reviewed, and by this tool -- an imported marker must not qualify.
    assert review_logic.annotation_reviewed(data) is True
    assert review_logic.annotation_status(data, "wf_background") == "background"


# --- one click to an oriented box ------------------------------------------

def _armed(win, label_id="wf_assist"):
    _define(win, label_id)
    win.set_active_label(label_id)
    image = _capture(win, label_id, "assist.jpg")
    win.canvas.clear_boxes()
    win.set_outline_assist(True)
    return image


class _FakeAssistant:
    """Stands in for the segmentation model: records the click, returns a quad."""

    def __init__(self, quad=None, why=""):
        self.quad = quad if quad is not None else [[10.0, 20.0], [110.0, 20.0],
                                                   [110.0, 80.0], [10.0, 80.0]]
        self.why = why
        self.calls = []

    def outline(self, frame, x, y, max_px):
        self.calls.append((float(x), float(y), int(max_px)))
        return ([], self.why) if self.why else (list(self.quad), "")


def test_a_click_while_armed_becomes_an_obb_carrying_the_label_id():
    win = _window()
    _armed(win, "wf_assist_ok")
    win._assistant = _FakeAssistant()

    win._outline_at(60.0, 50.0)

    assert len(win.canvas.boxes) == 1
    box = win.canvas.boxes[0]
    assert box.kind == "obb"
    assert box.points == [[10.0, 20.0], [110.0, 20.0], [110.0, 80.0], [10.0, 80.0]]
    # The identity comes with it: the recipe counts label ids, and re-typing
    # one per box is the work this feature exists to remove.
    assert box.label_id == "wf_assist_ok"
    assert win._assistant.calls[0][:2] == (60.0, 50.0)
    win.set_outline_assist(False)


def test_the_canvas_reports_the_click_instead_of_starting_a_drag():
    """The wiring the whole feature hangs on: while armed, a plain left-click
    is a request to outline, not the first corner of a drag."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    win = _window()
    _armed(win, "wf_assist_click")
    # Offscreen the window is never shown, so give the canvas a size and aim at
    # the middle of where the image is actually drawn.
    win.canvas.resize(400, 300)
    win.canvas.fit_to_window()
    target = win.canvas._target_rect()
    centre = QPointF(target.center().x(), target.center().y())

    seen = []
    win.canvas.assist_requested.connect(lambda x, y: seen.append((x, y)))
    try:
        event = QMouseEvent(QMouseEvent.Type.MouseButtonPress, centre,
                            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        win.canvas.mousePressEvent(event)
        assert len(seen) == 1
        # The click reported image coordinates, not screen ones, and did not
        # begin a drag.
        assert seen[0] == pytest.approx((200.0, 100.0), abs=2)
        assert win.canvas.drawing is False
    finally:
        win.set_outline_assist(False)


def test_a_refused_outline_says_why_and_stays_armed():
    """The answer to a bad click is another click. Dropping the mode would make
    the operator re-arm to retry."""
    win = _window()
    _armed(win, "wf_assist_bad")
    win._assistant = _FakeAssistant(why="That outlined most of the frame")

    win._outline_at(5.0, 5.0)

    assert win.canvas.boxes == []
    assert "most of the frame" in win.status.currentMessage()
    assert win.canvas.assist_mode is True
    win.set_outline_assist(False)


def test_the_outline_is_one_undo_step():
    win = _window()
    _armed(win, "wf_assist_undo")
    win._assistant = _FakeAssistant()

    win._outline_at(60.0, 50.0)
    assert len(win.canvas.boxes) == 1
    win.canvas.undo()
    assert win.canvas.boxes == []
    win.set_outline_assist(False)


def test_opening_a_camera_drops_the_armed_mode():
    """Clicks never reach the assistant in live mode, so a lit button would be
    a mode that looks armed and is not."""
    win = _window()
    _armed(win, "wf_assist_live")
    real = win.camera
    try:
        win.camera = _FakeCamera(True)
        win._refresh_live_mode()
        assert win.canvas.assist_mode is False
        assert win.assist_btn.isChecked() is False
    finally:
        win.camera = real
        win._refresh_live_mode()


def test_the_first_outline_asks_before_it_may_download(monkeypatch):
    """The load is inline, so a checkpoint that is not on disk yet freezes the
    window while tens of megabytes arrive. I hit exactly that writing this: a
    click with no stand-in reached the real model and hung the test run."""
    import label_detections.ui.main_window as mw_mod
    from label_detections.core.storage import save_test_settings, load_test_settings

    win = _window()
    _armed(win, "wf_assist_ask")
    win._assistant = None
    win._assist_confirmed = False
    settings = dict(load_test_settings() or {})
    save_test_settings({**settings, "assist_confirmed": False})

    asked = {}
    monkeypatch.setattr(
        mw_mod.QMessageBox, "question",
        staticmethod(lambda parent, title, text, *a, **k: (
            asked.update({"text": text}) or mw_mod.QMessageBox.No)))
    try:
        win._outline_at(60.0, 50.0)
        # Declined: nothing loaded, nothing drawn, and the mode is off rather
        # than armed and silently doing nothing on every click.
        assert win._assistant is None
        assert win.canvas.boxes == []
        assert win.canvas.assist_mode is False
        assert "downloads now" in asked.get("text", "")
    finally:
        save_test_settings(settings)
        win.set_outline_assist(False)


def test_it_does_not_ask_again_once_confirmed(monkeypatch):
    import label_detections.ui.main_window as mw_mod
    from label_detections.core.storage import save_test_settings, load_test_settings

    win = _window()
    settings = dict(load_test_settings() or {})
    save_test_settings({**settings, "assist_confirmed": True})
    monkeypatch.setattr(
        mw_mod.QMessageBox, "question",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("asked again after it was confirmed"))))
    try:
        assert win._confirm_assist_download() is True
    finally:
        save_test_settings(settings)


def test_the_assist_turns_itself_off_once_a_box_lands():
    """The next thing anyone does after an outline is check it, and checking it
    means dragging a corner -- which an armed canvas would swallow as another
    outline request."""
    win = _window()
    _armed(win, "wf_assist_oneshot")
    win._assistant = _FakeAssistant()

    win._outline_at(60.0, 50.0)

    assert len(win.canvas.boxes) == 1
    assert win.canvas.assist_mode is False
    assert win.assist_btn.isChecked() is False
    # And the canvas takes handle drags again rather than reporting clicks.
    assert win.canvas.drawing_enabled is True


def test_the_outline_model_can_be_changed_from_the_labeling_panel():
    """It was only reachable by editing a settings file, which is the same as
    not being reachable."""
    from label_detections.ui.segment_assist import DEFAULT_MODEL

    win = _window()
    assert win._assist_model_name() == DEFAULT_MODEL

    index = win.assist_model_combo.findData("sam2.1_t.pt")
    assert index >= 0, "the picker should offer SAM 2.1 tiny"
    win._assistant = object()                     # something is loaded
    win.assist_model_combo.setCurrentIndex(index)

    assert win._assist_model_name() == "sam2.1_t.pt"
    # The loaded model is dropped, or the next click would use the old one.
    assert win._assistant is None
    win.assist_model_combo.setCurrentIndex(
        win.assist_model_combo.findData(DEFAULT_MODEL))


def test_a_typed_checkpoint_path_is_taken_as_given():
    """The list is what is worth suggesting, not what is allowed."""
    from label_detections.ui.segment_assist import DEFAULT_MODEL

    win = _window()
    try:
        win.assist_model_combo.setEditText(r"C:\models\my_sam.pt")
        assert win._assist_model_name() == r"C:\models\my_sam.pt"
    finally:
        win.assist_model_combo.setCurrentIndex(
            win.assist_model_combo.findData(DEFAULT_MODEL))


def test_the_selected_image_row_is_filled_not_marked_with_a_sliver():
    """Styling QListWidget at all turns off the native selection paint. Without
    an explicit rule the current row showed as a sliver at its left edge, which
    in a long list is invisible -- and saying which file is being edited is what
    the list is for."""
    win = _window()
    style = win.styleSheet()
    assert "QListWidget::item:selected" in style
    assert "QListWidget::item:selected:!active" in style, (
        "an unfocused list must stay highlighted -- the canvas has focus while "
        "you label, not the list")


def test_the_current_image_is_the_selected_row():
    win = _window()
    _define(win, "wf_row_select")
    win.set_active_label("wf_row_select")
    first = _capture(win, "wf_row_select", "row_a.jpg")
    second = _capture(win, "wf_row_select", "row_b.jpg")
    win._refresh_images(force=True)

    win._load_image_path(first)
    win._select_image_in_list()
    chosen = win.image_list.currentItem()
    assert chosen is not None
    assert win._image_name_from_list_item(chosen) == first.name
    assert chosen.isSelected() is True

    win._load_image_path(second)
    win._select_image_in_list()
    assert win._image_name_from_list_item(win.image_list.currentItem()) == second.name
