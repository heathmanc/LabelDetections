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
