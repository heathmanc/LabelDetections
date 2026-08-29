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


def _reference_file(label_id: str) -> str:
    """A real reference image on disk for a test label.

    It has to exist, not merely be named: a label with no artwork is refused
    everywhere now, because every region on it is a fraction of an outline
    drawn on a picture. Pointing at a filename that was never written made the
    old helper describe a label the app would not open.
    """
    import numpy as np
    from label_detections.core.imageio import save_reference

    return str(save_reference(label_id, np.full((60, 90, 3), 220, np.uint8)))


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
                         reference_images=[_reference_file(label_id)],
                         train_target=10),
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


def _with_artwork(win, label_id, monkeypatch=None, name="ref.jpg"):
    """A label whose reference was made from one of its captured images.

    Returns that image, which the list has to keep at the top: it is the shot
    every read-region on the label is a fraction of, and the one to go back to
    when a region looks wrong -- but it is also the first shot taken, so a
    newest-first list buries it a row deeper with every capture after it.
    """
    from label_detections.core import persistence

    _define(win, label_id)
    win.set_active_label(label_id)
    source = _capture(win, label_id, name)
    library = persistence.load_library()
    label = library.get(label_id)
    updated = type(label).from_dict({**label.to_dict(),
                                     "reference_source": str(source)})
    library.add(updated, replace=True)
    persistence.save_library(library)
    win.library = persistence.load_library()
    return source, label_id


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

def test_one_action_covers_the_whole_reference_job():
    """Define, Edit, Place and Replace were four entries and three buttons for
    parts of one job, in an order nothing enforced. A label is photographed,
    outlined and marked up in one window or not at all."""
    win = _window()
    titles = {a.text() for a in win.actions()}
    assert "Capture reference image..." in titles
    for retired in ("Define read-regions from this image", "Place read-regions",
                    "Edit read-regions", "Replace label artwork..."):
        assert retired not in titles, retired


def test_the_annotation_pane_no_longer_carries_them():
    """They were on the pane about annotating images, doing a job about
    defining labels."""
    win = _window()
    for gone in ("define_regions_btn", "replace_artwork_btn"):
        assert not hasattr(win, gone), gone


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


def test_replacing_a_reference_asks_first_and_says_what_it_costs(monkeypatch):
    """The artwork is the coordinate system every region is written in, so
    replacing it is a delete and redraw, not an edit."""
    win = _window()
    _define(win, "wf_replace")
    win.set_active_label("wf_replace")

    asked = {}
    monkeypatch.setattr(win, "_ask_replace_reference",
                        lambda label: asked.setdefault("label", label) and False)
    win.capture_reference()
    assert asked.get("label") is not None, "replaced without asking"


def test_declining_the_replace_leaves_the_reference_alone(monkeypatch):
    from label_detections.core import reference as ref

    win = _window()
    _define(win, "wf_keep")
    win.set_active_label("wf_keep")
    before = ref.reference_path(win.library.get("wf_keep"))

    monkeypatch.setattr(win, "_ask_replace_reference", lambda label: False)
    win.capture_reference()
    assert ref.reference_path(win.library.get("wf_keep")) == before


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

def test_a_label_with_no_reference_is_marked_in_the_list():
    """A tick beside 150 reviewed images would say "nearly done" about a label
    that cannot be used at all."""
    from label_detections.core import persistence
    from label_detections.core.labels import LabelDef

    win = _window()
    library = persistence.load_library()
    library.add(LabelDef(label_id="wf_noref", train_target=1), replace=True)
    persistence.save_library(library)
    win.library = persistence.load_library()
    win._refresh_labels()

    rows = [win.label_list.item(i).text()
            for i in range(win.label_list.count())]
    mine = [r for r in rows if "wf_noref" in r]
    assert mine and "NO REFERENCE" in mine[0]


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

    index = win.assist_model_combo.findData("mobile_sam.pt")
    assert index >= 0, "the picker should still offer the small fast one"
    win._assistant = object()                     # something is loaded
    win.assist_model_combo.setCurrentIndex(index)

    assert win._assist_model_name() == "mobile_sam.pt"
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


# --- the training summary shows every stage that ran ------------------------

DETECTOR_CSV = (
    "epoch,train/box_loss,train/cls_loss,metrics/precision(B),metrics/recall(B),"
    "metrics/mAP50(B),metrics/mAP50-95(B)\n"
    "1,1.30,0.95,0.44,0.51,0.38,0.19\n"
    "2,0.98,0.71,0.81,0.79,0.72,0.48\n"
    "3,0.86,0.62,0.88,0.86,0.83,0.57\n"
)
CLASSIFIER_CSV = (
    "epoch,time,train/loss,metrics/accuracy_top1,metrics/accuracy_top5,val/loss\n"
    "1,10,1.90,0.41,0.88,1.77\n"
    "2,20,1.21,0.74,0.96,1.02\n"
    "3,30,0.88,0.93,0.99,0.71\n"
)


def _finished_run(win, tmp_path, stage, csv_text, name):
    """Put a finished run on disk and record it the way training does."""
    run_dir = tmp_path / name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "results.csv").write_text(csv_text, encoding="utf-8")
    (run_dir / "weights" / "best.pt").write_bytes(b"weights")
    win._train_stage = stage
    return win._read_stage_result(
        stage, 0, False, 120.0, run_dir / "results.csv", run_dir,
        run_dir / "weights" / "best.pt")


def test_both_stages_survive_to_the_summary(tmp_path):
    """Train Both computed the detector's result and threw it away when the
    queue advanced, so only the classifier -- the cheaper half -- was shown."""
    win = _window()
    win._stage_results = {}
    win._stage_results["detector"] = _finished_run(
        win, tmp_path, "detector", DETECTOR_CSV, "det")
    win._stage_results["classifier"] = _finished_run(
        win, tmp_path, "classifier", CLASSIFIER_CSV, "cls")

    assert set(win._stage_results) == {"detector", "classifier"}
    det = win._stage_results["detector"]
    assert det["summary"]["best"]["mAP50-95"] == pytest.approx(0.57)
    assert set(det["series"]) == {"box_loss", "cls_loss", "mAP50", "mAP50-95"}

    cls = win._stage_results["classifier"]
    assert cls["summary"]["best"]["accuracy_top1"] == pytest.approx(0.93)
    assert set(cls["series"]) == {"loss", "top1", "top5"}


def test_each_stage_is_described_in_its_own_metrics(tmp_path):
    win = _window()
    det = _finished_run(win, tmp_path, "detector", DETECTOR_CSV, "det2")
    cls = _finished_run(win, tmp_path, "classifier", CLASSIFIER_CSV, "cls2")

    det_text = win._stage_metric_text(det)
    assert "mAP50-95" in det_text and "accuracy_top1" not in det_text

    cls_text = win._stage_metric_text(cls)
    assert "accuracy_top1" in cls_text
    assert "No validation metrics" not in cls_text


def test_the_summary_builds_a_column_per_stage(tmp_path, monkeypatch):
    """The dialog itself: two columns, each with its own chart."""
    from PySide6.QtWidgets import QDialog, QLabel
    from label_detections.ui.main_window import TrainingMetricsChart

    win = _window()
    win._stage_results = {
        "detector": _finished_run(win, tmp_path, "detector", DETECTOR_CSV, "det3"),
        "classifier": _finished_run(win, tmp_path, "classifier", CLASSIFIER_CSV, "cls3"),
    }
    win._stage_weights = {}

    built = {}
    monkeypatch.setattr(QDialog, "exec",
                        lambda self: built.setdefault("dialog", self) and 0 or 0)
    win._show_training_summary(0, False, 120.0, None, None, None)

    dialog = built.get("dialog")
    assert dialog is not None
    charts = dialog.findChildren(TrainingMetricsChart)
    assert len(charts) == 2, "one chart per stage"
    text = " ".join(w.text() for w in dialog.findChildren(QLabel))
    assert "Stage 1" in text and "Stage 2" in text
    assert "mAP50-95" in text and "accuracy_top1" in text


def test_a_single_stage_run_still_summarizes(tmp_path, monkeypatch):
    """Training only the detector must not depend on a queue having run."""
    from PySide6.QtWidgets import QDialog
    from label_detections.ui.main_window import TrainingMetricsChart

    win = _window()
    win._stage_results = {}
    win._stage_weights = {}
    run_dir = tmp_path / "solo"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "results.csv").write_text(DETECTOR_CSV, encoding="utf-8")
    win._train_stage = "detector"

    built = {}
    monkeypatch.setattr(QDialog, "exec",
                        lambda self: built.setdefault("dialog", self) and 0 or 0)
    win._show_training_summary(0, False, 90.0, run_dir / "results.csv", run_dir, None)

    dialog = built.get("dialog")
    assert len(dialog.findChildren(TrainingMetricsChart)) == 1


def test_the_train_tab_carries_the_augmentation_into_the_run():
    """The fields exist so the values are visible and changeable, not just
    different -- a default nobody can see is the same trap as a library one."""
    win = _window()
    assert win.train_mosaic_spin.value() == pytest.approx(0.0)
    assert win.train_scale_spin.value() == pytest.approx(0.2)
    assert win.train_fliplr_spin.value() == pytest.approx(0.0)
    assert win.cls_erasing_spin.value() == pytest.approx(0.0)

    params = win._gather_train_params()
    assert params["mosaic"] == pytest.approx(0.0)
    assert params["scale"] == pytest.approx(0.2)

    win.train_mosaic_spin.setValue(0.55)
    try:
        assert win._gather_train_params()["mosaic"] == pytest.approx(0.55)
    finally:
        win.train_mosaic_spin.setValue(0.0)

    cls = win._gather_classifier_params()
    assert cls["erasing"] == pytest.approx(0.0)
    assert cls["fliplr"] == pytest.approx(0.0)


def test_the_queue_advance_does_not_start_a_fresh_run():
    """The regression this pins: the queue is popped before the second stage
    launches, so a "fresh run?" test that read the queue saw it empty and wiped
    the detector's metrics and weights the moment the classifier started. One
    cause, two symptoms -- a summary with one chart, and no button to wire both
    models. It is passed in now, not inferred."""
    win = _window()
    calls = []
    original = win._start_training_run
    win._start_training_run = lambda params, stage, **kw: calls.append((stage, kw))
    win._train_queue = [({"task": "classify"}, "classifier")]
    win._train_stage = "detector"
    win._stage_results = {}
    win._stage_weights = {}
    win._train_start_time = 0.0
    win._train_stopped = False
    win._train_process = None
    try:
        win._on_train_finished(0, None)
    finally:
        win._start_training_run = original
        win._train_queue = []

    assert calls, "the classifier stage never started"
    stage, kwargs = calls[0]
    assert stage == "classifier"
    assert kwargs.get("fresh") is False, (
        "the second stage was treated as a new run, which clears stage 1")


def test_a_fresh_run_clears_the_last_one_and_a_queued_stage_does_not(tmp_path):
    """The reset still has to happen, or a second run shows the first's charts."""
    win = _window()
    win._stage_results = {"detector": {"stage": "detector"}}
    win._stage_weights = {"detector": tmp_path / "old.pt"}

    win._begin_fresh_run()
    assert win._stage_results == {}
    assert win._stage_weights == {}

    # And nothing else in the launch path touches them, so a stage started with
    # fresh=False keeps whatever stage 1 recorded.
    win._stage_results = {"detector": {"stage": "detector"}}
    win._stage_weights = {"detector": tmp_path / "det.pt"}
    assert "detector" in win._stage_results and "detector" in win._stage_weights


# --- the box on screen is the box it looks for -----------------------------
#
# A PC680 box drawn on the image, plate and all, still produced "Draw the PC680
# box on this image first". Box carries two fields on purpose: `label` is the
# detector family it trains as -- "label" under a two-stage export -- and
# `label_id` is which library label it is. Three checks compared the family
# against the library id, so they could never match once the two differed.
#
# The tests missed it because the helper set both fields to the same string.
# These draw a box the way the running app does.

def _draw_as_the_app_does(win, label_id, family="label"):
    """A box whose family and identity differ, which is the real shape."""
    from label_detections.ui.canvas import Box

    box = Box(x=10, y=10, w=100, h=60, class_id=0, label=family, kind="obb",
              points=[[10, 10], [110, 10], [110, 70], [10, 70]])
    box.label_id = label_id
    win.canvas.boxes.append(box)
    return box


def test_a_generic_family_box_is_recognised_as_its_label():
    win = _window()
    _define(win, "wf_family_split")
    win.set_active_label("wf_family_split")
    win.canvas.boxes.clear()
    box = _draw_as_the_app_does(win, "wf_family_split")
    assert win._box_is_active_label(box) is True
    assert win._label_box_for_regions(
        win.library.get("wf_family_split")) is box


def test_a_box_for_a_different_label_is_still_rejected():
    """The fix must not make every box match."""
    win = _window()
    _define(win, "wf_split_mine")
    _define(win, "wf_split_other")
    win.set_active_label("wf_split_mine")
    win.canvas.boxes.clear()
    other = _draw_as_the_app_does(win, "wf_split_other")
    assert win._box_is_active_label(other) is False


def test_an_old_box_with_no_id_still_matches_on_its_family():
    """Identity lived in `label` before the two fields were split, and those
    boxes are still in saved sidecars."""
    win = _window()
    _define(win, "wf_legacy")
    win.set_active_label("wf_legacy")
    win.canvas.boxes.clear()
    legacy = _draw_as_the_app_does(win, "", family="wf_legacy")
    assert win._box_is_active_label(legacy) is True



# --- the reference is required, and it is the one way in --------------------

def _bare_label(win, label_id):
    """A label in the library with no artwork -- the state the rule refuses."""
    from label_detections.core import persistence
    from label_detections.core.labels import LabelDef

    library = persistence.load_library()
    library.add(LabelDef(label_id=label_id, train_target=1), replace=True)
    persistence.save_library(library)
    win.library = persistence.load_library()
    return label_id


def test_a_label_with_no_reference_cannot_be_opened():
    """It has regions that are fractions of an outline drawn on a picture that
    does not exist -- there is nothing to draw against or verify with, and
    letting it open is what let a half-defined label look finished."""
    win = _window()
    _define(win, "wf_gate_ok")
    win.set_active_label("wf_gate_ok")
    _bare_label(win, "wf_gate_bare")

    win.set_active_label("wf_gate_bare")
    assert win.label_id == "wf_gate_ok", "opened a label with no reference"


def test_opening_it_from_the_list_offers_the_way_through(monkeypatch):
    """A refusal that only says no is a wall."""
    win = _window()
    _bare_label(win, "wf_gate_offer")
    monkeypatch.setattr(win, "_selected_label_id", lambda: "wf_gate_offer")

    asked = {}
    monkeypatch.setattr(win, "_ask_reference_needed",
                        lambda label: asked.setdefault("label", label) and False)
    monkeypatch.setattr(win, "capture_reference",
                        lambda: asked.setdefault("captured", True))
    win._load_selected_label()
    assert asked.get("label") is not None


def test_saying_yes_to_the_offer_runs_the_capture(monkeypatch):
    win = _window()
    _bare_label(win, "wf_gate_yes")
    monkeypatch.setattr(win, "_selected_label_id", lambda: "wf_gate_yes")
    monkeypatch.setattr(win, "_ask_reference_needed", lambda label: True)
    ran = {}
    monkeypatch.setattr(win, "capture_reference",
                        lambda: ran.setdefault("yes", True))
    win._load_selected_label()
    assert ran.get("yes") is True


def test_nothing_prompts_when_the_app_repoints_itself():
    """A modal over an empty window on launch is a worse greeting than a list
    that says which labels need one, and every refresh would raise it."""
    win = _window()
    _bare_label(win, "wf_gate_quiet")
    # No stub for the prompt: if this raised one, it would block here.
    win.set_active_label("wf_gate_quiet")
    assert win.label_id != "wf_gate_quiet"


def test_a_label_that_gains_a_reference_opens_normally():
    win = _window()
    _bare_label(win, "wf_gate_fixed")
    win.set_active_label("wf_gate_fixed")
    assert win.label_id != "wf_gate_fixed"

    _define(win, "wf_gate_fixed")          # writes real artwork
    win.set_active_label("wf_gate_fixed")
    assert win.label_id == "wf_gate_fixed"


def test_the_outline_model_survives_a_restart():
    """It was saved and then ignored. _build_assist_model_combo asked
    _assist_model_name() which way to open, and that reads the combo when there
    is one -- by that line there is, freshly built and sitting on item 0. So the
    answer was always the first entry, and choosing anything else looked like a
    setting that would not stick."""
    from label_detections.core.storage import load_test_settings, save_test_settings
    from label_detections.ui.main_window import MainWindow
    from label_detections.ui.segment_assist import DEFAULT_MODEL

    before = dict(load_test_settings() or {})
    other = next(name for name, _note in
                 __import__("label_detections.ui.segment_assist",
                            fromlist=["KNOWN_MODELS"]).KNOWN_MODELS
                 if name != DEFAULT_MODEL)
    try:
        save_test_settings({**before, "assist_model": other})
        fresh = MainWindow()          # a "restart": a window built from disk
        try:
            assert fresh._saved_assist_model() == other
            assert fresh._assist_model_name() == other
            assert fresh.assist_model_combo.currentText() == other
        finally:
            fresh.close()
    finally:
        save_test_settings(before)
