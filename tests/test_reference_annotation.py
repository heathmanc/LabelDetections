"""The reference capture is an annotated image, and the library is one truth.

Two things that made no sense from the outside:

The photograph a reference is flattened out of had no sidecar. It is the one
image in the dataset whose label position is known rather than guessed --
somebody drew it, corner by corner -- and it sat in the list as "unlabeled"
next to frames the model had a go at.

And a region added to the reference only appeared on images labelled after it.
Regions are attached to a box when the box is labelled, out of whatever the
library held at that moment, so a dataset ended up with the field box on some
images and not on others for a reason invisible in either the images or the
library.
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
                      tempfile.mkdtemp(prefix="labelvision-refann-"))

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

# A wide label sitting at an angle in a wider frame.
QUAD = [[120.0, 140.0], [880.0, 180.0], [872.0, 380.0], [112.0, 340.0]]
FIELD = {"name": "part_number", "policy": "must_be_present",
         "pattern": "PC680", "region": [0.70, 0.20, 0.22, 0.55]}

_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


def _label(win, label_id):
    from label_detections.core import persistence
    from label_detections.core.labels import LabelDef

    library = persistence.load_library()
    library.add(LabelDef(label_id=label_id, train_target=10), replace=True)
    persistence.save_library(library)
    win.library = persistence.load_library()
    win.label_id = label_id
    return win.library.get(label_id)


def _frame():
    image = np.full((520, 1000, 3), 200, np.uint8)
    cv2.fillPoly(image, [np.array(QUAD, np.int32)], (40, 40, 190))
    return image


def _result(**over):
    out = {"artwork": np.full((150, 500, 3), 220, np.uint8),
           "frame": _frame(),
           "source_path": "",
           "quad": [list(p) for p in QUAD],
           "codes": [],
           "text_fields": [dict(FIELD)],
           "anchor_region": []}
    out.update(over)
    return out


def _capture(label_id, name):
    from label_detections.core import storage
    folder = storage.dataset_folder(label_id)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    cv2.imwrite(str(path), _frame())
    return path


def _annotate(label_id, path, quad=QUAD):
    """An image labelled with a box but no regions -- the state every image
    labelled before a field was added is in."""
    from label_detections.core import annotations as ann
    from label_detections.core import persistence

    data = ann.new_annotation(path.name, label_id, 1000, 520)
    data["boxes"].append(ann.make_box("spec_plate", [list(p) for p in quad],
                                      label_id=label_id))
    persistence.save_annotation(label_id, path, data)
    return data


# --- the reference capture gets a sidecar ------------------------------------

def test_saving_a_reference_labels_the_photograph_it_came_from():
    from label_detections.core import annotations as ann
    from label_detections.core import persistence

    win = _window()
    label = _label(win, "refann_photo")
    win._save_reference(label, _result())

    saved = persistence.load_library().get("refann_photo")
    source = Path(saved.reference_source)
    assert source.is_file(), "the capture itself must be kept"

    data = persistence.load_annotation("refann_photo", source)
    assert data is not None, "the reference image must have a sidecar"
    boxes = ann.boxes_for(data, "refann_photo")
    assert len(boxes) == 1
    assert [[round(x), round(y)] for x, y in ann.box_polygon(boxes[0])] == \
        [[round(x), round(y)] for x, y in QUAD]


def test_the_reference_box_carries_the_regions_that_were_just_drawn():
    from label_detections.core import annotations as ann
    from label_detections.core import persistence

    win = _window()
    label = _label(win, "refann_regions")
    win._save_reference(label, _result())

    saved = persistence.load_library().get("refann_regions")
    data = persistence.load_annotation("refann_regions", Path(saved.reference_source))
    box = ann.boxes_for(data, "refann_regions")[0]
    placed = ann.regions(box, "text")
    assert [r["field"] for r in placed] == ["part_number"]


def test_the_reference_capture_is_not_approved_on_the_operators_behalf():
    """It knows where one label is, which is not the same as saying every
    label in the frame is boxed. Approval is still the operator's to give."""
    from label_detections.core import persistence, review

    win = _window()
    label = _label(win, "refann_review")
    win._save_reference(label, _result())

    saved = persistence.load_library().get("refann_review")
    data = persistence.load_annotation("refann_review", Path(saved.reference_source))
    assert review.annotation_status(data, "refann_review") == "needs_review"


def test_an_image_that_was_already_labelled_keeps_its_own_outline():
    """A reference can be made from a frame already in the dataset. The outline
    drawn there may be better than the one this window took, and it is what
    every region on that image is already positioned against."""
    from label_detections.core import annotations as ann
    from label_detections.core import persistence

    win = _window()
    label = _label(win, "refann_existing")
    path = _capture("refann_existing", "already.jpg")
    drawn = [[100.0, 100.0], [900.0, 100.0], [900.0, 300.0], [100.0, 300.0]]
    _annotate("refann_existing", path, drawn)

    win._save_reference(label, _result(source_path=str(path)))

    data = persistence.load_annotation("refann_existing", path)
    boxes = ann.boxes_for(data, "refann_existing")
    assert len(boxes) == 1, "no second box for the same label"
    assert ann.box_polygon(boxes[0]) == drawn
    assert ann.regions(boxes[0], "text"), "but its regions are brought up to date"


def test_the_artworks_shape_is_recorded_with_it():
    """Which is what stops a label photographed standing up getting its regions
    a quarter turn out."""
    from label_detections.core import persistence

    win = _window()
    label = _label(win, "refann_aspect")
    win._save_reference(label, _result(artwork=np.full((150, 600, 3), 220, np.uint8)))

    saved = persistence.load_library().get("refann_aspect")
    assert round(saved.reference_aspect, 3) == 4.0


# --- the dataset is brought up to date ---------------------------------------

def test_images_labelled_before_a_field_existed_get_it_too():
    """The whole complaint: some images have the field box and some don't."""
    from label_detections.core import annotations as ann
    from label_detections.core import persistence

    win = _window()
    label = _label(win, "refann_backfill")
    old = [_capture("refann_backfill", f"old_{i}.jpg") for i in range(3)]
    for path in old:
        _annotate("refann_backfill", path)
    for path in old:
        data = persistence.load_annotation("refann_backfill", path)
        assert not ann.regions(ann.boxes_for(data, "refann_backfill")[0])

    win._save_reference(label, _result())

    for path in old:
        data = persistence.load_annotation("refann_backfill", path)
        box = ann.boxes_for(data, "refann_backfill")[0]
        assert [r["field"] for r in ann.regions(box, "text")] == ["part_number"], \
            f"{path.name} was left behind"


def test_a_region_that_moved_on_the_artwork_moves_on_the_images():
    """Replaced, not merged. A region that only ever gets added leaves the old
    position sitting there -- which is how a deleted region survived a redraw."""
    from label_detections.core import annotations as ann
    from label_detections.core import persistence

    win = _window()
    label = _label(win, "refann_moved")
    path = _capture("refann_moved", "one.jpg")
    _annotate("refann_moved", path)

    win._save_reference(label, _result())
    first = ann.regions(ann.boxes_for(
        persistence.load_annotation("refann_moved", path), "refann_moved")[0], "text")

    moved = dict(FIELD, region=[0.05, 0.20, 0.22, 0.55])
    label = persistence.load_library().get("refann_moved")
    win._save_reference(label, _result(text_fields=[moved]))
    second = ann.regions(ann.boxes_for(
        persistence.load_annotation("refann_moved", path), "refann_moved")[0], "text")

    assert len(second) == 1, "one field means one region, not two"
    assert second[0]["points"] != first[0]["points"]


def test_boxes_of_other_labels_are_not_touched():
    from label_detections.core import annotations as ann
    from label_detections.core import persistence

    win = _window()
    label = _label(win, "refann_others")
    path = _capture("refann_others", "mixed.jpg")
    data = _annotate("refann_others", path)
    data["boxes"].append(ann.make_box("spec_plate", [[10, 10], [90, 10], [90, 60], [10, 60]],
                                      label_id="somebody_else"))
    persistence.save_annotation("refann_others", path, data)

    win._save_reference(label, _result())

    after = persistence.load_annotation("refann_others", path)
    stranger = ann.boxes_for(after, "somebody_else")[0]
    assert not ann.regions(stranger)


def test_refreshing_reports_what_it_changed():
    """So a caller can say so, and so a second run is visibly a no-op."""
    from label_detections.core import annotations as ann
    from label_detections.core.labels import LabelDef, TextField

    label = LabelDef(label_id="refann_count", reference_aspect=3.3)
    label.text_fields = [TextField(**FIELD)]
    data = ann.new_annotation("x.jpg", "refann_count", 1000, 520)
    data["boxes"].append(ann.make_box("spec_plate", [list(p) for p in QUAD],
                                      label_id="refann_count"))

    assert ann.refresh_reference_regions(data, label) == 1
    assert ann.refresh_reference_regions(data, label) == 0


def test_opening_an_image_places_its_regions_from_the_library():
    """The regions on an image are derived from the library, not decisions
    stored on the image -- so a sidecar written before a field existed shows
    the field the moment it is opened, without anything being re-run."""
    win = _window()
    label = _label(win, "refann_onload")
    win._save_reference(label, _result())

    late = _capture("refann_onload", "late.jpg")
    _annotate("refann_onload", late)          # a box, no regions
    win._dataset_index_dirty = True
    win._load_image_path(late)

    box = [b for b in win.canvas._snapshot_boxes() if b.get("label_id") == "refann_onload"][0]
    assert [r["field"] for r in box.get("regions", [])] == ["part_number"]


def test_opening_an_image_leaves_a_stranger_alone():
    """A box whose label is not in the library has nothing to be placed from,
    and inventing regions for it would be worse than showing none."""
    from label_detections.core import annotations as ann
    from label_detections.core import persistence

    win = _window()
    label = _label(win, "refann_onload_other")
    win._save_reference(label, _result())

    path = _capture("refann_onload_other", "mixed.jpg")
    data = _annotate("refann_onload_other", path)
    data["boxes"].append(ann.make_box("spec_plate", [[10, 10], [90, 10], [90, 60], [10, 60]],
                                      label_id="not_in_the_library"))
    persistence.save_annotation("refann_onload_other", path, data)
    win._load_image_path(path)

    stranger = [b for b in win.canvas._snapshot_boxes()
                if b.get("label_id") == "not_in_the_library"][0]
    assert not stranger.get("regions")


# --- orientation resolved by looking at the label ----------------------------

def _asymmetric():
    """Artwork with a clearly different end from the other, so which way up it
    was presented is answerable from the pixels alone."""
    art = np.full((90, 300, 3), 235, np.uint8)
    art[:, :90] = (30, 30, 30)
    cv2.putText(art, "PC680", (110, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    return art


def _photo_of(art, turns):
    label = art.copy()
    for _ in range(turns % 4):
        label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
    h, w = label.shape[:2]
    frame = np.full((h + 200, w + 200, 3), 175, np.uint8)
    frame[100:100 + h, 100:100 + w] = label
    quad = [[100.0, 100.0], [100.0 + w, 100.0],
            [100.0 + w, 100.0 + h], [100.0, 100.0 + h]]
    return frame, quad


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_the_window_works_out_which_way_up_from_the_artwork(turns):
    """Not from the detector's angle, which reports the same number for all
    four of these, and not from reading, which the label may not support."""
    win = _window()
    label_id = f"refann_turn_{turns}"
    label = _label(win, label_id)
    art = _asymmetric()
    win._save_reference(label, _result(artwork=art, frame=_photo_of(art, 0)[0],
                                       quad=_photo_of(art, 0)[1]))

    label = win.library.get(label_id)
    frame, quad = _photo_of(art, turns)
    box = {"kind": "obb", "label": "spec_plate", "label_id": label_id,
           "points": [list(p) for p in quad]}
    assert win._label_turn(box, label, frame) == turns


def test_the_field_lands_on_the_printing_whichever_way_up():
    """The end of it: a region drawn over the part number on the artwork sits
    over the part number on the image, at every quarter turn."""
    from label_detections.core import annotations as ann

    win = _window()
    label = _label(win, "refann_lands")
    art = _asymmetric()
    upright, quad0 = _photo_of(art, 0)
    # A field over the dark block, which is the end that is easy to check for.
    field = dict(FIELD, region=[0.0, 0.0, 0.30, 1.0])
    win._save_reference(label, _result(artwork=art, frame=upright, quad=quad0,
                                       text_fields=[field]))
    label = win.library.get("refann_lands")

    for turns in (0, 1, 2, 3):
        frame, quad = _photo_of(art, turns)
        box = {"kind": "obb", "label": "spec_plate", "label_id": "refann_lands",
               "points": [list(p) for p in quad]}
        ann.apply_reference_regions(
            box, label, overwrite=True, turn=win._label_turn(box, label, frame))
        placed = ann.regions(box, "text")[0]["points"]
        xs = [int(x) for x, _ in placed]
        ys = [int(y) for _, y in placed]
        patch = frame[min(ys) + 4:max(ys) - 4, min(xs) + 4:max(xs) - 4]
        assert patch.mean() < 60, f"a quarter turn of {turns} missed the dark end"
