"""The one window that sets a label's reference image up.

Photograph the label, outline it, mark what to read on it. That was four
controls across two panes in an order nothing enforced, and it depended on the
main window's live preview already running somewhere else -- so a reference
could only be shot from a state you had to get into first, elsewhere.

The outline is four corners rather than a rectangle, and the artwork is that
quad rectified. Both matter for the same reason: a label at an angle inside an
axis-aligned box brings a wedge of background with it, and the outline is the
coordinate system every region is a fraction of.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-refsetup-"))

import pytest

pytest.importorskip("PySide6.QtWidgets")
cv2 = pytest.importorskip("cv2")
import numpy as np
from PySide6.QtWidgets import QApplication, QMessageBox

QApplication.instance() or QApplication([])

from label_detections.core.labels import LabelDef
from label_detections.ui import reference_setup as rs

# A label lying at an angle: the case an axis-aligned outline gets wrong.
TILTED = [[160.0, 150.0], [1020.0, 210.0], [1000.0, 430.0], [140.0, 370.0]]


def _frame():
    image = np.full((600, 1200, 3), 210, np.uint8)
    cv2.fillPoly(image, [np.array(TILTED, np.int32)], (40, 40, 200))
    return image


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "frame.png"
    cv2.imwrite(str(path), _frame())
    return path


def _dialog(photo=None, frames=None, label=None):
    return rs.ReferenceSetupDialog(
        label or LabelDef(label_id="PC680"),
        frames=frames or (lambda: None),
        images=[photo] if photo else [])


def _through_outline(photo=None, frames=None, quad=None):
    """Get to step 3 with an outline drawn."""
    dialog = _dialog(photo, frames)
    if frames is None:
        dialog.existing_radio.setChecked(True)
    else:
        dialog._shoot()
    if dialog.stack.currentIndex() == rs.FRAME_PAGE:
        dialog._go_next()
    dialog.outline_canvas.set_quad(quad or TILTED)
    dialog._go_next()
    return dialog


# --- step 1: the window owns the camera -------------------------------------

def test_it_shows_its_own_preview_rather_than_borrowing_one():
    """Reaching into the main window's preview meant the reference could only
    be shot from a tab that happened to be running."""
    live = _frame()
    dialog = _dialog(frames=lambda: live)
    dialog._tick_preview()
    assert dialog.preview.image_size() == (1200, 600)


def test_the_shutter_takes_the_frame_and_moves_on():
    live = _frame()
    dialog = _dialog(frames=lambda: live)
    dialog._shoot()
    assert dialog.frame is not None
    assert dialog.stack.currentIndex() == rs.OUTLINE_PAGE


def test_shooting_with_no_camera_says_so_rather_than_failing():
    dialog = _dialog(frames=lambda: None)
    dialog._shoot()
    assert dialog.stack.currentIndex() == rs.FRAME_PAGE
    assert "No camera frame" in dialog.problem.text()


def test_an_existing_image_works_with_no_camera_at_all(photo):
    """On a rig where the camera is on the line and the labelling happens at a
    desk, insisting on a live frame means the work can only be done in one room."""
    dialog = _dialog(photo)
    dialog.existing_radio.setChecked(True)
    dialog._go_next()
    assert dialog.stack.currentIndex() == rs.OUTLINE_PAGE
    assert dialog.frame is not None


# --- step 2: four corners, not a rectangle ----------------------------------

def test_it_will_not_move_on_without_an_outline(photo):
    dialog = _dialog(photo)
    dialog.existing_radio.setChecked(True)
    dialog._go_next()
    dialog._go_next()
    assert dialog.stack.currentIndex() == rs.OUTLINE_PAGE
    assert "Draw the outline first" in dialog.problem.text()


def test_the_outline_is_a_quad_and_the_artwork_is_it_rectified(photo):
    """The tilt comes out of the artwork, which is what lets the regions on it
    be plain rectangles."""
    dialog = _through_outline(photo)
    assert dialog.stack.currentIndex() == rs.REGION_PAGE
    assert dialog.artwork is not None
    height, width = dialog.artwork.shape[:2]
    assert width == 900                       # rectified to a known width
    # The tilted label is about 4:1; the flattened artwork keeps that.
    assert 2.5 < width / height < 6.0


def test_a_degenerate_outline_is_refused_with_a_reason(photo):
    dialog = _dialog(photo)
    dialog.existing_radio.setChecked(True)
    dialog._go_next()
    dialog.outline_canvas.set_quad([[10, 10], [11, 10], [12, 10], [11, 11]])
    dialog._go_next()
    assert dialog.stack.currentIndex() == rs.OUTLINE_PAGE
    assert "could not be flattened" in dialog.problem.text()


def test_corners_are_put_in_a_known_order_as_they_are_drawn():
    """Leaving a quad wound the other way rectifies it into a mirror image --
    a failure that looks like nothing at all until a barcode will not decode."""
    from label_detections.ui.quad_canvas import QuadCanvas

    canvas = QuadCanvas()
    canvas.set_frame(_frame())
    canvas.set_quad([[0, 0], [0, 300], [900, 300], [900, 0]])   # anticlockwise
    assert canvas.quad[0] == [0.0, 0.0]
    assert canvas.quad[1] == [900.0, 0.0]


def test_going_back_to_the_camera_drops_the_still(photo):
    """Otherwise the shutter is offered over a frozen frame."""
    dialog = _through_outline(photo)
    dialog._go_back()
    dialog._go_back()
    assert dialog.stack.currentIndex() == rs.FRAME_PAGE
    assert dialog.frame is None


# --- step 3: regions on the flattened artwork -------------------------------

def test_regions_come_back_as_fractions_of_the_label(photo, monkeypatch):
    """The conversion the whole feature rests on."""
    from PySide6.QtCore import QRectF

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog = _through_outline(photo)
    width, height = dialog.body.canvas.image_size()
    dialog.body.canvas.regions.append(
        {"role": "code", "name": "serial",
         "rect": QRectF(width * 0.8, height * 0.2, width * 0.15, height * 0.3)})
    dialog._finish_clicked()

    region = dialog.result["codes"][0]["region"]
    assert region[0] == pytest.approx(0.80, abs=0.02)
    assert region[2] == pytest.approx(0.15, abs=0.02)


def test_saving_with_no_regions_asks_first(photo, monkeypatch):
    """A reference with nothing marked on it can be detected and classified and
    cannot be verified -- which is what stops an unenrolled label being
    reported as this one."""
    asked = {}

    def refuse(parent, title, text, *a, **k):
        asked["text"] = text
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(refuse))
    dialog = _through_outline(photo)
    dialog._finish_clicked()
    assert "cannot be verified" in asked.get("text", "")
    assert dialog.result is None


def test_the_saved_result_carries_the_artwork_and_the_quad(photo, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog = _through_outline(photo)
    dialog._finish_clicked()
    assert dialog.result["artwork"] is not None
    assert len(dialog.result["quad"]) == 4


def test_nothing_is_written_until_the_last_step_finishes(photo):
    """What makes the artwork safe to treat as immutable: there is no
    half-completed state that could have moved it."""
    dialog = _through_outline(photo)
    assert dialog.result is None
    dialog.reject()
    assert dialog.result is None
