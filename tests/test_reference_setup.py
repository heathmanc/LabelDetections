"""The one window that sets a label's reference image up.

Photograph the label, say where it is in the photograph, say what to read on
it. That was four controls across two panes in an order nothing enforced, and
a label could sit half-defined indefinitely while looking finished.
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
from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QApplication, QMessageBox

QApplication.instance() or QApplication([])

from label_detections.core.labels import LabelDef
from label_detections.ui import reference_setup as rs


@pytest.fixture
def photo(tmp_path):
    """A raw frame: a label with a lot of table around it."""
    path = tmp_path / "frame.png"
    image = np.full((600, 1200, 3), 210, np.uint8)
    cv2.rectangle(image, (150, 180), (1050, 420), (40, 40, 200), -1)
    cv2.imwrite(str(path), image)
    return path


def _dialog(photo, label=None, capture=None):
    return rs.ReferenceSetupDialog(label or LabelDef(label_id="PC680"),
                                   capture=capture or (lambda: None),
                                   images=[photo])


def _at_draw_page(photo, **kw):
    dialog = _dialog(photo, **kw)
    dialog.existing_radio.setChecked(True)
    dialog._go_next()
    return dialog


def _outline(dialog, rect=(150, 180, 900, 240)):
    dialog.body.canvas.outline = QRectF(*rect)
    dialog.body.canvas.outline_is_default = False


# --- the shape of the flow --------------------------------------------------

def test_it_starts_by_asking_where_the_photograph_comes_from(photo):
    dialog = _dialog(photo)
    assert dialog.stack.currentIndex() == rs.SOURCE_PAGE
    assert "photograph of PC680" in dialog.heading.text()


def test_an_existing_image_can_be_used_without_the_camera(photo):
    """On a rig where the camera is on the line and the labelling happens at a
    desk, insisting on a live frame means the work can only happen in one room."""
    dialog = _at_draw_page(photo)
    assert dialog.stack.currentIndex() == rs.DRAW_PAGE
    assert dialog.body.has_image()


def test_choosing_to_shoot_with_no_camera_says_so_rather_than_failing(photo):
    dialog = _dialog(photo, capture=lambda: None)
    dialog.shoot_radio.setChecked(True)
    dialog._go_next()
    assert dialog.stack.currentIndex() == rs.SOURCE_PAGE
    assert "No frame" in dialog.problem.text()


def test_you_can_go_back_and_choose_again(photo):
    dialog = _at_draw_page(photo)
    dialog._go_back()
    assert dialog.stack.currentIndex() == rs.SOURCE_PAGE


# --- the outline is the whole point -----------------------------------------

def test_finishing_without_drawing_the_outline_is_refused(photo):
    """Loading defaults the outline to the whole image, which is right for an
    artwork already cropped to the label and badly wrong for a raw photograph:
    every region would be a fraction of the picture rather than of the label,
    and at runtime those fractions land on the detector's tight box and miss by
    the width of the margin."""
    dialog = _at_draw_page(photo)
    dialog._finish_clicked()
    assert dialog.result is None
    assert "outline first" in dialog.problem.text()


def test_a_drawn_outline_is_accepted(photo, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog = _at_draw_page(photo)
    _outline(dialog)
    dialog._finish_clicked()
    assert dialog.result is not None
    assert [round(v) for v in dialog.result["outline"]] == [150, 180, 900, 240]


def test_regions_come_back_as_fractions_of_the_label_not_the_photograph(photo,
                                                                       monkeypatch):
    """The one conversion the whole feature rests on."""
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    dialog = _at_draw_page(photo)
    _outline(dialog)
    dialog.body.canvas.regions.append(
        {"role": "code", "name": "serial", "rect": QRectF(800, 220, 200, 120)})
    dialog._finish_clicked()

    region = dialog.result["codes"][0]["region"]
    # x: (800 - 150) / 900, w: 200 / 900
    assert region[0] == pytest.approx(0.7222, abs=1e-3)
    assert region[2] == pytest.approx(0.2222, abs=1e-3)


def test_saving_with_no_regions_at_all_asks_first(photo, monkeypatch):
    """A reference with nothing marked on it can be detected and classified and
    cannot be verified -- which is the thing that stops an unenrolled label
    being reported as this one."""
    asked = {}
    monkeypatch.setattr(
        QMessageBox, "question",
        staticmethod(lambda parent, title, text, *a, **k:
                     asked.setdefault("text", text) and QMessageBox.No
                     or QMessageBox.No))
    dialog = _at_draw_page(photo)
    _outline(dialog)
    dialog._finish_clicked()
    assert "cannot be verified" in asked.get("text", "")
    assert dialog.result is None, "saved after the offer was declined"


def test_nothing_is_written_until_the_whole_thing_finishes(photo):
    """What makes the artwork safe to treat as immutable: there is no
    half-completed state that could have moved it."""
    dialog = _at_draw_page(photo)
    dialog._finish_clicked()          # refused, no outline
    assert dialog.result is None
    dialog.reject()
    assert dialog.result is None
