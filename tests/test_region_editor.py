"""The read-region editor: drag a box on artwork, get a fraction of the label.

The conversion is the whole thing. Get it wrong and every region points at the
wrong part of every unit forever, which is exactly the kind of bug that only
shows up on a conveyor.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-regions-"))

import pytest

try:
    from PySide6.QtCore import QRectF
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception as exc:  # pragma: no cover - depends on the environment
    HAVE_QT = False
    _WHY = exc

pytestmark = pytest.mark.skipif(not HAVE_QT, reason="PySide6 not available")


def _canvas(outline=(0, 0, 400, 200)):
    QApplication.instance() or QApplication([])
    from label_detections.ui.region_editor import RegionCanvas

    canvas = RegionCanvas()
    canvas.outline = QRectF(*outline)
    return canvas


EXISTING = [{"role": "serial", "symbology": "code128", "policy": "must_match_pattern",
             "pattern": r"^SN\d{6}$", "region": [0.1, 0.1, 0.2, 0.2],
             "code_width_mm": 30.0, "x_dim_mm": 0.33}]


def _dialog_with_artwork(tmp_path, codes=(), text_fields=(), anchor=()):
    """A real 400x200 image on disk, so the canvas actually loads."""
    from PySide6.QtGui import QImage
    from label_detections.ui.region_editor import RegionEditorDialog

    QApplication.instance() or QApplication([])
    path = tmp_path / "ref.png"
    QImage(400, 200, QImage.Format_RGB32).save(str(path))
    return RegionEditorDialog(str(path), [dict(c) for c in codes],
                              [dict(t) for t in text_fields], list(anchor))


def test_a_rect_becomes_a_fraction_of_the_label():
    canvas = _canvas()
    assert canvas.to_fraction(QRectF(100, 50, 200, 100)) == [0.25, 0.25, 0.5, 0.5]


def test_the_outline_is_what_fractions_are_measured_against():
    """A reference photo has margin; the label is what matters, not the image."""
    canvas = _canvas(outline=(100, 100, 200, 100))
    # The same pixels are the whole label now, not a quarter of the image.
    assert canvas.to_fraction(QRectF(100, 100, 200, 100)) == [0.0, 0.0, 1.0, 1.0]
    assert canvas.to_fraction(QRectF(150, 125, 50, 25)) == [0.25, 0.25, 0.25, 0.25]


def test_fractions_round_trip_back_to_pixels():
    canvas = _canvas()
    rect = QRectF(80, 40, 120, 60)
    back = canvas.from_fraction(canvas.to_fraction(rect))
    assert back.x() == pytest.approx(rect.x(), abs=0.5)
    assert back.width() == pytest.approx(rect.width(), abs=0.5)


def test_a_region_dragged_past_the_label_edge_is_clamped():
    """Storing it would put the runtime crop off the label entirely."""
    canvas = _canvas()
    fraction = canvas.to_fraction(QRectF(300, 150, 400, 200))
    x, y, w, h = fraction
    assert x + w <= 1.0 and y + h <= 1.0
    assert w > 0 and h > 0


def test_no_outline_yields_no_fraction_rather_than_a_divide_by_zero():
    canvas = _canvas(outline=(0, 0, 0, 0))
    assert canvas.to_fraction(QRectF(0, 0, 10, 10)) == []
    assert canvas.from_fraction([0.1, 0.1, 0.2, 0.2]).isNull()


def test_default_names_do_not_collide():
    canvas = _canvas()
    first = canvas._default_name("code")
    canvas.regions.append({"role": "code", "name": first, "rect": QRectF()})
    assert canvas._default_name("code") != first
    assert canvas._default_name("text") == "field_1"


def test_drawn_regions_come_back_as_wizard_rows(tmp_path):
    dialog = _dialog_with_artwork(tmp_path)
    dialog.canvas.outline = QRectF(0, 0, 400, 200)
    dialog.canvas.regions = [
        {"role": "code", "name": "serial", "rect": QRectF(40, 20, 80, 40)},
        {"role": "text", "name": "date_code", "rect": QRectF(40, 120, 200, 30)},
        {"role": "anchor", "name": "anchor", "rect": QRectF(0, 0, 400, 100)},
    ]
    result = dialog.result_regions()
    assert result["codes"][0]["role"] == "serial"
    assert result["codes"][0]["region"] == [0.1, 0.1, 0.2, 0.2]
    assert result["text_fields"][0]["name"] == "date_code"
    assert result["anchor_region"] == [0.0, 0.0, 1.0, 0.5]


def test_existing_policies_survive_a_second_visit(tmp_path):
    """The editor supplies geometry. It must not reset what was already decided."""
    dialog = _dialog_with_artwork(tmp_path, codes=EXISTING)
    assert len(dialog.canvas.regions) == 1, "the saved region should be shown"
    row = dialog.result_regions()["codes"][0]
    assert row["policy"] == "must_match_pattern"
    assert row["pattern"] == r"^SN\d{6}$"
    assert row["x_dim_mm"] == 0.33
    assert row["region"] == [0.1, 0.1, 0.2, 0.2]


def test_moving_a_region_updates_only_its_geometry(tmp_path):
    dialog = _dialog_with_artwork(tmp_path, codes=EXISTING)
    dialog.canvas.regions[0]["rect"] = QRectF(200, 100, 80, 40)
    row = dialog.result_regions()["codes"][0]
    assert row["region"] == [0.5, 0.5, 0.2, 0.2]
    assert row["pattern"] == r"^SN\d{6}$"


def test_unreadable_artwork_returns_the_rows_untouched():
    """A moved reference must never cost the operator what they already entered."""
    from label_detections.ui.region_editor import RegionEditorDialog

    QApplication.instance() or QApplication([])
    dialog = RegionEditorDialog("/no/such/file.png", [dict(c) for c in EXISTING], [], [])
    result = dialog.result_regions()
    assert result["codes"] == EXISTING
