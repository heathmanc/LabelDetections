"""Which way up a label was presented.

Four corners cannot say. An upside-down label produces corners in the same
slots as an upright one -- order_quad normalises to the IMAGE's top-left, which
is all geometry can do, because nothing in a quadrilateral says which way up
the printing is. So every region measured from the top-left lands at the
diagonally opposite end of the label, which is exactly what a batch of
auto-labelled images showed: the part-number region drawn over blank artwork at
the far corner from the part number.

The only thing that settles it is reading the label and seeing which reading
produces what it is supposed to say.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import annotations as ann
from label_detections.core import geometry as geo
from label_detections.core.labels import CodeSpec, LabelDef

# A wide label, and the barcode near its right-hand end.
QUAD = [[0.0, 0.0], [1000.0, 0.0], [1000.0, 300.0], [0.0, 300.0]]
BARCODE = [0.842, 0.239, 0.151, 0.197]


def _span(placed):
    xs = [x for x, _ in placed]
    return round(min(xs)), round(max(xs))


# --- the geometry ------------------------------------------------------------

def test_flipping_twice_is_the_same_quad():
    assert geo.flip_quad(geo.flip_quad(QUAD)) == QUAD


def test_a_flip_moves_a_region_to_the_far_end():
    """The symptom, in one assertion. A region pinned to the right-hand end of
    an upright label belongs at the left-hand end of an upside-down one."""
    upright = geo.place_unit_rect(geo.oriented(QUAD), BARCODE, orient=False)
    flipped = geo.place_unit_rect(geo.oriented(QUAD, True), BARCODE, orient=False)
    assert _span(upright) == (842, 993)
    assert _span(flipped) == (7, 158)


def test_ordering_first_is_what_makes_the_flip_survive():
    """Flipping and then ordering normalises the flip straight back out, which
    is why the two are combined in one place instead of left to callers."""
    assert geo.order_quad(geo.flip_quad(QUAD)) == geo.order_quad(QUAD)
    assert geo.oriented(QUAD, True) != geo.oriented(QUAD, False)


def test_the_default_path_is_unchanged():
    """Every label with a fixed rotation must place exactly as it always did."""
    assert geo.place_unit_rect(QUAD, BARCODE) == \
        geo.place_unit_rect(geo.oriented(QUAD), BARCODE, orient=False)


def test_a_degenerate_quad_is_handed_back_rather_than_flipped():
    assert geo.flip_quad([[0, 0], [1, 1]]) == [[0, 0], [1, 1]]


# --- who is allowed to be upside down ---------------------------------------

def test_only_a_label_that_may_turn_over_is_tried_both_ways():
    """A read costs time, and for a label with a fixed rotation it is a read
    for an answer already known."""
    from label_detections.core.code_reader import orientations

    assert orientations("fixed") == [False]
    assert orientations("") == [False]
    assert orientations("flip_ok") == [False, True]
    assert orientations("any") == [False, True]


def test_the_upright_reading_is_always_tried_first():
    """Most labels arrive the right way up, and the second crop is only worth
    paying for when the first says nothing."""
    from label_detections.core.code_reader import orientations

    assert orientations("any")[0] is False


# --- placing regions for display --------------------------------------------

def _box():
    return {"x": 0, "y": 0, "w": 1000, "h": 300, "kind": "obb",
            "label": "label", "label_id": "PC680", "points": [list(p) for p in QUAD]}


def _label():
    label = LabelDef(label_id="PC680", rotation_policy="flip_ok")
    label.codes = [CodeSpec(role="serial", policy="must_decode",
                            pattern="^635241140996$", region=list(BARCODE))]
    return label


def test_regions_are_drawn_at_the_end_the_label_actually_is():
    upright = ann.place_label_regions(_box(), _label(), flipped=False)
    flipped = ann.place_label_regions(_box(), _label(), flipped=True)
    assert _span(upright[0]["points"]) == (842, 993)
    assert _span(flipped[0]["points"]) == (7, 158)


def test_placing_defaults_to_upright_so_nothing_moves_unasked():
    assert ann.place_label_regions(_box(), _label()) == \
        ann.place_label_regions(_box(), _label(), flipped=False)


def test_apply_reference_regions_passes_the_orientation_through():
    box = _box()
    ann.apply_reference_regions(box, _label(), flipped=True)
    assert _span(ann.regions(box)[0]["points"]) == (7, 158)
