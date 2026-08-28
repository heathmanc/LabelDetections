"""Deciding which way up a label was presented by looking at it.

The obvious idea is to use the detector's angle. It cannot work: Ultralytics
puts ground-truth corners through ``cv2.minAreaRect`` when it trains, which
discards which corner was drawn first, and regularises predictions into a
quarter-turn range with the longer side as the width. The model is never shown
which end of a label is its top.

What is left is the artwork. Every label is required to have a reference image,
and that image is the label the right way up by definition -- somebody drew its
outline on it. Flattening the detection each way round and scoring it against
that picture answers the question without reading anything, which matters
because the frames this gets wrong are the ones where the print is unreadable.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from label_detections.core import artwork as artwork_logic
from label_detections.core import geometry as geo
from label_detections.core.labels import LabelDef

ART_W, ART_H = 300, 90


def _artwork(serial: str = "SN000001") -> np.ndarray:
    """A label that is obviously not symmetric: dark block at one end, print at
    the other, and a per-unit serial down in the corner."""
    image = np.full((ART_H, ART_W, 3), 235, np.uint8)
    image[:, :90] = (30, 30, 30)
    cv2.putText(image, "PC680", (110, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(image, serial, (110, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return image


def _photograph(art, *, quarter_turns: int = 0, standing: bool = False):
    """The artwork sitting somewhere in a bigger frame, turned as asked.

    Returns ``(frame, quad)`` with the quad in the corner order a detector
    would hand over -- axis-aligned from its top-left, which says nothing about
    which way up the printing inside it is.
    """
    label = art.copy()
    for _ in range(quarter_turns % 4):
        label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)
    if standing and label.shape[1] > label.shape[0]:
        label = cv2.rotate(label, cv2.ROTATE_90_CLOCKWISE)

    h, w = label.shape[:2]
    frame = np.full((h + 200, w + 200, 3), 175, np.uint8)
    frame[100:100 + h, 100:100 + w] = label
    quad = [[100.0, 100.0], [100.0 + w, 100.0],
            [100.0 + w, 100.0 + h], [100.0, 100.0 + h]]
    return frame, quad


ASPECT = ART_W / ART_H


# --- what the angle could and could not have told us -------------------------

def test_every_reading_of_one_box_reports_the_same_angle():
    """The decisive fact. ``cv2.minAreaRect`` is what Ultralytics runs on
    ground-truth corners, and it gives the same angle and the same size for a
    quad and for every quarter turn of that quad. Four presentations that need
    four different region placements are one number to anything angular, so the
    corner order the operator drew never survives into training."""
    quad = [[0.0, 0.0], [300.0, 40.0], [288.0, 130.0], [-12.0, 90.0]]
    base = cv2.minAreaRect(np.array(quad, np.float32))
    for turn in (1, 2, 3):
        other = cv2.minAreaRect(np.array(geo.turn_quad(quad, turn), np.float32))
        assert round(other[2], 3) == round(base[2], 3)
        assert [round(v, 3) for v in other[1]] == [round(v, 3) for v in base[1]]


def test_the_angle_only_ever_spans_a_quarter_turn():
    """Turn a label through a whole revolution and every angle reported lands
    inside a ninety degree band -- whichever convention the build uses. There
    is no room in it for which way up, which is why predictions come back
    through ``regularize_rboxes`` with the longer side as the width."""
    import math

    angles = []
    for degrees in range(0, 360, 15):
        t = math.radians(degrees)
        angles.append(cv2.minAreaRect(np.array(
            [[math.cos(t) * x - math.sin(t) * y, math.sin(t) * x + math.cos(t) * y]
             for x, y in ((-150, -45), (150, -45), (150, 45), (-150, 45))],
            np.float32))[2])
    assert max(angles) - min(angles) <= 90


# --- the comparison ----------------------------------------------------------

def test_an_upright_label_is_recognised_as_upright():
    art = _artwork()
    frame, quad = _photograph(art)
    turn, score, margin = artwork_logic.best_turn(frame, quad, art, aspect=ASPECT)
    assert turn == 0
    assert score > 0.9 and artwork_logic.settled(score, margin)


def test_an_upside_down_label_is_recognised_as_upside_down():
    """The case that put the part-number box over blank artwork at the far end."""
    art = _artwork()
    frame, quad = _photograph(art, quarter_turns=2)
    turn, score, margin = artwork_logic.best_turn(frame, quad, art, aspect=ASPECT)
    assert turn == 2
    assert score > 0.9 and artwork_logic.settled(score, margin)


@pytest.mark.parametrize("turns", [1, 3])
def test_a_standing_label_is_recognised_whichever_way_up(turns):
    """The quarter-turn case from the screenshot: proportion narrows it to two
    readings and the artwork picks between them."""
    art = _artwork()
    frame, quad = _photograph(art, quarter_turns=turns)
    turn, score, margin = artwork_logic.best_turn(frame, quad, art, aspect=ASPECT)
    assert turn == turns
    assert score > 0.9 and artwork_logic.settled(score, margin)


def test_only_the_readings_of_the_right_shape_are_paid_for():
    """Two warps, not four. Proportion has already ruled the other two out."""
    art = _artwork()
    frame, quad = _photograph(art, quarter_turns=1)
    scored = artwork_logic.score_turns(
        frame, quad, art, turns=geo.candidate_turns(quad, ASPECT))
    assert sorted(scored) == [1, 3]


def test_brightness_and_contrast_do_not_change_the_answer():
    """Zero-mean and unit-norm: the question is where the light and dark parts
    are, not how bright the room was."""
    art = _artwork()
    frame, quad = _photograph(art, quarter_turns=2)
    dim = np.clip(frame.astype(np.float32) * 0.45 + 30, 0, 255).astype(np.uint8)
    assert artwork_logic.best_turn(dim, quad, art, aspect=ASPECT)[0] == 2


def test_a_label_that_looks_the_same_both_ways_admits_it():
    """A symmetric label scores alike either way up, and picking the winner of
    that is a coin toss dressed as a measurement."""
    art = np.full((ART_H, ART_W, 3), 235, np.uint8)
    art[:, :60] = (30, 30, 30)
    art[:, -60:] = (30, 30, 30)
    frame, quad = _photograph(art, quarter_turns=2)
    _turn, score, margin = artwork_logic.best_turn(frame, quad, art, aspect=ASPECT)
    assert not artwork_logic.settled(score, margin)
    assert "not conclusive" in artwork_logic.note(_turn, score, margin)


def test_a_crop_of_something_else_is_not_this_label():
    """Best-of-four on a box that does not contain the label at all means
    nothing, so a bad box does not get a confident orientation."""
    art = _artwork()
    frame = np.random.default_rng(0).integers(0, 255, (290, 500, 3), dtype=np.uint8)
    quad = [[100.0, 100.0], [400.0, 100.0], [400.0, 190.0], [100.0, 190.0]]
    _turn, score, margin = artwork_logic.best_turn(frame, quad, art, aspect=ASPECT)
    assert not artwork_logic.settled(score, margin)


def test_no_artwork_means_no_opinion():
    frame, quad = _photograph(_artwork())
    assert artwork_logic.best_turn(frame, quad, None, aspect=ASPECT) == (0, 0.0, 0.0)
    assert "no reference artwork" in artwork_logic.note(0, 0.0, 0.0)


# --- labels whose printing changes per unit ---------------------------------

def test_a_variable_serial_is_scored_out_of_the_comparison():
    """A serial that differs on every unit counts against both readings, and on
    a label where the rest is nearly symmetric it can swamp the difference. The
    anchor is the part that never changes, which is what it is for."""
    art = _artwork("SN000001")
    other = _artwork("SN994417")
    frame, quad = _photograph(other, quarter_turns=2)

    label = LabelDef(label_id="PC680", variable_data=True,
                     anchor_region=[0.0, 0.0, 1.0, 0.62])
    turn, score, margin = artwork_logic.best_turn(
        frame, quad, art, aspect=ASPECT, region=label.match_region())
    assert turn == 2 and artwork_logic.settled(score, margin)


def test_a_fixed_label_scores_against_all_of_its_artwork():
    assert LabelDef(label_id="x").match_region() == [0.0, 0.0, 1.0, 1.0]


# --- caching -----------------------------------------------------------------

def test_artwork_is_read_from_disk_once(tmp_path, monkeypatch):
    """Asked once per box per frame; re-decoding the PNG each time would be the
    expensive part of an otherwise cheap check."""
    path = tmp_path / "ref.png"
    cv2.imwrite(str(path), _artwork())
    label = LabelDef(label_id="cache_me", reference_images=[str(path)])

    reads = []
    real = cv2.imread
    monkeypatch.setattr(cv2, "imread", lambda *a, **k: (reads.append(a[0]), real(*a, **k))[1])

    artwork_logic._cache.clear()
    assert artwork_logic.load(label) is not None
    assert artwork_logic.load(label) is not None
    assert len(reads) == 1


def test_replaced_artwork_is_not_served_from_the_cache(tmp_path):
    """A reference is deleted and drawn again, and the label's coordinate
    system changes with it."""
    path = tmp_path / "ref.png"
    cv2.imwrite(str(path), _artwork())
    label = LabelDef(label_id="replace_me", reference_images=[str(path)])

    artwork_logic._cache.clear()
    first = artwork_logic.load(label)
    os.utime(path, (0, 0))
    cv2.imwrite(str(path), cv2.rotate(_artwork(), cv2.ROTATE_180))
    second = artwork_logic.load(label)
    assert not np.array_equal(first, second)
