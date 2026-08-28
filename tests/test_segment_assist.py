"""One click to an oriented box: the geometry half.

No model here on purpose. What can go wrong in this feature is not the
segmentation -- it is what happens to the mask afterwards: fitted at the wrong
scale, clamped after being judged, corners in a different order than last time,
or a click that missed the label handed back as a confident box around the
whole battery.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from label_detections.core import segment_assist as sa


def _mask(w, h, quad):
    import cv2
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m, [np.array(quad, dtype=np.int32)], 1)
    return m


def _rot(cx, cy, w, h, deg):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return [[cx + x * c - y * s, cy + x * s + y * c]
            for x, y in ((-w / 2, -h / 2), (w / 2, -h / 2),
                         (w / 2, h / 2), (-w / 2, h / 2))]


# --- scale ------------------------------------------------------------------

def test_a_frame_bigger_than_the_budget_is_shrunk_to_it():
    assert sa.assist_scale(5496, 3672, 1024) == pytest.approx(1024 / 5496)


def test_a_small_frame_is_never_enlarged():
    """Upscaling costs time and invents no detail."""
    assert sa.assist_scale(640, 480, 1024) == 1.0


# --- mask to quad -----------------------------------------------------------

def test_a_tilted_label_comes_back_as_a_tilted_rectangle():
    quad = _rot(300, 200, 240, 90, 25)
    got = sa.quad_from_mask(_mask(600, 400, quad))
    assert got is not None
    # Same area and same side lengths as what was drawn, within a pixel or two
    # of rasterisation.
    assert sa.quad_area(got) == pytest.approx(240 * 90, rel=0.03)
    sides = sorted(math.dist(got[i], got[(i + 1) % 4]) for i in range(4))
    assert sides[0] == pytest.approx(90, abs=3)
    assert sides[3] == pytest.approx(240, abs=3)


def test_a_speck_is_not_a_label():
    """A click that lands on a screw head or a glare spot must not produce a
    confident-looking box around nothing."""
    assert sa.quad_from_mask(_mask(600, 400, _rot(300, 200, 8, 8, 0))) is None


def test_stray_pixels_elsewhere_do_not_stretch_the_box():
    """Segmentation often leaves a few pixels somewhere else in the frame. A
    rectangle around the union of a label and a speck is around neither."""
    m = _mask(600, 400, _rot(200, 200, 200, 80, 0))
    m[10:24, 560:574] = 1                       # a blob in the far corner
    got = sa.quad_from_mask(m)
    assert sa.quad_area(got) == pytest.approx(200 * 80, rel=0.05)


# --- scale back, clamp, order ----------------------------------------------

def test_the_quad_is_scaled_back_to_full_resolution():
    quad = [[10.0, 20.0], [110.0, 20.0], [110.0, 80.0], [10.0, 80.0]]
    assert sa.scale_quad(quad, 0.25)[2] == [440.0, 320.0]


def test_a_corner_outside_the_frame_is_pulled_in():
    """A mask touching the edge fits a rectangle a pixel or two past it, which
    stored that way exports as an out-of-range coordinate."""
    quad = [[-4.0, -3.0], [610.0, -3.0], [610.0, 410.0], [-4.0, 410.0]]
    assert sa.clamp_quad(quad, 600, 400) == [[0.0, 0.0], [600.0, 0.0],
                                             [600.0, 400.0], [0.0, 400.0]]


def test_corners_come_back_in_the_same_order_however_the_fit_started():
    """minAreaRect starts wherever it likes. The corner handles are numbered,
    so the same label outlined twice must not put corner 1 in a new place."""
    quad = _rot(300, 200, 240, 90, 15)
    first = sa.order_quad(quad)
    rolled = sa.order_quad(quad[2:] + quad[:2])
    assert first == rolled
    # And corner 1 really is the top-left one.
    assert first[0] == min(first, key=lambda p: p[0] + p[1])


# --- the judgement ----------------------------------------------------------

def test_outlining_the_whole_battery_is_refused():
    """A click that misses the label by a few pixels does not fail -- it
    succeeds at outlining the battery, which looks right in the list and is
    wrong in the dataset."""
    whole = [[0.0, 0.0], [590.0, 0.0], [590.0, 390.0], [0.0, 390.0]]
    why = sa.rejection(whole, 600, 400)
    assert "most of the frame" in why


def test_a_long_thin_strip_is_refused():
    strip = [[10.0, 100.0], [580.0, 100.0], [580.0, 112.0], [10.0, 112.0]]
    assert "thin strip" in sa.rejection(strip, 600, 400)


def test_a_normal_label_is_accepted():
    assert sa.rejection(_rot(300, 200, 240, 90, 25), 600, 400) == ""


# --- the whole path ---------------------------------------------------------

def test_the_mask_is_judged_at_full_resolution_not_at_model_scale():
    """The order matters: a 240x90 label on a downscaled frame is a large
    fraction of *that* frame. Judged before scaling back, every good outline
    on a big camera frame would be refused as 'most of the frame'."""
    factor = 0.25
    small_quad = _rot(150, 100, 120, 45, 20)          # on a 300x200 working image
    mask = _mask(300, 200, small_quad)

    quad, why = sa.outline_from_mask(mask, 1200, 800, factor)
    assert why == ""
    assert sa.quad_area(quad) == pytest.approx(480 * 180, rel=0.05)
    assert max(p[0] for p in quad) <= 1200


def test_a_bad_click_returns_the_reason_and_no_quad():
    mask = _mask(300, 200, [[0, 0], [299, 0], [299, 199], [0, 199]])
    quad, why = sa.outline_from_mask(mask, 1200, 800, 0.25)
    assert quad == []
    assert why
