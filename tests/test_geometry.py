from __future__ import annotations

import math

import pytest

from label_detections.core import geometry as geo
from conftest import rect


def test_point_in_polygon_inside_and_outside():
    poly = rect(0, 0, 10, 10)
    assert geo.point_in_polygon(5, 5, poly)
    assert not geo.point_in_polygon(15, 5, poly)
    assert not geo.point_in_polygon(-1, -1, poly)


def test_point_in_polygon_handles_rotated_quad():
    diamond = [[5, 0], [10, 5], [5, 10], [0, 5]]
    assert geo.point_in_polygon(5, 5, diamond)
    assert not geo.point_in_polygon(0.5, 0.5, diamond)


def test_rect_iou():
    assert geo.rect_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert geo.rect_iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0
    assert geo.rect_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_quad_angle_is_zero_for_upright_and_signed_for_rotation():
    assert geo.quad_angle_deg(rect(0, 0, 10, 5)) == pytest.approx(0.0)
    rotated = [[0, 0], [0, 10], [-5, 10], [-5, 0]]
    assert geo.quad_angle_deg(rotated) == pytest.approx(90.0)


def test_angle_delta_wraps():
    assert geo.angle_delta_deg(179, -179) == pytest.approx(2.0)
    assert geo.angle_delta_deg(0, 180) == pytest.approx(180.0)


def test_quad_size_averages_opposite_edges():
    keystoned = [[0, 0], [100, 0], [98, 50], [2, 50]]
    w, h = geo.quad_size(keystoned)
    assert w == pytest.approx(98.0, abs=1.0)
    assert h == pytest.approx(50.0, abs=1.0)


def test_homography_round_trip_through_label_space():
    """A pixel mapped to label space and back must land where it started."""
    quad = [[100, 50], [700, 80], [690, 400], [90, 380]]
    to_ref = geo.homography_to_unit(quad)
    from_ref = geo.homography_from_unit(quad)
    for px, py in [(120, 60), (400, 200), (650, 350)]:
        mx, my = geo.apply_homography(to_ref, px, py)
        bx, by = geo.apply_homography(from_ref, mx, my)
        assert bx == pytest.approx(px, abs=1e-6)
        assert by == pytest.approx(py, abs=1e-6)


def test_homography_maps_corners_onto_the_unit_square():
    quad = [[10, 20], [210, 25], [205, 125], [5, 120]]
    to_ref = geo.homography_to_unit(quad)
    assert geo.apply_homography(to_ref, *quad[0]) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert geo.apply_homography(to_ref, *quad[2]) == pytest.approx((1.0, 1.0), abs=1e-9)


def test_place_unit_rect_puts_a_code_where_the_artwork_says():
    """The labeling shortcut: draw the label, get the barcode box for free."""
    label_quad = rect(100, 100, 200, 120)
    placed = geo.place_unit_rect(label_quad, [0.1, 0.1, 0.4, 0.3])
    assert placed[0] == pytest.approx([120.0, 112.0])
    assert placed[2] == pytest.approx([200.0, 148.0])


def test_placement_needs_no_size_and_no_calibration():
    """The same fractions land proportionally on a label of any size."""
    small = geo.place_unit_rect(rect(0, 0, 100, 60), [0.25, 0.5, 0.5, 0.25])
    large = geo.place_unit_rect(rect(0, 0, 400, 240), [0.25, 0.5, 0.5, 0.25])
    assert small[0] == pytest.approx([25.0, 30.0])
    assert large[0] == pytest.approx([100.0, 120.0])


def test_place_unit_rect_follows_a_rotated_label():
    upright = rect(0, 0, 100, 100)
    rotated = [[0, 0], [0, 100], [-100, 100], [-100, 0]]
    a = geo.place_unit_rect(upright, [0.1, 0.2, 0.3, 0.3])
    b = geo.place_unit_rect(rotated, [0.1, 0.2, 0.3, 0.3])
    assert a != b
    # Same shape, just turned: the placed region keeps its size.
    assert geo.quad_size(a) == pytest.approx(geo.quad_size(b), abs=1e-6)


def test_a_region_with_no_area_places_nothing():
    assert geo.place_unit_rect(rect(0, 0, 10, 10), [0.1, 0.1, 0.0, 0.0]) is None
    assert geo.place_unit_rect(rect(0, 0, 10, 10), []) is None


def test_degenerate_quad_returns_none_instead_of_raising():
    collapsed = [[0, 0], [0, 0], [0, 0], [0, 0]]
    assert geo.homography_to_unit(collapsed) is None
    assert geo.place_unit_rect(collapsed, [0.1, 0.1, 0.2, 0.2]) is None


# --- flattening a label out of a full-resolution frame ----------------------

def _big_frame_and_quad(label_w=2000, label_h=800, deg=21):
    import math
    import cv2
    import numpy as np
    frame = np.full((3672, 5496, 3), 60, np.uint8)
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    quad = [[2700 + x * c - y * s, 1800 + x * s + y * c]
            for x, y in ((-label_w / 2, -label_h / 2), (label_w / 2, -label_h / 2),
                         (label_w / 2, label_h / 2), (-label_w / 2, label_h / 2))]
    cv2.fillPoly(frame, [np.array(quad, np.int32)], (230, 230, 225))
    return frame, quad


def test_capping_the_warp_shrinks_the_output_and_keeps_its_shape():
    """warpPerspective costs what its destination costs, so flattening a
    2000 px label at native size only to letterbox it down to 224 pays for
    2000 px of warp to keep 224."""
    from label_detections.core.imageio import rectify_quad

    frame, quad = _big_frame_and_quad()
    native = rectify_quad(frame, quad)
    capped = rectify_quad(frame, quad, max_side=448)

    assert max(native.shape[:2]) > 1900
    assert max(capped.shape[:2]) == 448
    # Aspect survives -- it is a real cue for the classifier, a tall trace tag
    # against a wide spec plate.
    native_aspect = native.shape[1] / native.shape[0]
    capped_aspect = capped.shape[1] / capped.shape[0]
    assert capped_aspect == pytest.approx(native_aspect, rel=0.02)


def test_the_cap_never_enlarges_a_small_label():
    """A 300 px label asked to fill 448 would be invented detail."""
    from label_detections.core.imageio import rectify_quad

    frame, quad = _big_frame_and_quad(label_w=300, label_h=120)
    capped = rectify_quad(frame, quad, max_side=448)
    assert max(capped.shape[:2]) == pytest.approx(300, abs=4)


def test_no_cap_is_the_old_behaviour():
    """Everything that stores artwork still gets the full-resolution flatten."""
    from label_detections.core.imageio import rectify_quad

    frame, quad = _big_frame_and_quad()
    assert rectify_quad(frame, quad).shape == rectify_quad(frame, quad, max_side=0).shape


def test_the_capped_crop_still_reads_the_same():
    """The point of the crop is that a classifier can read what is printed on
    it; a cheaper warp that blurred the print would be a false economy."""
    import numpy as np
    from label_detections.core.imageio import rectify_quad
    from label_detections.core.classify_export import letterbox

    frame, quad = _big_frame_and_quad()
    native = letterbox(rectify_quad(frame, quad), 224)
    capped = letterbox(rectify_quad(frame, quad, max_side=448), 224)
    diff = np.abs(native.astype(np.int16) - capped.astype(np.int16))
    assert diff.mean() < 4.0, "the cheaper warp changed the crop materially"
