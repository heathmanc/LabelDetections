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


# --- corner order, and the mirror it caused ---------------------------------
#
# rectify_quad maps slot 0 to the output's top-left and slot 1 to its
# top-right. A quad wound anticlockwise therefore flattens to a mirror image --
# and a detector's oriented box winds whichever way the predicted angle
# implies. It showed up as a barcode that would not decode; the same crop was
# feeding the classifier, which had trained on unmirrored ones.

CANONICAL = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def test_an_already_canonical_quad_is_left_alone():
    """Drawn boxes are stored in this order, so ordering them must be a no-op
    or every existing reference artwork changes."""
    assert geo.order_quad(CANONICAL) == CANONICAL


def test_an_anticlockwise_quad_is_rewound():
    """The mirror. Same four corners, opposite winding."""
    assert geo.order_quad([[0, 0], [0, 10], [10, 10], [10, 0]]) == CANONICAL


def test_a_quad_starting_at_any_corner_is_rotated_to_the_top_left():
    for start in range(4):
        rolled = CANONICAL[start:] + CANONICAL[:start]
        assert geo.order_quad(rolled) == CANONICAL


def test_ordering_survives_a_rotated_label():
    """A label at an angle still has a corner nearest the image's top-left, and
    the winding still has to come out clockwise."""
    tilted = [[50, 10], [90, 50], [50, 90], [10, 50]]      # diamond, clockwise
    ordered = geo.order_quad(tilted)
    assert ordered[0] == [50.0, 10.0]
    # Shoelace with y down: clockwise on screen is a negative signed area.
    signed = sum(ordered[i][0] * ordered[(i + 1) % 4][1]
                 - ordered[(i + 1) % 4][0] * ordered[i][1] for i in range(4))
    assert signed > 0, "came back wound the wrong way"


def test_a_degenerate_quad_is_handed_back_rather_than_reordered():
    assert geo.order_quad([[0, 0], [1, 1]]) == [[0, 0], [1, 1]]


def test_the_flatten_is_identical_however_the_corners_arrive():
    """The point of all of it, on real pixels rather than coordinates."""
    import numpy as np
    import pytest

    pytest.importorskip("cv2")
    import cv2

    from label_detections.core.imageio import rectify_quad

    frame = np.full((300, 600, 3), 255, np.uint8)
    cv2.putText(frame, "ODYSSEY", (60, 180), cv2.FONT_HERSHEY_DUPLEX, 2.2,
                (0, 0, 0), 4)
    upright = [[40, 100], [560, 100], [560, 220], [40, 220]]
    good = rectify_quad(frame, upright)

    for name, quad in (
        ("anticlockwise", [[40, 100], [40, 220], [560, 220], [560, 100]]),
        ("started bottom-right", [[560, 220], [40, 220], [40, 100], [560, 100]]),
    ):
        assert np.array_equal(rectify_quad(frame, quad), good), name


def test_without_ordering_the_flatten_really_does_come_out_mirrored():
    """Proves the bug is real rather than the fix being decorative -- and that
    the landscape label turns portrait, which is how it was first spotted."""
    import numpy as np
    import pytest

    pytest.importorskip("cv2")

    from label_detections.core.imageio import rectify_quad

    frame = np.full((300, 600, 3), 255, np.uint8)
    frame[100:220, 40:300] = 0                      # asymmetric, so a mirror shows
    upright = [[40, 100], [560, 100], [560, 220], [40, 220]]
    anticlockwise = [[40, 100], [40, 220], [560, 220], [560, 100]]

    good = rectify_quad(frame, upright)
    raw = rectify_quad(frame, anticlockwise, orient=False)
    assert not np.array_equal(raw, good)
    assert (raw.shape[1], raw.shape[0]) == (good.shape[0], good.shape[1]), (
        "the wide label should have flattened portrait without ordering")
