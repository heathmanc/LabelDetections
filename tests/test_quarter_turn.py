"""A label photographed standing up, and the regions that follow it.

``order_quad`` normalises a quad to the IMAGE's top-left. That fixes winding
and mirroring, and it is not enough on its own: which physical corner of the
label is topmost-leftmost depends on how the label was lying. Lay a long label
flat and its top-left corner leads into the long edge; stand the same label up
and the top-left corner leads into the short edge instead. Both orders are
legal and clockwise -- but the unit square has been laid on the label rotated a
quarter turn, so a region drawn on landscape artwork comes out across the
label's width instead of along its length: right size, wrong shape, wrong end.

Proportion survives rotation, which is why the artwork's aspect is what settles
it. It cannot settle which END is the top -- that is ``flipped``, decided by
reading.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import annotations as ann
from label_detections.core import geometry as geo
from label_detections.core import reference as reference_logic
from label_detections.core.labels import LabelDef, TextField

# A label three and a third times as wide as it is tall, and a tall narrow
# field near its right-hand end -- the part number down the end of a battery
# side label.
ASPECT = 1000.0 / 300.0
FIELD = [0.842, 0.239, 0.151, 0.197]

FLAT = [[0.0, 0.0], [1000.0, 0.0], [1000.0, 300.0], [0.0, 300.0]]
STANDING = [[0.0, 0.0], [300.0, 0.0], [300.0, 1000.0], [0.0, 1000.0]]


def _bounds(placed):
    xs = [x for x, _ in placed]
    ys = [y for _, y in placed]
    return (round(min(xs)), round(max(xs)), round(min(ys)), round(max(ys)))


# --- the geometry ------------------------------------------------------------

def test_a_flat_label_is_already_right_and_is_left_alone():
    """The aspect matches the way the corners already read, so nothing moves --
    every label that was placing correctly must keep placing identically."""
    assert geo.align_quad(FLAT, ASPECT) == geo.order_quad(FLAT)


def test_a_standing_label_reads_its_long_edge_as_the_width():
    """The correction itself. Before it, the quad read 300 wide by 1000 tall
    against artwork that is 1000 by 300 -- the unit square a quarter turn out."""
    assert round(geo.quad_aspect(geo.order_quad(STANDING)), 3) == 0.3
    assert round(geo.quad_aspect(geo.align_quad(STANDING, ASPECT)), 3) == 3.333


def test_the_field_runs_along_the_label_not_across_it():
    """The symptom the operator saw: a tall field drawn near one end of the
    label appearing as a wide box squatting at the top of a standing one."""
    wrong = geo.place_unit_rect(geo.order_quad(STANDING), FIELD, orient=False)
    right = geo.place_unit_rect(geo.align_quad(STANDING, ASPECT), FIELD, orient=False)

    # Unaligned: across the label's width, up at the top. Which is where the
    # screenshot showed it.
    assert _bounds(wrong) == (253, 298, 239, 436)
    # Aligned: down the far end of the label, hugging one side.
    assert _bounds(right) == (169, 228, 842, 993)


def test_both_readings_stay_inside_the_label():
    """It is not that the region escaped the box -- it is that it landed in the
    wrong part of it. Nothing here is out of bounds, which is why this was
    invisible to any check that only asks whether a region fits."""
    for placed in (geo.place_unit_rect(geo.order_quad(STANDING), FIELD, orient=False),
                   geo.place_unit_rect(geo.align_quad(STANDING, ASPECT), FIELD, orient=False)):
        assert all(0 <= x <= 300 and 0 <= y <= 1000 for x, y in placed)


def test_alignment_keeps_the_corners_wound_clockwise():
    """Rotating the order must not re-wind the quad: an anticlockwise quad
    flattens to a mirror image, and a mirrored barcode does not decode."""
    def shoelace(quad):
        return sum(quad[i][0] * quad[(i + 1) % 4][1]
                   - quad[(i + 1) % 4][0] * quad[i][1] for i in range(4))

    # Positive with y pointing down the image is clockwise on screen, which is
    # what TL/TR/BR/BL gives.
    assert shoelace(geo.order_quad(FLAT)) > 0
    assert shoelace(geo.align_quad(STANDING, ASPECT)) > 0


def test_a_square_label_is_left_alone():
    """Nothing to match. Rotating a square-ish quad on the strength of a couple
    of pixels of keystone would move regions for no reason at all."""
    square = [[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]]
    assert geo.align_quad(square, 1.02) == geo.order_quad(square)
    assert geo.align_quad(STANDING, 1.02) == geo.order_quad(STANDING)


def test_no_aspect_means_the_old_behaviour_exactly():
    """A label whose artwork has gone, or one from before this was recorded."""
    assert geo.align_quad(STANDING, 0.0) == geo.order_quad(STANDING)
    assert geo.oriented(STANDING) == geo.order_quad(STANDING)


def test_the_half_turn_is_still_left_to_reading():
    """Alignment picks between the two readings that have the right shape. The
    remaining pair differ by 180 degrees, and no arrangement of four corners
    says which way up printing is."""
    upright = geo.oriented(STANDING, False, ASPECT)
    turned = geo.oriented(STANDING, True, ASPECT)
    assert turned == geo.flip_quad(upright)
    assert _bounds(geo.place_unit_rect(turned, FIELD, orient=False)) == (72, 131, 7, 158)


# --- where the aspect comes from --------------------------------------------

def test_a_label_reports_the_aspect_it_recorded():
    assert reference_logic.aspect_of(LabelDef(label_id="x", reference_aspect=3.5)) == 3.5


def test_a_label_with_no_artwork_reports_nothing():
    assert reference_logic.aspect_of(LabelDef(label_id="x")) == 0.0


def test_artwork_already_on_disk_is_measured(tmp_path):
    """Labels defined before this was recorded still have a shaped reference
    image sitting there, so nobody has to re-capture one."""
    import struct
    import zlib

    png = tmp_path / "art.png"
    ihdr = struct.pack(">IIBBBBB", 900, 300, 8, 2, 0, 0, 0)
    png.write_bytes(b"\x89PNG\r\n\x1a\n"
                    + struct.pack(">I", len(ihdr)) + b"IHDR" + ihdr
                    + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr)))
    assert reference_logic.image_size(png) == (900, 300)
    assert reference_logic.aspect_of(
        LabelDef(label_id="x", reference_images=[str(png)])) == 3.0


def test_something_that_is_not_a_png_measures_nothing(tmp_path):
    junk = tmp_path / "art.png"
    junk.write_bytes(b"not an image at all, but long enough to read 24 bytes")
    assert reference_logic.image_size(junk) == (0, 0)
    assert reference_logic.image_size(tmp_path / "absent.png") == (0, 0)


# --- placing regions on a detection ------------------------------------------

def _label(aspect=ASPECT):
    label = LabelDef(label_id="PC680", reference_aspect=aspect)
    label.text_fields = [TextField(name="part_number", region=list(FIELD))]
    return label


def _box(quad):
    return {"kind": "obb", "label": "label", "label_id": "PC680",
            "points": [list(p) for p in quad]}


def test_placing_uses_the_labels_own_proportions():
    placed = ann.place_label_regions(_box(STANDING), _label())
    assert _bounds(placed[0]["points"]) == (169, 228, 842, 993)


def test_a_label_with_no_recorded_aspect_places_as_before():
    placed = ann.place_label_regions(_box(STANDING), _label(aspect=0.0))
    assert _bounds(placed[0]["points"]) == (253, 298, 239, 436)


def test_a_flat_detection_is_unaffected_either_way():
    with_aspect = ann.place_label_regions(_box(FLAT), _label())
    without = ann.place_label_regions(_box(FLAT), _label(aspect=0.0))
    assert with_aspect == without


# --- crops the classifier sees -----------------------------------------------

def test_a_crop_is_flattened_the_same_way_whichever_pose_it_was_shot_in():
    """The same label lying flat and standing up otherwise flattens to a 300x90
    and a 90x300 -- one picture turned a quarter, and two pictures to anything
    learning from them. It had to learn the label twice, out of a dataset that
    did not have enough for once."""
    cv2 = __import__("pytest").importorskip("cv2")
    np = __import__("pytest").importorskip("numpy")
    from label_detections.core.imageio import rectify_quad

    lying = np.zeros((900, 900, 3), np.uint8)
    lying[100:190, 100:400] = 255
    standing = np.zeros((900, 900, 3), np.uint8)
    standing[100:400, 100:190] = 255
    flat_quad = [[100., 100.], [400., 100.], [400., 190.], [100., 190.]]
    tall_quad = [[100., 100.], [190., 100.], [190., 400.], [100., 400.]]

    assert rectify_quad(lying, flat_quad).shape[:2] == (90, 300)
    assert rectify_quad(standing, tall_quad).shape[:2] == (300, 90), "the old way"

    assert rectify_quad(lying, flat_quad, landscape=True).shape[:2] == (90, 300)
    assert rectify_quad(standing, tall_quad, landscape=True).shape[:2] == (90, 300)


def test_turning_a_crop_flat_says_nothing_about_which_way_up_it_is():
    """It claims only that the long side runs across, which is true of a label
    however it was presented. The half-turn stays with whatever reads it."""
    assert geo.landscape_quad(STANDING) == geo.flip_quad(
        geo.flip_quad(geo.landscape_quad(STANDING)))
    assert round(geo.quad_aspect(geo.landscape_quad(STANDING)), 2) == \
        round(geo.quad_aspect(geo.landscape_quad(FLAT)), 2)


def test_a_square_crop_is_left_alone():
    square = [[0.0, 0.0], [400.0, 0.0], [400.0, 400.0], [0.0, 400.0]]
    assert geo.landscape_quad(square) == geo.order_quad(square)


def test_training_and_runtime_crops_are_taken_the_same_way():
    """They must agree or the classifier is handed a pose it was never shown --
    which is worse than the problem this fixes, because it is invisible."""
    import inspect

    from label_detections.core import classify_export
    from label_detections.ui import live_detect as live_ui

    for source in (inspect.getsource(classify_export._write_crops),
                   inspect.getsource(classify_export.write_region_dataset),
                   inspect.getsource(live_ui.InferenceWorker._identify)):
        assert "landscape=True" in source


# --- the band where turning a crop flat is not repeatable ---------------------

def _tilted(aspect: float, tilt: float):
    """A standing label at ``aspect``, foreshortened by ``tilt``."""
    side = 100.0 * aspect * tilt
    return [[0.0, 0.0], [100.0, 0.0], [100.0, side], [0.0, side]]


def _turned(aspect: float, tilt: float) -> bool:
    q = _tilted(aspect, tilt)
    return geo.landscape_quad(q) != geo.order_quad(q)


def test_a_square_label_is_never_turned_however_it_is_tilted():
    """The band cannot be removed, only moved -- so it is moved off the shapes
    somebody is most likely to enrol next: square stickers and round decals in
    square die-cuts. At the old 1.15 threshold every one of those was in it."""
    for aspect in (1.0, 1.05, 1.15):
        assert {_turned(aspect, t) for t in (0.87, 1.0, 1.15)} == {False}


def test_a_clearly_long_label_is_always_turned_however_it_is_tilted():
    for aspect in (1.8, 2.0, 3.3, 6.0):
        assert {_turned(aspect, t) for t in (0.87, 1.0, 1.15)} == {True}


def test_every_shape_that_is_not_repeatable_is_one_the_operator_was_warned_about():
    """The guarantee that makes the band acceptable: nothing is inconsistent
    without having been named at enrolment."""
    for aspect in [1.0 + i / 100 for i in range(0, 250)]:
        consistent = len({_turned(aspect, t) for t in (0.87, 1.0, 1.15)}) == 1
        if not consistent:
            assert not geo.landscape_is_stable(aspect), \
                f"aspect {aspect:.2f} flips untold"


def test_the_band_is_stated_rather_than_left_implicit():
    low, high = geo.landscape_band()
    assert low < geo.LANDSCAPE_TOL < high
    assert not geo.landscape_is_stable((low + high) / 2)
    assert geo.landscape_is_stable(1.0) and geo.landscape_is_stable(10.0)
    # And it reads the same for a portrait label as for its landscape mirror.
    assert geo.landscape_is_stable(0.2) and not geo.landscape_is_stable(1 / 1.4)


def test_placing_a_region_keeps_its_own_narrower_tolerance():
    """Two different questions. Placing a region compares against the artwork's
    own aspect, which is recorded and exact; turning a crop compares a measured
    quad against nothing, which is why it needs more room."""
    assert geo.SQUARE_TOL < geo.LANDSCAPE_TOL
    # _tilted builds a STANDING label, so this quad reads 0.8:1.
    quad = _tilted(1.25, 1.0)
    assert round(geo.quad_aspect(geo.order_quad(quad)), 2) == 0.8
    # Against 1.25:1 artwork it is a quarter turn out, and placing corrects it:
    # that comparison is against a recorded, exact number.
    assert geo.align_quad(quad, 1.25) != geo.order_quad(quad)
    # Against 0.8:1 artwork it already agrees.
    assert geo.align_quad(quad, 0.8) == geo.order_quad(quad)
    # But its CROP is left alone either way, because that decision compares the
    # measurement against nothing and 1.25 is inside the band tilt can cross.
    assert geo.landscape_quad(quad) == geo.order_quad(quad)


def test_a_near_square_label_is_told_at_enrolment():
    """Discovered later it is unexplained classifier noise. Said here it is a
    known property of one label, with a reason and a thing to do about it."""
    warning = reference_logic.crop_shape_warning(1.4)
    assert "close to square" in warning
    assert "1.22" in warning and "1.61" in warning
    assert "square-on to the camera" in warning
    assert "still works" in warning, "it is a caveat, not a refusal"


def test_a_label_of_any_usable_shape_is_not_warned_about():
    for aspect in (0.0, 1.0, 1.1, 1.9, 3.3):
        assert reference_logic.crop_shape_warning(aspect) == ""
