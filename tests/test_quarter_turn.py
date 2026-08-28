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
