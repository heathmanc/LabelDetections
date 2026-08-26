from __future__ import annotations

import pytest

from label_detections.core import annotations as ann
from label_detections.core.labels import CodeSpec, LabelDef, TextField
from conftest import frame, label_box, rect


def test_detector_family_and_library_identity_stay_separate():
    """Conflating them is what forces a retrain on every new SKU."""
    box = ann.make_box("spec_plate", rect(0, 0, 10, 10), label_id="spec_plate_31agm")
    assert box["label"] == "spec_plate"
    assert box["label_id"] == "spec_plate_31agm"


def test_inventory_counts_identities_not_families():
    data = frame(boxes=[
        label_box("a", "spec_plate", 0, 0, 10, 10),
        label_box("a", "spec_plate", 20, 0, 10, 10),
        label_box("b", "spec_plate", 40, 0, 10, 10),
    ])
    assert ann.label_inventory(data) == {"a": 2, "b": 1}
    assert ann.family_inventory(data) == {"spec_plate": 3}


def test_unidentified_boxes_are_findable_not_dropped():
    data = frame(boxes=[ann.make_box("spec_plate", rect(0, 0, 10, 10))])
    assert ann.identified_boxes(data) == []
    assert len(ann.unidentified_boxes(data)) == 1


def test_battery_side_is_excluded_from_label_boxes():
    data = frame(boxes=[ann.make_box("battery_side", rect(0, 0, 100, 100)),
                        label_box("a", "spec_plate", 0, 0, 10, 10)])
    assert len(ann.label_boxes(data)) == 1


def test_legacy_axis_aligned_box_still_yields_a_polygon():
    poly = ann.box_polygon({"x": 10, "y": 20, "w": 30, "h": 40})
    assert poly[0] == [10.0, 20.0] and poly[2] == [40.0, 60.0]


def test_box_center_norm_is_a_fraction_of_the_frame():
    # Centre of (100, 50)+200x100 is (200, 100) -> (0.20, 0.20) of a 1000x500 frame.
    box = label_box("a", "spec_plate", 100, 50, 200, 100)
    assert ann.box_center_norm(box, 1000, 500) == pytest.approx((0.2, 0.2))
    assert ann.box_center_norm(box, 0, 0) is None


def label_with_code():
    label = LabelDef(label_id="sp", size_mm=[100.0, 60.0])
    label.codes = [CodeSpec(role="serial", region_mm=[10, 10, 40, 20])]
    label.text_fields = [TextField(name="date_code", region_mm=[10, 40, 50, 10])]
    return label


def test_regions_are_placed_from_the_library_instead_of_drawn():
    """The labeling shortcut: draw four corners, get the barcode box for free."""
    box = label_box("sp", "spec_plate", 100, 100, 200, 120)
    ann.apply_reference_regions(box, label_with_code())
    code = ann.code_region(box, "serial")
    assert code is not None
    assert code["points"][0] == pytest.approx([120.0, 120.0])
    assert code["points"][2] == pytest.approx([200.0, 160.0])
    assert [r["role"] for r in ann.regions(box)] == ["code", "text"]


def test_placement_is_skipped_when_the_label_has_no_size():
    box = label_box("sp", "spec_plate", 0, 0, 10, 10)
    ann.apply_reference_regions(box, LabelDef(label_id="sp"))
    assert ann.regions(box) == []


def test_hand_adjusted_regions_survive_a_replacement_pass():
    """An operator who nudged a region knows more than the library does."""
    box = label_box("sp", "spec_plate", 100, 100, 200, 120)
    box["regions"] = [ann.make_region("code", rect(0, 0, 5, 5), code_role="serial")]
    ann.apply_reference_regions(box, label_with_code())
    code = ann.code_region(box, "serial")
    assert code["points"][0] == [0.0, 0.0]      # the hand-drawn one, untouched


def test_overwrite_replaces_only_reference_placed_regions():
    box = label_box("sp", "spec_plate", 100, 100, 200, 120)
    ann.apply_reference_regions(box, label_with_code())
    hand = ann.make_region("text", rect(1, 1, 2, 2), field="hand_drawn")
    box["regions"].append(hand)
    ann.apply_reference_regions(box, label_with_code(), overwrite=True)
    assert hand in box["regions"]


def test_read_value_prefers_a_decoded_code_over_ocr():
    box = label_box("sp", "spec_plate", 0, 0, 10, 10)
    box["regions"] = [
        ann.make_region("code", rect(0, 0, 5, 5), code_role="serial",
                        decoded="SN1", decode_ok=True),
        ann.make_region("text", rect(0, 0, 5, 5), field="serial", ocr="5N1"),
    ]
    assert ann.read_value(box, "serial") == "SN1"


def test_read_value_falls_back_to_ocr_when_the_code_did_not_decode():
    box = label_box("sp", "spec_plate", 0, 0, 10, 10)
    box["regions"] = [
        ann.make_region("code", rect(0, 0, 5, 5), code_role="serial", decode_ok=False),
        ann.make_region("text", rect(0, 0, 5, 5), field="serial", ocr="SN1"),
    ]
    assert ann.read_value(box, "serial") == "SN1"


def test_new_annotation_records_provenance_for_the_split():
    data = ann.new_annotation("a.jpg", "sp", 100, 50, session="2026-08-14", source="")
    assert data["session"] == "2026-08-14"
    assert "source" not in data      # blanks are not stored
