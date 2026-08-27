from __future__ import annotations

import pytest

from label_detections.core.labels import (
    CodeSpec, LabelDef, LabelLibrary, TextField, validate_label_def,
)


def test_round_trips_through_json_shape():
    label = LabelDef(label_id="sp", reference_images=["a.png"],
                     confusable_with=["other"])
    label.codes = [CodeSpec(role="serial", region=[0.1, 0.2, 0.3, 0.4])]
    label.text_fields = [TextField(name="date_code", region=[0.1, 0.6, 0.5, 0.2])]
    back = LabelDef.from_dict(label.to_dict())
    assert back.codes[0].role == "serial"
    assert back.codes[0].region == [0.1, 0.2, 0.3, 0.4]
    assert back.text_fields[0].name == "date_code"
    assert back.confusable_with == ["other"]


def test_from_dict_ignores_unknown_keys_from_a_newer_file():
    label = LabelDef.from_dict({"label_id": "sp", "invented_later": True})
    assert label.label_id == "sp"


def test_a_label_needs_no_physical_size():
    """Nothing computes with it: region placement is proportional."""
    assert validate_label_def(LabelDef(label_id="sp")) == []


def test_a_label_with_nothing_to_read_needs_no_artwork():
    """Artwork comes from a capture, so it is not a prerequisite for a label."""
    assert validate_label_def(LabelDef(label_id="sp")) == []


def test_a_label_that_reads_something_needs_artwork_to_position_it_on():
    label = LabelDef(label_id="sp")
    label.codes = [CodeSpec(role="serial", region=[0.1, 0.1, 0.2, 0.2])]
    assert any("Define Regions" in i for i in validate_label_def(label))


def test_variable_data_without_an_anchor_is_flagged():
    label = LabelDef(label_id="sp", variable_data=True)
    assert any("anchor" in i.lower() for i in validate_label_def(label))


def test_must_decode_code_without_a_region_is_flagged():
    label = LabelDef(label_id="sp", reference_images=["a.png"])
    label.codes = [CodeSpec(role="serial", policy="must_decode")]
    assert any("region" in i.lower() for i in validate_label_def(label))


def test_a_region_outside_the_label_is_rejected():
    label = LabelDef(label_id="sp", reference_images=["a.png"])
    label.codes = [CodeSpec(role="serial", region=[0.8, 0.1, 0.4, 0.2])]
    assert any("region" in i.lower() for i in validate_label_def(label))


def test_a_text_field_with_no_region_has_nowhere_to_read_from():
    label = LabelDef(label_id="sp", reference_images=["a.png"])
    label.text_fields = [TextField(name="date_code")]
    assert any("nowhere to read" in i for i in validate_label_def(label))


def test_min_pixels_needed_scales_with_symbology():
    """2D codes need more pixels per module than 1D, and the number says so."""
    one_d = CodeSpec(symbology="code128", code_width_mm=40, x_dim_mm=0.254)
    two_d = CodeSpec(symbology="datamatrix", code_width_mm=40, x_dim_mm=0.254)
    assert two_d.min_pixels_needed() > one_d.min_pixels_needed()
    assert one_d.min_pixels_needed() == pytest.approx(40 / 0.254 * 2, rel=1e-6)


def test_min_pixels_is_silent_without_the_print_spec():
    """It is an optional hint, not a gate -- both numbers are off the print job."""
    assert CodeSpec(symbology="qr", region=[0.1, 0.1, 0.2, 0.2]).min_pixels_needed() == 0.0


def test_match_region_uses_the_anchor_only_for_variable_data():
    static = LabelDef(label_id="a")
    variable = LabelDef(label_id="b", variable_data=True,
                        anchor_region=[0.0, 0.0, 0.4, 0.2])
    assert static.match_region() == [0.0, 0.0, 1.0, 1.0]
    assert variable.match_region() == [0.0, 0.0, 0.4, 0.2]


def test_regions_lists_everything_drawn():
    label = LabelDef(label_id="a", variable_data=True, anchor_region=[0, 0, 0.5, 0.5])
    label.codes = [CodeSpec(role="serial", region=[0.1, 0.1, 0.2, 0.2])]
    label.text_fields = [TextField(name="date_code", region=[0.1, 0.5, 0.3, 0.1])]
    roles = [r[0] for r in label.regions()]
    assert roles == ["code", "text", "anchor"]


def test_an_older_sidecar_with_mm_keys_still_loads():
    """Fields the schema dropped are ignored rather than crashing the library."""
    label = LabelDef.from_dict({
        "label_id": "sp",
        "codes": [{"role": "serial", "region_mm": [1, 2, 3, 4], "symbology": "qr"}],
    })
    assert label.codes[0].symbology == "qr"
    assert label.codes[0].region == []


def test_library_add_get_remove():
    lib = LabelLibrary()
    lib.add(LabelDef(label_id="a"))
    assert "a" in lib and lib.get("a").label_id == "a"
    with pytest.raises(ValueError):
        lib.add(LabelDef(label_id="a"))
    lib.add(LabelDef(label_id="a", name="new"), replace=True)
    assert lib.get("a").name == "new"
    assert lib.remove("a") and not lib.remove("a")


def test_families_in_use_always_includes_the_battery_face():
    lib = LabelLibrary([LabelDef(label_id="a", family="cert_mark")])
    assert lib.families_in_use() == ["battery_side", "cert_mark"]


# --- searching a library that has grown large -------------------------------

def _library_of(*specs):
    return LabelLibrary([
        LabelDef(label_id=i, name=n, family=f, revision=r, part_number=p)
        for i, n, f, r, p in specs
    ])


LIB = _library_of(
    ("warning_g31_en", "G31 warning, English", "warning_label", "C", "LBL-8871"),
    ("warning_g31_fr", "G31 warning, French", "warning_label", "C", "LBL-8872"),
    ("spec_plate_g31", "G31 spec plate", "spec_plate", "D", "LBL-4410"),
    ("spec_plate_g27", "G27 spec plate", "spec_plate", "A", "LBL-4409"),
    ("ul_mark", "UL certification mark", "cert_mark", "", "LBL-0031"),
)


def test_an_empty_query_matches_everything():
    assert len(LIB.search("")) == 5
    assert len(LIB.search("   ")) == 5


def test_every_term_must_appear_but_the_order_does_not_matter():
    """The operator remembers something about the label, not its exact id."""
    both = ["warning_g31_en", "warning_g31_fr"]
    assert [l.label_id for l in LIB.search("g31 warn")] == both
    assert [l.label_id for l in LIB.search("warn g31")] == both


def test_a_term_can_match_any_field():
    assert [l.label_id for l in LIB.search("french")] == ["warning_g31_fr"]
    assert [l.label_id for l in LIB.search("LBL-4410")] == ["spec_plate_g31"]
    assert [l.label_id for l in LIB.search("cert_mark")] == ["ul_mark"]
    assert len(LIB.search("D")) >= 1          # revision


def test_matching_is_case_insensitive():
    assert [l.label_id for l in LIB.search("G31 SPEC")] == ["spec_plate_g31"]


def test_terms_that_match_nothing_return_nothing():
    assert LIB.search("g31 forklift") == []


def test_the_family_filter_narrows_the_search_rather_than_replacing_it():
    assert [l.label_id for l in LIB.search("g31", "spec_plate")] == ["spec_plate_g31"]
    assert len(LIB.search("", "warning_label")) == 2
