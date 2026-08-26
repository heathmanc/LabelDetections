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
    label = LabelDef(label_id="sp", reference_images=["a.png"])
    assert validate_label_def(label) == []


def test_missing_reference_is_flagged_because_regions_are_drawn_on_it():
    issues = validate_label_def(LabelDef(label_id="sp"))
    assert any("reference image" in i.lower() for i in issues)


def test_variable_data_without_an_anchor_is_flagged():
    label = LabelDef(label_id="sp", reference_images=["a.png"], variable_data=True)
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
