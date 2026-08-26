from __future__ import annotations

import pytest

from label_detections.core.labels import (
    CodeSpec, LabelDef, LabelLibrary, TextField, validate_label_def,
)


def test_round_trips_through_json_shape():
    label = LabelDef(label_id="sp", size_mm=[90, 60], reference_images=["a.png"],
                     confusable_with=["other"])
    label.codes = [CodeSpec(role="serial", region_mm=[1, 2, 3, 4])]
    label.text_fields = [TextField(name="date_code")]
    back = LabelDef.from_dict(label.to_dict())
    assert back.codes[0].role == "serial"
    assert back.text_fields[0].name == "date_code"
    assert back.confusable_with == ["other"]


def test_from_dict_ignores_unknown_keys_from_a_newer_file():
    label = LabelDef.from_dict({"label_id": "sp", "invented_later": True})
    assert label.label_id == "sp"


def test_missing_size_is_flagged_because_it_is_the_scale_check():
    issues = validate_label_def(LabelDef(label_id="sp", reference_images=["a.png"]))
    assert any("size" in i.lower() for i in issues)


def test_variable_data_without_an_anchor_is_flagged():
    label = LabelDef(label_id="sp", size_mm=[10, 10], reference_images=["a.png"],
                     variable_data=True)
    assert any("anchor" in i.lower() for i in validate_label_def(label))


def test_must_decode_code_without_a_region_is_flagged():
    label = LabelDef(label_id="sp", size_mm=[10, 10], reference_images=["a.png"])
    label.codes = [CodeSpec(role="serial", policy="must_decode", x_dim_mm=0.254)]
    assert any("region" in i.lower() for i in validate_label_def(label))


def test_min_pixels_needed_scales_with_symbology():
    """2D codes need more pixels per module than 1D, and the number says so."""
    one_d = CodeSpec(symbology="code128", region_mm=[0, 0, 40, 12], x_dim_mm=0.254)
    two_d = CodeSpec(symbology="datamatrix", region_mm=[0, 0, 40, 12], x_dim_mm=0.254)
    assert two_d.min_pixels_needed() > one_d.min_pixels_needed()
    assert one_d.min_pixels_needed() == pytest.approx(40 / 0.254 * 2, rel=1e-6)


def test_match_region_uses_the_anchor_only_for_variable_data():
    static = LabelDef(label_id="a", size_mm=[90, 60])
    variable = LabelDef(label_id="b", size_mm=[90, 60], variable_data=True,
                        anchor_region_mm=[0, 0, 40, 20])
    assert static.match_region_mm() == [0.0, 0.0, 90.0, 60.0]
    assert variable.match_region_mm() == [0, 0, 40, 20]


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
