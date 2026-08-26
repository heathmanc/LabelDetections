from __future__ import annotations

import pytest

from label_detections.core.labels import LabelDef, LabelLibrary
from label_detections.core.recipes import (
    CrossCheck, LabelRequirement, Recipe, ViewSpec, format_count, normalise_roi,
    parse_count, parse_ref, roi_contains, roi_pixels, rois_overlap, validate_recipe,
)


@pytest.mark.parametrize("spec,expected", [
    (1, (1, 1)), ("2", (2, 2)), ("0..1", (0, 1)), ("1..*", (1, None)),
    ("", (1, 1)), ("nonsense", (1, 1)), (0, (0, 0)),
])
def test_parse_count(spec, expected):
    assert parse_count(spec) == expected


def test_format_count_reads_as_english():
    assert format_count(1) == "exactly 1"
    assert format_count("0..1") == "0 to 1"
    assert format_count("1..*") == "at least 1"


def test_normalise_roi_rejects_degenerate_input():
    assert normalise_roi([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]
    assert normalise_roi([0.1, 0.2, 0, 0.4]) == []
    assert normalise_roi("nonsense") == []
    assert normalise_roi(None) == []


def test_empty_roi_means_anywhere():
    assert roi_contains([], 0.99, 0.99)


def test_roi_contains_respects_slack():
    roi = [0.1, 0.1, 0.2, 0.2]
    assert roi_contains(roi, 0.2, 0.2)
    assert not roi_contains(roi, 0.4, 0.2)
    assert roi_contains(roi, 0.31, 0.2, tol=0.02)


def test_roi_pixels_scales_to_the_frame():
    assert roi_pixels([0.1, 0.2, 0.3, 0.4], 1000, 500) == [100, 100, 300, 200]
    assert roi_pixels([], 640, 480) == [0, 0, 640, 480]


def test_rois_overlap():
    assert rois_overlap([0, 0, 0.5, 0.5], [0.4, 0.4, 0.3, 0.3])
    assert not rois_overlap([0, 0, 0.3, 0.3], [0.5, 0.5, 0.3, 0.3])


def test_safe_name_keeps_the_legacy_form_for_the_default_category():
    assert Recipe(group="AGM", model="31-950").safe_name == "AGM__31-950"
    assert Recipe(group="AGM", model="31-950",
                  category="Industrial").safe_name == "Industrial__AGM__31-950"


def test_round_trips_through_json_shape():
    recipe = Recipe(group="g", model="m", views=[
        ViewSpec(view="v", labels=[LabelRequirement("a", roi=[0.1, 0.1, 0.2, 0.2])],
                 forbidden=["b"])],
        cross_checks=[CrossCheck(left="v.a.serial", right="v.a.lot")])
    back = Recipe.from_dict(recipe.to_dict())
    assert back.view("v").labels[0].roi == [0.1, 0.1, 0.2, 0.2]
    assert back.view("v").forbidden == ["b"]
    assert back.cross_checks[0].left == "v.a.serial"


def test_label_ids_lists_everything_that_needs_training():
    recipe = Recipe(group="g", model="m", views=[
        ViewSpec(view="v", labels=[LabelRequirement("a")], forbidden=["b"])])
    assert recipe.label_ids() == {"a", "b"}


def test_parse_ref():
    assert parse_ref("side_a.plate.serial") == ("side_a", "plate", "serial")
    assert parse_ref("side_a.plate") is None


def test_validate_flags_a_label_nobody_trained():
    recipe = Recipe(group="g", model="m",
                    views=[ViewSpec(view="v", labels=[LabelRequirement("ghost")])])
    issues = validate_recipe(recipe, LabelLibrary([]))
    assert any("not in the label library" in i for i in issues)


def test_validate_flags_a_pixel_roi_typed_as_fractions():
    recipe = Recipe(group="g", model="m", views=[
        ViewSpec(view="v", labels=[LabelRequirement("a", roi=[20, 15, 90, 60])])])
    assert any("outside the frame" in i for i in validate_recipe(recipe))


def test_validate_flags_required_and_forbidden_together():
    recipe = Recipe(group="g", model="m", views=[
        ViewSpec(view="v", labels=[LabelRequirement("a")], forbidden=["a"])])
    assert any("both required and forbidden" in i for i in validate_recipe(recipe))


def test_validate_flags_a_cross_check_that_can_never_fire():
    """A role the label does not carry fails silently in production, not loudly."""
    lib = LabelLibrary([LabelDef(label_id="a")])
    recipe = Recipe(group="g", model="m",
                    views=[ViewSpec(view="v", labels=[LabelRequirement("a")])],
                    cross_checks=[CrossCheck(left="v.a.serial", right="v.a.lot")])
    issues = validate_recipe(recipe, lib)
    assert any("can never fire" in i for i in issues)


def test_validate_flags_duplicate_views_and_labels():
    recipe = Recipe(group="g", model="m", views=[
        ViewSpec(view="v", labels=[LabelRequirement("a"), LabelRequirement("a")]),
        ViewSpec(view="v")])
    issues = validate_recipe(recipe)
    assert any("Duplicate view" in i for i in issues)
    assert any("listed twice" in i for i in issues)
