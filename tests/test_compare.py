"""The inspection engine, exercised as production scenarios."""
from __future__ import annotations

import pytest

from label_detections.core import annotations as ann
from label_detections.core import compare as cmp
from label_detections.core.recipes import CrossCheck, LabelRequirement, Recipe, ViewSpec
from conftest import FRAME, code_region, frame, label_box, rect


def good_side_a():
    # Centre (200, 150) -> (0.20, 0.30) of frame, inside ROI [0.05, 0.1, 0.3, 0.4].
    plate = label_box("spec_plate_31agm", "spec_plate", 125, 100, 150, 100)
    plate["regions"] = [code_region("serial", "SN123456"),
                        ann.make_region("text", rect(0, 0, 5, 5), field="date_code",
                                        ocr="2026-08-14")]
    # Centre (600, 100) -> (0.60, 0.20), inside ROI [0.5, 0.1, 0.2, 0.3].
    warn = label_box("warning_en", "warning_label", 560, 70, 80, 60)
    return frame("side_a", boxes=[plate, warn])


def good_side_b():
    tag = label_box("trace_tag", "trace_tag", 200, 100, 100, 50)
    tag["regions"] = [code_region("serial", "SN123456")]
    return frame("side_b", boxes=[tag])


def good_unit():
    return {"side_a": good_side_a(), "side_b": good_side_b()}


# --- the happy path --------------------------------------------------------

def test_matching_unit_passes(recipe, library):
    result = cmp.compare_unit(good_unit(), recipe, library, "U1")
    assert result.verdict == cmp.PASS, result.summary_text()
    assert result.all_findings() == []


def test_summary_text_lists_every_requirement(recipe, library):
    text = cmp.compare_unit(good_unit(), recipe, library, "U1").summary_text()
    assert "spec_plate_31agm" in text and "trace_tag" in text
    assert "PASS" in text


# --- the failure the whole system exists to catch --------------------------

def test_missing_label_is_a_failure_naming_the_label(recipe, library):
    views = good_unit()
    views["side_a"]["boxes"] = [b for b in views["side_a"]["boxes"]
                                if b.get("label_id") != "warning_en"]
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert result.verdict == cmp.FAIL
    codes = [f.code for f in result.failures()]
    assert cmp.MISSING in codes
    assert any("warning_en" in f.message for f in result.failures())


def test_missing_label_is_found_even_with_nothing_detected(recipe, library):
    """A blank battery must fail loudly, not pass for lack of detections."""
    views = {"side_a": frame("side_a"), "side_b": frame("side_b")}
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert result.verdict == cmp.FAIL
    assert len([f for f in result.failures() if f.code == cmp.MISSING]) == 3


def test_duplicate_label_is_a_count_failure(recipe, library):
    views = good_unit()
    views["side_a"]["boxes"].append(label_box("warning_en", "warning_label", 570, 80, 80, 60))
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert result.verdict == cmp.FAIL
    assert cmp.WRONG_COUNT in [f.code for f in result.failures()]


def test_range_count_accepts_both_ends(library):
    view = ViewSpec(view="v", labels=[LabelRequirement("promo", count="0..1")])
    recipe = Recipe(group="g", model="m", views=[view])
    none = {"v": frame("v")}
    one = {"v": frame("v", boxes=[label_box("promo", "promo_label", 0, 0, 60, 40)])}
    two = {"v": frame("v", boxes=[label_box("promo", "promo_label", 0, 0, 60, 40),
                                  label_box("promo", "promo_label", 90, 0, 60, 40)])}
    assert cmp.compare_unit(none, recipe, library).verdict == cmp.PASS
    assert cmp.compare_unit(one, recipe, library).verdict == cmp.PASS
    assert cmp.compare_unit(two, recipe, library).verdict == cmp.FAIL


# --- the wrong-label case --------------------------------------------------

def test_forbidden_lookalike_fails_even_though_everything_required_is_present(recipe, library):
    """The neighbouring model's plate is present and correct-looking.

    Nothing in the required bill notices it, which is exactly why the
    forbidden list exists.
    """
    views = good_unit()
    views["side_a"]["boxes"].append(
        label_box("spec_plate_27agm", "spec_plate", 700, 300, 150, 100))
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert result.verdict == cmp.FAIL
    assert cmp.FORBIDDEN in [f.code for f in result.failures()]


def test_unexpected_label_warns_but_does_not_fail(recipe, library):
    views = good_unit()
    views["side_a"]["boxes"].append(label_box("promo", "promo_label", 800, 350, 60, 40))
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert result.verdict == cmp.WARN
    assert cmp.UNEXPECTED in [f.code for f in result.all_findings()]


def test_unidentified_detection_is_never_silently_dropped(recipe, library):
    """A detected label nobody could name is a new SKU or a wrong label."""
    views = good_unit()
    views["side_a"]["boxes"].append(ann.make_box("spec_plate", rect(700, 350, 150, 100)))
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.UNIDENTIFIED in [f.code for f in result.all_findings()]


# --- placement -------------------------------------------------------------

def test_label_outside_its_roi_fails(recipe, library):
    views = good_unit()
    for box in views["side_a"]["boxes"]:
        if box.get("label_id") == "spec_plate_31agm":
            box["points"] = rect(700, 300, 150, 100)      # far side of the frame
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.OUT_OF_ROI in [f.code for f in result.failures()]


def test_label_just_inside_roi_slack_passes(recipe, library):
    """20 px right of the ROI edge on a 1000 px frame is 0.02 -- exactly the slack."""
    views = good_unit()
    for box in views["side_a"]["boxes"]:
        if box.get("label_id") == "spec_plate_31agm":
            # Centre at x = 0.36 of frame; ROI ends at 0.35, slack is 0.02.
            box["points"] = rect(285, 100, 150, 100)
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.OUT_OF_ROI not in [f.code for f in result.all_findings()]


def test_roi_check_is_resolution_independent(recipe, library):
    """The same battery shot at double resolution must give the same verdict."""
    views = good_unit()
    for data in views.values():
        data["width"] *= 2
        data["height"] *= 2
        for box in data["boxes"]:
            box["points"] = [[p[0] * 2, p[1] * 2] for p in box["points"]]
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert result.verdict == cmp.PASS, result.summary_text()


def test_requirement_without_an_roi_accepts_a_label_anywhere(recipe, library):
    recipe.view("side_a").labels[0].roi = []
    views = good_unit()
    for box in views["side_a"]["boxes"]:
        if box.get("label_id") == "spec_plate_31agm":
            box["points"] = rect(800, 380, 150, 100)
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.OUT_OF_ROI not in [f.code for f in result.all_findings()]


def test_rotated_label_fails_its_rotation_tolerance(recipe, library):
    views = good_unit()
    for box in views["side_a"]["boxes"]:
        if box.get("label_id") == "warning_en":
            # Same centre, turned 90 degrees.
            box["points"] = [[630, 60], [630, 140], [570, 140], [570, 60]]
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.ROTATED in [f.code for f in result.failures()]


def test_merged_detection_warns_on_shape_without_needing_mm_calibration(recipe, library):
    """A box that swallowed a neighbour has a badly wrong aspect ratio."""
    views = good_unit()
    for box in views["side_a"]["boxes"]:
        if box.get("label_id") == "warning_en":
            box["points"] = rect(500, 70, 400, 60)
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.WRONG_SHAPE in [f.code for f in result.all_findings()]


def test_unknown_frame_size_reports_rather_than_guessing(recipe, library):
    views = good_unit()
    views["side_a"]["width"] = 0
    views["side_a"]["height"] = 0
    recipe.view("side_a").frame_size = []
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.NO_FRAME_SIZE in [f.code for f in result.all_findings()]


def test_frame_size_falls_back_to_the_recipe(recipe, library):
    """A detection payload without image dimensions still gets checked."""
    views = good_unit()
    for data in views.values():
        data.pop("width")
        data.pop("height")
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert result.verdict == cmp.PASS, result.summary_text()


# --- codes -----------------------------------------------------------------

def test_undecoded_code_fails(recipe, library):
    views = good_unit()
    for box in views["side_b"]["boxes"]:
        if box.get("label_id") == "trace_tag":
            box["regions"] = [code_region("serial", "", ok=False, px_per_module=1.4)]
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.CODE_UNREADABLE in [f.code for f in result.failures()]


def test_code_content_must_match_its_pattern(recipe, library):
    views = good_unit()
    for box in views["side_a"]["boxes"]:
        if box.get("label_id") == "spec_plate_31agm":
            box["regions"] = [code_region("serial", "XX999")]
    result = cmp.compare_unit(views, recipe, library, "U1")
    codes = [f.code for f in result.failures()]
    assert cmp.CODE_PATTERN in codes


def test_absent_code_region_is_reported(recipe, library):
    views = good_unit()
    for box in views["side_b"]["boxes"]:
        if box.get("label_id") == "trace_tag":
            box["regions"] = []
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.CODE_MISSING in [f.code for f in result.failures()]


def test_recipe_can_relax_a_code_policy(recipe, library):
    """One recipe demands a decode where another only needs the code present."""
    view = recipe.view("side_b")
    view.labels[0].code_policy = {"serial": "must_be_present"}
    views = good_unit()
    for box in views["side_b"]["boxes"]:
        if box.get("label_id") == "trace_tag":
            box["regions"] = [code_region("serial", "", ok=False)]
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.CODE_UNREADABLE not in [f.code for f in result.all_findings()]


# --- cross-view checks -----------------------------------------------------

def test_serials_that_disagree_across_cameras_fail(recipe, library):
    """No single image sees both labels. Only the unit-level check can catch this."""
    views = good_unit()
    for box in views["side_b"]["boxes"]:
        if box.get("label_id") == "trace_tag":
            box["regions"] = [code_region("serial", "SN999999")]
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert result.verdict == cmp.FAIL
    failure = next(f for f in result.failures() if f.code == cmp.CROSS_CHECK)
    assert "SN123456" in failure.message and "SN999999" in failure.message


def test_cross_check_with_an_unread_value_fails_rather_than_passing_quietly(recipe, library):
    views = good_unit()
    for box in views["side_b"]["boxes"]:
        if box.get("label_id") == "trace_tag":
            box["regions"] = [code_region("serial", "", ok=False)]
    result = cmp.compare_unit(views, recipe, library, "U1")
    assert cmp.CROSS_CHECK in [f.code for f in result.failures()]


def test_pattern_cross_check(library):
    view = ViewSpec(view="v", labels=[LabelRequirement("trace_tag")])
    recipe = Recipe(group="g", model="m", views=[view],
                    cross_checks=[CrossCheck(type="pattern", left="v.trace_tag.serial",
                                             pattern=r"^SN\d{6}$")])
    tag = label_box("trace_tag", "trace_tag", 0, 0, 100, 50)
    tag["regions"] = [code_region("serial", "NOPE")]
    result = cmp.compare_unit({"v": frame("v", boxes=[tag])}, recipe, library)
    assert result.verdict == cmp.FAIL


def test_read_values_are_collected_for_traceability(recipe, library):
    result = cmp.compare_unit(good_unit(), recipe, library, "U1")
    assert result.values["side_a.spec_plate_31agm.serial"] == "SN123456"
    assert result.values["side_b.trace_tag.serial"] == "SN123456"
    assert result.values["side_a.spec_plate_31agm.date_code"] == "2026-08-14"


# --- unit-level plumbing ---------------------------------------------------

def test_a_camera_that_did_not_capture_fails_the_unit(recipe, library):
    result = cmp.compare_unit({"side_a": good_side_a(), "side_b": None}, recipe, library, "U1")
    assert result.verdict == cmp.FAIL
    assert any("side_b" in f.message for f in result.failures())


def test_unconstrained_recipe_passes_anything(recipe, library):
    recipe.constrained = False
    views = {"side_a": frame("side_a"), "side_b": None}
    assert cmp.compare_unit(views, recipe, library, "U1").verdict == cmp.PASS


def test_requirement_for_a_label_missing_from_the_library_still_checks_presence(recipe):
    from label_detections.core.labels import LabelLibrary
    result = cmp.compare_unit(good_unit(), recipe, LabelLibrary([]), "U1")
    assert cmp.NOT_IN_LIBRARY in [f.code for f in result.all_findings()]
    # Presence was still verified, so no missing-label failure.
    assert cmp.MISSING not in [f.code for f in result.failures()]


def test_result_serialises_for_the_traceability_log(recipe, library):
    payload = cmp.compare_unit(good_unit(), recipe, library, "U1").to_dict()
    assert payload["unit_id"] == "U1"
    assert {v["view"] for v in payload["views"]} == {"side_a", "side_b"}
    assert payload["views"][0]["rows"][0]["label_id"] == "spec_plate_31agm"
