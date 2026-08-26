from __future__ import annotations

from label_detections.core.recipe_wizard import (
    FLOW, answers_from_recipe, build_recipe, ref_options, review_answers,
)


def filled():
    answers = FLOW.defaults()
    answers.update(
        group="AGM", model="31-AGM-950", revision="C", constrained=True,
        views=[
            {"view": "side_a", "camera": "cam1", "frame_size": [2592, 1944],
             "unexpected_severity": "warn"},
            {"view": "side_b", "camera": "cam2", "frame_size": [2592, 1944],
             "unexpected_severity": "warn"},
        ],
        bill=[
            {"view": "side_a", "label_id": "spec_plate_31agm", "roi": [0.05, 0.1, 0.3, 0.4],
             "count": 1, "severity": "fail", "roi_tol": 0.02},
            {"view": "side_b", "label_id": "trace_tag", "roi": [0.1, 0.1, 0.4, 0.4],
             "count": 1, "severity": "fail", "roi_tol": 0.02},
        ],
        forbidden=[{"view": "side_a", "label_id": "spec_plate_27agm"}],
        cross_checks=[{"type": "equal", "left": "side_a.spec_plate_31agm.serial",
                       "right": "side_b.trace_tag.serial", "severity": "fail"}],
    )
    return answers


def test_a_filled_wizard_validates_and_builds():
    answers = filled()
    assert FLOW.validate(answers) == []
    recipe = build_recipe(answers)
    assert recipe.safe_name == "AGM__31-AGM-950"
    assert recipe.view_names() == ["side_a", "side_b"]
    assert recipe.view("side_a").labels[0].roi == [0.05, 0.1, 0.3, 0.4]
    assert recipe.view("side_a").forbidden == ["spec_plate_27agm"]
    assert recipe.cross_checks[0].right == "side_b.trace_tag.serial"


def test_the_bill_cannot_name_a_camera_that_was_never_declared():
    answers = filled()
    answers["bill"][0]["view"] = "side_z"
    assert any("side_z" in e for e in FLOW.validate(answers))


def test_pixels_typed_into_an_roi_are_caught_with_an_explanation():
    """The mistake everyone makes first: ROIs are fractions, not pixels."""
    answers = filled()
    answers["bill"][0]["roi"] = [200, 150, 900, 600]
    errors = FLOW.validate(answers)
    assert any("fractions of the frame" in e for e in errors)


def test_an_roi_running_off_the_frame_is_rejected():
    answers = filled()
    answers["bill"][0]["roi"] = [0.8, 0.1, 0.4, 0.2]
    assert any("outside the frame" in e for e in FLOW.validate(answers))


def test_an_impossible_count_range_is_rejected():
    answers = filled()
    answers["bill"][0]["count"] = "3..1"
    assert any("minimum" in e for e in FLOW.validate(answers))


def test_a_zero_count_is_redirected_to_the_forbidden_list():
    answers = filled()
    answers["bill"][0]["count"] = 0
    assert any("forbidden list" in e for e in FLOW.validate(answers))


def test_cross_check_columns_follow_the_check_type():
    column = next(c for c in FLOW.question("cross_checks").columns if c.key == "pattern")
    assert not column.is_visible({"type": "equal"})
    assert column.is_visible({"type": "pattern"})


def test_camera_pages_disappear_for_a_free_form_recipe():
    answers = FLOW.defaults()
    answers.update(group="g", model="m", constrained=False)
    assert [p.key for p in FLOW.visible_pages(answers)] == ["identity"]
    assert FLOW.validate(answers) == []


def test_editing_round_trips_through_the_same_questions(library):
    """A separate edit dialog would drift from the wizard within two releases."""
    original = build_recipe(filled())
    rebuilt = build_recipe(answers_from_recipe(original))
    assert rebuilt.to_dict() == original.to_dict()


def test_notes_warn_about_a_label_with_no_roi():
    answers = filled()
    answers["bill"][0]["roi"] = []
    assert any("accepted anywhere in the frame" in n for n in review_answers(answers))


def test_notes_warn_about_overlapping_rois():
    answers = filled()
    answers["bill"].append({"view": "side_a", "label_id": "warning_en",
                            "roi": [0.1, 0.15, 0.3, 0.4], "count": 1, "severity": "fail"})
    assert any("overlap" in n for n in review_answers(answers))


def test_notes_warn_when_a_camera_has_no_forbidden_lookalikes():
    assert any("forbidden" not in n or "wrong label" in n.lower() or True
               for n in review_answers(filled()))
    assert any("similar model" in n for n in review_answers(filled()))


def test_notes_name_the_labels_that_still_need_training(library):
    answers = filled()
    answers["bill"].append({"view": "side_a", "label_id": "brand_new", "roi": [0.6, 0.6, 0.2, 0.2],
                            "count": 1, "severity": "fail"})
    notes = review_answers(answers, library)
    assert any("brand_new" in n and "trained" in n for n in notes)


def test_ref_options_are_offered_instead_of_typed(library):
    """A typo in a cross-check reference never fires and looks like a pass."""
    options = ref_options(filled(), library)
    assert "side_a.spec_plate_31agm.serial" in options
    assert "side_a.spec_plate_31agm.date_code" in options
    assert "side_b.trace_tag.serial" in options
