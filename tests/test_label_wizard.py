from __future__ import annotations

from label_detections.core.label_wizard import FLOW, build_label, review_answers
from label_detections.core.labels import validate_label_def


def minimal():
    answers = FLOW.defaults()
    answers.update(label_id="spec plate 31", name="31-AGM spec plate",
                   family="spec_plate", reference_images=["ref.png"], size_mm=[90, 60])
    return answers


def test_minimal_answers_produce_a_valid_label():
    answers = minimal()
    assert FLOW.validate(answers) == []
    label = build_label(answers)
    assert validate_label_def(label) == []


def test_label_id_is_made_filesystem_safe():
    assert build_label(minimal()).label_id == "spec_plate_31"


def test_size_is_required_because_it_is_the_scale_check():
    answers = minimal()
    answers["size_mm"] = [0, 0]
    assert any("Physical size" in e for e in FLOW.validate(answers))


def test_a_bad_regex_is_caught_at_entry_not_at_runtime():
    answers = minimal()
    answers["codes"] = [{"role": "serial", "symbology": "qr", "policy": "must_match_pattern",
                         "pattern": "[unclosed", "region_mm": [1, 1, 10, 10], "x_dim_mm": 0.5}]
    assert any("regular expression" in e for e in FLOW.validate(answers))


def test_a_code_region_outside_the_label_is_rejected():
    answers = minimal()
    answers["codes"] = [{"role": "serial", "symbology": "qr", "policy": "must_decode",
                         "region_mm": [-1, 1, 10, 10]}]
    assert any("negative" in e for e in FLOW.validate(answers))


def test_anchor_question_appears_only_for_variable_data_labels():
    answers = minimal()
    page = next(p for p in FLOW.pages if p.key == "orientation")
    assert "anchor_region_mm" not in [q.key for q in page.visible_questions(answers)]
    answers["variable_data"] = True
    assert "anchor_region_mm" in [q.key for q in page.visible_questions(answers)]


def test_variable_data_without_an_anchor_warns_about_matching_moving_text():
    answers = minimal()
    answers["variable_data"] = True
    assert any("every battery" in n for n in review_answers(answers))


def test_decode_feasibility_is_quoted_in_pixels_at_entry_time():
    """The camera-resolution conversation happens now, not after a failed trial."""
    answers = minimal()
    answers["codes"] = [{"role": "serial", "symbology": "datamatrix", "policy": "must_decode",
                         "region_mm": [10, 10, 12, 12], "x_dim_mm": 0.254}]
    assert any("px" in n and "decode" in n for n in review_answers(answers))


def test_glossy_labels_prompt_a_lighting_check():
    answers = minimal()
    answers["surface"] = "foil"
    assert any("cross-polarised" in n for n in review_answers(answers))


def test_codes_and_text_fields_survive_the_build():
    answers = minimal()
    answers["codes"] = [{"role": "serial", "symbology": "datamatrix", "policy": "must_decode",
                         "region_mm": [10, 10, 30, 30], "x_dim_mm": 0.254}]
    answers["text_fields"] = [{"name": "date_code", "region_mm": [10, 45, 60, 10],
                               "policy": "must_be_present"}]
    label = build_label(answers)
    assert label.code_by_role("serial").symbology == "datamatrix"
    assert label.text_fields[0].name == "date_code"


def test_training_settings_belong_to_the_label_not_a_recipe():
    answers = minimal()
    answers.update(train_target=400, confusable_with=["spec_plate_27agm"])
    label = build_label(answers)
    assert label.train_target == 400
    assert label.confusable_with == ["spec_plate_27agm"]


def test_no_lookalikes_named_prompts_the_hard_negative_question():
    assert any("hard negative" in n for n in review_answers(minimal()))
