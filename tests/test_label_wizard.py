from __future__ import annotations

from label_detections.core.label_wizard import FLOW, build_label, review_answers
from label_detections.core.labels import validate_label_def


def minimal():
    """The whole of a new label: an id, a name and a family.

    No artwork file and no measurements. The tool is already collecting
    pictures of the label; one of them, flattened, becomes its artwork.
    """
    answers = FLOW.defaults()
    answers.update(label_id="spec plate 31", name="31-AGM spec plate",
                   family="spec_plate")
    return answers


def test_minimal_answers_produce_a_valid_label():
    answers = minimal()
    assert FLOW.validate(answers) == []
    label = build_label(answers)
    assert validate_label_def(label) == []


def test_label_id_is_made_filesystem_safe():
    assert build_label(minimal()).label_id == "spec_plate_31"


def test_no_physical_size_is_needed():
    """Regions are proportional, so nothing has to be measured or calibrated."""
    answers = minimal()
    assert answers["size_mm"] == [0.0, 0.0]
    assert FLOW.validate(answers) == []
    assert build_label(answers).size_mm == [0.0, 0.0]


def test_the_wizard_never_asks_for_what_cannot_exist_yet():
    """A label's dataset is keyed by its id, so no image of it exists yet.

    Asking for artwork -- or for regions drawn on artwork -- would be a circle:
    you cannot capture until the label exists, and the label could not be
    finished until you had captured.
    """
    keys = {q.key for q in FLOW.questions()}
    assert "reference_images" not in keys
    assert "draw_regions" not in keys
    assert "regions" not in [p.key for p in FLOW.pages]

    answers = minimal()
    assert answers["size_mm"] == [0.0, 0.0]
    assert FLOW.validate(answers) == []
    assert build_label(answers).reference_images == []


def test_a_bad_regex_is_caught_at_entry_not_at_runtime():
    answers = minimal()
    answers["codes"] = [{"role": "serial", "symbology": "qr", "policy": "must_match_pattern",
                         "pattern": "[unclosed", "region": [0.1, 0.1, 0.2, 0.2]}]
    assert any("regular expression" in e for e in FLOW.validate(answers))


def test_a_region_outside_the_label_is_rejected_with_an_explanation():
    answers = minimal()
    answers["codes"] = [{"role": "serial", "symbology": "qr", "policy": "must_decode",
                         "region": [0.8, 0.1, 0.4, 0.2]}]
    assert any("fractions of the label" in e for e in FLOW.validate(answers))


def test_anchor_question_appears_only_for_variable_data_labels():
    answers = minimal()
    page = next(p for p in FLOW.pages if p.key == "orientation")
    assert "anchor_region" not in [q.key for q in page.visible_questions(answers)]
    answers["variable_data"] = True
    assert "anchor_region" in [q.key for q in page.visible_questions(answers)]


def test_code_and_text_pages_survive_for_policies_known_up_front():
    """Where a code sits is drawn later; what it must satisfy can be known now."""
    pages = [p.key for p in FLOW.pages]
    assert "codes" in pages and "text" in pages


def test_variable_data_without_an_anchor_warns_about_matching_moving_text():
    answers = minimal()
    answers["variable_data"] = True
    assert any("every battery" in n for n in review_answers(answers))


def test_decode_feasibility_is_quoted_when_the_print_spec_is_given():
    """The camera-resolution conversation happens now, not after a failed trial."""
    answers = minimal()
    answers["codes"] = [{"role": "serial", "symbology": "datamatrix", "policy": "must_decode",
                         "region": [0.1, 0.1, 0.2, 0.2],
                         "code_width_mm": 12, "x_dim_mm": 0.254}]
    assert any("px" in n and "decode" in n for n in review_answers(answers))


def test_no_feasibility_noise_without_the_print_spec():
    """It is an optional hint. A label with no print numbers is still finished."""
    answers = minimal()
    answers["codes"] = [{"role": "serial", "symbology": "datamatrix",
                         "policy": "must_decode", "region": [0.1, 0.1, 0.2, 0.2]}]
    assert not any("px" in n for n in review_answers(answers))


def test_glossy_labels_prompt_a_lighting_check():
    answers = minimal()
    answers["surface"] = "foil"
    assert any("cross-polarised" in n for n in review_answers(answers))


def test_codes_and_text_fields_survive_the_build():
    answers = minimal()
    answers["codes"] = [{"role": "serial", "symbology": "datamatrix", "policy": "must_decode",
                         "region": [0.1, 0.1, 0.3, 0.3]}]
    answers["text_fields"] = [{"name": "date_code", "region": [0.1, 0.75, 0.6, 0.15],
                               "policy": "must_be_present"}]
    label = build_label(answers)
    assert label.code_by_role("serial").symbology == "datamatrix"
    assert label.code_by_role("serial").region == [0.1, 0.1, 0.3, 0.3]
    assert label.text_fields[0].region == [0.1, 0.75, 0.6, 0.15]


def test_training_settings_belong_to_the_label_not_a_recipe():
    answers = minimal()
    answers.update(train_target=400, confusable_with=["spec_plate_27agm"])
    label = build_label(answers)
    assert label.train_target == 400
    assert label.confusable_with == ["spec_plate_27agm"]


def test_no_lookalikes_named_prompts_the_hard_negative_question():
    assert any("hard negative" in n for n in review_answers(minimal()))
