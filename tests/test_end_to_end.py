"""One pass through the whole thing: wizard answers -> library -> recipe -> verdict.

Each module has its own tests; this one exists to catch the seams between them,
which is where a schema rename or a renamed key actually breaks a build.
"""
from __future__ import annotations

from label_detections.core import annotations as ann
from label_detections.core import compare as cmp
from label_detections.core import label_wizard, persistence, recipe_wizard, review
from conftest import code_region, frame, label_box


def add_plate(tmp_path):
    answers = label_wizard.FLOW.defaults()
    answers.update(
        label_id="spec_plate_31agm", name="31-AGM spec plate", family="spec_plate",
        reference_images=["ref.png"], size_mm=[90, 60], train_target=200,
        confusable_with=["spec_plate_27agm"],
        codes=[{"role": "serial", "symbology": "datamatrix",
                "policy": "must_match_pattern", "pattern": r"^SN\d{6}$",
                "region_mm": [10, 10, 30, 30], "x_dim_mm": 0.254}],
    )
    assert label_wizard.FLOW.validate(answers) == []
    persistence.add_label(label_wizard.build_label(answers), tmp_path)
    return answers


def add_lookalike(tmp_path):
    answers = label_wizard.FLOW.defaults()
    answers.update(label_id="spec_plate_27agm", name="27-AGM spec plate",
                   family="spec_plate", reference_images=["other.png"], size_mm=[90, 60])
    persistence.add_label(label_wizard.build_label(answers), tmp_path)


def make_recipe(tmp_path):
    answers = recipe_wizard.FLOW.defaults()
    answers.update(
        group="AGM", model="31-AGM-950", revision="C",
        views=[{"view": "side_a", "camera": "cam1", "frame_size": [1000, 500],
                "unexpected_severity": "warn"}],
        bill=[{"view": "side_a", "label_id": "spec_plate_31agm",
               "roi": [0.05, 0.1, 0.3, 0.4], "count": 1, "severity": "fail",
               "roi_tol": 0.02}],
        forbidden=[{"view": "side_a", "label_id": "spec_plate_27agm"}],
    )
    assert recipe_wizard.FLOW.validate(answers) == []
    recipe = recipe_wizard.build_recipe(answers)
    persistence.save_recipe(recipe, tmp_path)
    return recipe


def good_frame(library):
    plate = label_box("spec_plate_31agm", "spec_plate", 125, 100, 150, 100)
    # The barcode box is placed from the artwork, not drawn by hand.
    ann.apply_reference_regions(plate, library.get("spec_plate_31agm"))
    region = ann.code_region(plate, "serial")
    region.update(decoded="SN123456", decode_ok=True)
    return frame("side_a", boxes=[plate], label_id="spec_plate_31agm")


def test_wizards_library_recipe_and_verdict_all_line_up(tmp_path):
    add_plate(tmp_path)
    add_lookalike(tmp_path)
    recipe = make_recipe(tmp_path)

    library = persistence.load_library(tmp_path)
    assert library.get("spec_plate_31agm").train_target == 200

    reloaded = persistence.list_recipes(tmp_path)[0]
    assert reloaded.to_dict() == recipe.to_dict()

    result = cmp.compare_unit({"side_a": good_frame(library)}, reloaded, library, "U1")
    assert result.verdict == cmp.PASS, result.summary_text()
    assert result.values["side_a.spec_plate_31agm.serial"] == "SN123456"


def test_the_lookalike_fails_even_though_the_required_plate_is_present(tmp_path):
    add_plate(tmp_path)
    add_lookalike(tmp_path)
    recipe = make_recipe(tmp_path)
    library = persistence.load_library(tmp_path)

    data = good_frame(library)
    data["boxes"].append(label_box("spec_plate_27agm", "spec_plate", 700, 300, 150, 100))
    result = cmp.compare_unit({"side_a": data}, recipe, library, "U1")
    assert result.verdict == cmp.FAIL
    assert cmp.FORBIDDEN in [f.code for f in result.failures()]


def test_a_recipe_naming_an_untrained_label_says_so_before_it_ships(tmp_path):
    add_plate(tmp_path)
    library = persistence.load_library(tmp_path)      # look-alike never added
    answers = recipe_wizard.FLOW.defaults()
    answers.update(
        group="AGM", model="31-AGM-950",
        views=[{"view": "side_a", "frame_size": [1000, 500], "unexpected_severity": "warn"}],
        bill=[{"view": "side_a", "label_id": "spec_plate_31agm",
               "roi": [0.05, 0.1, 0.3, 0.4], "count": 1, "severity": "fail"}],
        forbidden=[{"view": "side_a", "label_id": "spec_plate_27agm"}],
    )
    notes = recipe_wizard.review_answers(answers, library)
    assert any("spec_plate_27agm" in n and "trained" in n for n in notes)


def test_a_labeled_image_walks_the_review_gate_to_export_ready(tmp_path):
    add_plate(tmp_path)
    library = persistence.load_library(tmp_path)
    data = good_frame(library)

    assert review.annotation_status(data, "spec_plate_31agm") == "needs_review"
    assert review.validate_boxes(data, "spec_plate_31agm") == []

    review.stamp(data, review.make_review_record())
    persistence.save_annotation("spec_plate_31agm", "frame_001.jpg", data, tmp_path)

    saved = persistence.load_annotation("spec_plate_31agm", "frame_001.jpg", tmp_path)
    status = review.annotation_status(saved, "spec_plate_31agm")
    assert status == "ready" and review.export_ready(status)


def test_editing_a_reviewed_image_drops_its_approval(tmp_path):
    add_plate(tmp_path)
    library = persistence.load_library(tmp_path)
    data = good_frame(library)
    review.stamp(data, review.make_review_record())

    data["boxes"] = []                       # the operator deleted the label
    review.clear_review(data)
    assert review.annotation_status(data, "spec_plate_31agm") == "empty"
