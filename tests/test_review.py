from __future__ import annotations

from label_detections.core import annotations as ann
from label_detections.core import review
from conftest import frame, label_box, rect


def reviewed(data, **kw):
    return review.stamp(data, review.make_review_record(**kw))


def test_only_this_tools_marker_counts_as_reviewed():
    """Imported JSON is full of generic review fields that mean something else."""
    assert not review.annotation_reviewed({"reviewed": True})
    assert not review.annotation_reviewed({"reviewed": True, "review_status": "ok"})
    assert not review.annotation_reviewed({"review": {"reviewed": True, "tool": "Other"}})
    assert review.annotation_reviewed(reviewed({"boxes": []}))


def test_legacy_flat_marker_is_still_accepted():
    data = {"reviewed": True, "reviewed_by": "LabelVision Studio v0.1.0"}
    assert review.annotation_reviewed(data)


def test_force_review_records_why_so_the_defect_library_is_queryable():
    data = reviewed({"boxes": []}, force=True, defect_reason="smeared_code",
                    verdict="fail", findings=["code did not decode"])
    assert review.annotation_force_reviewed(data)
    assert data["review"]["defect_reason"] == "smeared_code"
    assert data["review"]["findings"] == ["code did not decode"]


def test_force_review_defaults_to_other_rather_than_going_unrecorded():
    data = reviewed({"boxes": []}, force=True)
    assert data["review"]["defect_reason"] == "other"


def test_clearing_review_after_an_edit_leaves_nothing_behind():
    """Editing is not approving. A stale approval is the dangerous bug."""
    data = reviewed({"boxes": []})
    review.clear_review(data)
    assert not review.annotation_reviewed(data)
    assert "review_status" not in data


def test_background_needs_the_explicit_flag_not_just_an_empty_box_list():
    assert not review.is_background_annotation({"boxes": []})
    assert review.is_background_annotation({"boxes": [], "background": True})
    # A drawn box contradicts the flag, so the box wins.
    assert not review.is_background_annotation({"boxes": [{}], "background": True})


def test_status_walks_from_unlabeled_to_ready():
    assert review.annotation_status(None, "sp") == "unlabeled"
    data = frame(label_id="sp")
    assert review.annotation_status(data, "sp") == "empty"
    data["boxes"].append(label_box("sp", "spec_plate", 0, 0, 100, 60))
    assert review.annotation_status(data, "sp") == "needs_review"
    reviewed(data)
    assert review.annotation_status(data, "sp") == "ready"


def test_approved_image_missing_its_label_is_a_stale_approval():
    data = frame(label_id="sp", boxes=[label_box("other", "spec_plate", 0, 0, 100, 60)])
    reviewed(data)
    assert review.annotation_status(data, "sp") == "problem"


def test_forced_images_export_but_stale_approvals_do_not():
    assert review.export_ready("forced")
    assert review.export_ready("background")
    assert not review.export_ready("problem")
    assert not review.export_ready("needs_review")


def test_validate_flags_degenerate_and_out_of_bounds_boxes():
    data = frame(label_id="sp", boxes=[
        label_box("sp", "spec_plate", 0, 0, 1, 1),
        label_box("sp", "spec_plate", 990, 490, 100, 100),
    ])
    issues = review.validate_boxes(data, "sp")
    assert any("degenerate" in i for i in issues)
    assert any("outside the image" in i for i in issues)


def test_validate_flags_a_duplicate_of_the_same_label():
    box = label_box("sp", "spec_plate", 100, 100, 200, 120)
    twin = label_box("sp", "spec_plate", 105, 102, 200, 120)
    issues = review.validate_boxes(frame(label_id="sp", boxes=[box, twin]), "sp")
    assert any("duplicate" in i.lower() for i in issues)


def test_validate_flags_a_region_that_escaped_its_label():
    box = label_box("sp", "spec_plate", 100, 100, 200, 120)
    box["regions"] = [ann.make_region("code", rect(400, 400, 20, 20), code_role="serial")]
    issues = review.validate_boxes(frame(label_id="sp", boxes=[box]), "sp")
    assert any("outside its label" in i for i in issues)


def test_validate_flags_an_image_with_none_of_the_label_it_was_collected_for():
    data = frame(label_id="sp", boxes=[label_box("other", "spec_plate", 0, 0, 100, 60)])
    assert any("No 'sp' is labeled" in i for i in review.validate_boxes(data, "sp"))


def test_dataset_summary_says_what_is_left_to_do():
    text = review.dataset_summary("sp", ["ready"] * 5 + ["needs_review", "problem"], want=10)
    assert "5 export-ready of 7" in text
    assert "1 still to label or review" in text
    assert "stale approvals" in text
    assert "below the 10-image working target" in text
