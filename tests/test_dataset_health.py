"""The readiness dashboard's tallies."""
from __future__ import annotations

from label_detections.core import dataset_health as dh
from label_detections.core import review


def reviewed(boxes, **kw):
    data = {"boxes": boxes}
    return review.stamp(data, review.make_review_record(**kw))


def box(label_id="sp"):
    return {"label_id": label_id, "label": "spec_plate",
            "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}


def test_new_tally_has_every_status():
    tally = dh.new_tally()
    assert set(dh.ALL_STATUSES) <= set(tally)
    assert tally["images"] == 0 and tally["labeled"] == 0


def test_each_status_lands_in_its_own_bucket():
    tally = dh.new_tally()
    assert dh.add_image(tally, None, "sp") == "unlabeled"
    assert dh.add_image(tally, {"boxes": []}, "sp") == "empty"
    assert dh.add_image(tally, {"boxes": [], "background": True}, "sp") == "background"
    assert dh.add_image(tally, {"boxes": [box()]}, "sp") == "needs_review"
    assert dh.add_image(tally, reviewed([box()]), "sp") == "ready"
    assert dh.add_image(tally, reviewed([box("other")]), "sp") == "problem"
    assert dh.add_image(tally, reviewed([box()], force=True,
                                        defect_reason="torn_or_wrinkled"), "sp") == "forced"
    assert tally["images"] == 7
    # Only the statuses that carry boxes count as labeled work.
    assert tally["labeled"] == 4


def test_export_ready_matches_the_gate():
    """One definition of ready: the dashboard cannot disagree with the export."""
    tally = dh.new_tally()
    dh.add_image(tally, reviewed([box()]), "sp")
    dh.add_image(tally, reviewed([box()], force=True), "sp")
    dh.add_image(tally, {"boxes": [], "background": True}, "sp")
    dh.add_image(tally, reviewed([box("other")]), "sp")     # stale approval
    dh.add_image(tally, {"boxes": [box()]}, "sp")           # unreviewed
    assert dh.export_ready(tally) == 3


def test_merge_tally_accumulates():
    a, b = dh.new_tally(), dh.new_tally()
    dh.add_image(a, reviewed([box()]), "sp")
    dh.add_image(b, reviewed([box()]), "sp")
    dh.merge_tally(a, b)
    assert a["images"] == 2 and a["ready"] == 2


def test_readiness_is_a_clamped_fraction_of_the_target():
    tally = dh.new_tally()
    for _ in range(5):
        dh.add_image(tally, reviewed([box()]), "sp")
    assert dh.readiness(tally, 10) == 0.5
    assert dh.readiness(tally, 2) == 1.0
    assert dh.readiness(tally, 0) == 1.0


def test_blockers_put_stale_approvals_first():
    """They are the only category that is wrong rather than merely unfinished."""
    tally = dh.new_tally()
    dh.add_image(tally, {"boxes": [box()]}, "sp")
    dh.add_image(tally, reviewed([box("other")]), "sp")
    dh.add_image(tally, None, "sp")
    blockers = dh.blockers(tally)
    assert "no longer carry the label" in blockers[0]
    assert len(blockers) == 3


def test_a_clean_dataset_has_no_blockers():
    tally = dh.new_tally()
    dh.add_image(tally, reviewed([box()]), "sp")
    assert dh.blockers(tally) == []
