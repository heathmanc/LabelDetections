from __future__ import annotations

from label_detections.core import dataset as ds
from label_detections.core import review


def entry(label_id="sp", image="", session="", boxes=None, **kw):
    return ds.Entry(label_id=label_id, image=image or f"{label_id}_{session}_{id(boxes)}.jpg",
                    session=session, annotation={"boxes": boxes or []}, **kw)


def burst(session, n, label_id="sp"):
    """Several near-identical frames from one capture session."""
    return [entry(label_id, f"{session}_{i}.jpg", session,
                  boxes=[{"label_id": label_id, "label": "spec_plate"}]) for i in range(n)]


def test_a_capture_session_is_never_split_across_train_and_val():
    """The bug this module exists to prevent: near-duplicates on both sides."""
    entries = [e for s in range(10) for e in burst(f"s{s}", 4)]
    train, val, _ = ds.split_entries(entries, 0.8, seed=1)
    train_sessions = {e.session for e in train}
    val_sessions = {e.session for e in val}
    assert not (train_sessions & val_sessions)


def test_split_is_reproducible_for_a_given_seed():
    entries = [e for s in range(10) for e in burst(f"s{s}", 3)]
    a, _, _ = ds.split_entries(entries, 0.8, seed=7)
    b, _, _ = ds.split_entries(entries, 0.8, seed=7)
    c, _, _ = ds.split_entries(entries, 0.8, seed=8)
    assert [e.image for e in a] == [e.image for e in b]
    assert [e.image for e in a] != [e.image for e in c]


def test_entries_without_provenance_each_become_their_own_group():
    """Degrades to per-image splitting rather than doing something surprising."""
    entries = [entry("sp", f"{i}.jpg") for i in range(10)]
    train, val, report = ds.split_entries(entries, 0.8, seed=1)
    assert report.train_groups + report.val_groups == 10
    assert len(train) + len(val) == 10


def test_every_label_is_forced_into_validation():
    """A label with no val images has no measured accuracy at all."""
    entries = burst("s1", 5, "common") + burst("s2", 5, "common") + burst("s3", 2, "rare")
    _, val, report = ds.split_entries(entries, 0.95, seed=3)
    assert "rare" in {e.label_id for e in val}
    assert any("rare" in w for w in report.warnings)


def test_a_single_group_warns_instead_of_silently_validating_on_train():
    entries = burst("only", 6)
    train, val, report = ds.split_entries(entries, 0.8, seed=1)
    assert train and val
    assert any("same images" in w for w in report.warnings)


def test_empty_input_is_not_an_error():
    train, val, report = ds.split_entries([], 0.8)
    assert (train, val) == ([], [])
    assert report.text()


def test_report_text_names_both_splits():
    entries = [e for s in range(5) for e in burst(f"s{s}", 3)]
    _, _, report = ds.split_entries(entries, 0.8, seed=2)
    text = report.text()
    assert "train:" in text and "val:" in text and "sp:" in text


def test_instance_counts_count_boxes_not_images():
    entries = [entry("sp", "a.jpg", boxes=[{"label_id": "sp"}, {"label_id": "sp"}])]
    assert ds.instance_counts(entries) == {"sp": 2}


def test_code_coverage_reports_the_decode_rate():
    """The number that separates a model problem from an optics problem."""
    entries = [entry("sp", "a.jpg", boxes=[{
        "label_id": "sp",
        "regions": [
            {"role": "code", "decode_ok": True},
            {"role": "code", "decode_ok": False},
            {"role": "text"},
        ],
    }])]
    assert ds.code_coverage(entries) == {"sp": {"regions": 2, "decoded": 1}}


def test_thin_coverage_names_the_labels_that_cannot_train_yet():
    entries = ([entry("common", f"{i}.jpg", boxes=[{"label_id": "common"}]) for i in range(40)]
               + [entry("rare", "r.jpg", boxes=[{"label_id": "rare"}])])
    thin = ds.thin_coverage(entries, minimum=30)
    assert len(thin) == 1 and thin[0].startswith("rare:")


def test_defect_mix_shows_whether_the_defect_library_is_real():
    stamped = review.stamp({"boxes": [{"label_id": "sp"}]},
                           review.make_review_record(force=True, defect_reason="torn_or_wrinkled"))
    entries = [ds.Entry(label_id="sp", image="a.jpg", annotation=stamped),
               entry("sp", "b.jpg", boxes=[{"label_id": "sp"}])]
    assert ds.defect_mix(entries) == {"torn_or_wrinkled": 1}


def test_entry_from_annotation_picks_up_provenance():
    e = ds.entry_from_annotation("sp", "a.jpg",
                                 {"session": "2026-08-14", "source": "line1", "boxes": []})
    assert e.group_key() == "2026-08-14"
    assert ds.Entry(label_id="x", image="i.jpg", source="line2").group_key() == "line2"
    assert ds.Entry(label_id="x", image="i.jpg").group_key() == "i.jpg"
