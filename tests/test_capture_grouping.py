"""Frames of one battery stay on one side of the split.

The split never separates a capture group, and that guarantee is what makes a
validation number mean anything. Two frames taken a second apart in the same
pose are very nearly the same image; put one in train and the other in val and
the model is being tested on something it memorised.

Grouping came from ``session`` in the sidecar, and only frames kept from Live
Detect ever carried one. The Capture button writes an image and no sidecar at
all, so ``group_key()`` fell through to the filename and every frame became its
own group -- which is not a group, it is the absence of one.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-groups-"))

import pytest

from label_detections.core import capture_session as cs
from label_detections.core import dataset as dataset_logic


# --- the record --------------------------------------------------------------

def test_a_session_token_reads_as_when_it_started():
    token = cs.new_id(now=1_700_000_000)
    assert token.startswith("cap_")
    assert "20231114" in token or "20231115" in token   # local time


def test_two_groups_started_in_one_second_are_still_two_groups():
    """A second-resolution token merged two batteries swapped quickly into one
    group -- the safe direction, but it silently defeats the button whose whole
    job is to say they are different."""
    assert cs.new_id(now=1_700_000_000) != cs.new_id(now=1_700_000_000)


def test_captures_are_remembered_against_their_session(tmp_path):
    cs.record("lbl", "/somewhere/a.jpg", "cap_1", root=tmp_path)
    cs.record("lbl", "/somewhere/b.jpg", "cap_1", root=tmp_path)
    cs.record("lbl", "/elsewhere/c.jpg", "cap_2", root=tmp_path)

    known = cs.load("lbl", root=tmp_path)
    assert known == {"a.jpg": "cap_1", "b.jpg": "cap_1", "c.jpg": "cap_2"}
    assert cs.session_for(known, "/any/path/b.jpg") == "cap_1"
    assert cs.session_for(known, "never_seen.jpg") == ""


def test_the_record_lives_beside_the_sidecars_not_inside_them(tmp_path):
    """An image is captured long before it is annotated, often in another
    sitting -- the sidecar does not exist at the moment the session is known."""
    cs.record("lbl", "a.jpg", "cap_1", root=tmp_path)
    assert (tmp_path / cs.FILENAME).is_file()


def test_a_bookkeeping_failure_never_stops_a_capture(tmp_path):
    """The alternative is refusing to photograph over a JSON file. A missing
    entry costs the grouping for one image, which is what every image had."""
    blocked = tmp_path / "nope"
    blocked.write_text("not a directory")
    cs.record("lbl", "a.jpg", "cap_1", root=blocked / "deeper")   # must not raise
    assert cs.load("lbl", root=blocked / "deeper") == {}


def test_junk_on_disk_reads_as_no_grouping(tmp_path):
    (tmp_path / cs.FILENAME).write_text("[1, 2, 3]")
    assert cs.load("lbl", root=tmp_path) == {}


# --- what it does to a split -------------------------------------------------

def _burst(session: str, n: int, start: int = 0):
    return [dataset_logic.Entry(label_id="pc680", image=f"f{start + i}.jpg",
                                annotation={"boxes": [{}]}, session=session)
            for i in range(n)]


def test_without_a_session_every_frame_is_its_own_group():
    """The bug, stated. Twenty frames of one battery, twenty groups."""
    loose = [dataset_logic.Entry(label_id="pc680", image=f"f{i}.jpg")
             for i in range(20)]
    assert len({e.group_key() for e in loose}) == 20


def test_a_burst_is_one_group_and_cannot_be_split():
    train, val, report = dataset_logic.split_entries(
        _burst("cap_1", 10) + _burst("cap_2", 10, start=10), seed=1)
    for side in (train, val):
        assert len({e.session for e in side}) <= 1, "a burst straddled the split"
    assert report.train_groups + report.val_groups == 2


def test_one_group_for_everything_is_called_out_rather_than_silently_split():
    """A camera left open across ten batteries is one group, and one group
    cannot be split. That is worth saying: a loud degenerate split beats a
    quiet leaky one."""
    _t, _v, report = dataset_logic.split_entries(_burst("cap_1", 20), seed=1)
    assert any("one capture group" in w.lower() for w in report.warnings)


def test_merging_two_batteries_is_the_safe_direction():
    """Nothing in software knows when the battery changed, so grouping errs
    large. Merging two batteries costs a little validation data; splitting one
    battery across the two sides costs the meaning of the number."""
    entries = _burst("cap_1", 8) + _burst("cap_2", 8, start=8)
    train, val, _r = dataset_logic.split_entries(entries, seed=3)
    images = {e.image for e in train} | {e.image for e in val}
    assert len(images) == 16, "no image was lost or duplicated"
    assert not ({e.image for e in train} & {e.image for e in val})


# --- what the report says ----------------------------------------------------

def test_the_report_says_when_nothing_is_grouped():
    note = cs.group_summary({}, images=20)
    assert "No capture grouping" in note
    assert "both sides of the split" in note


def test_the_report_says_when_everything_is_one_group():
    note = cs.group_summary({f"a{i}.jpg": "cap_1" for i in range(20)}, images=20)
    assert "1 capture group" in note
    assert "train and val will be the same images" in note


def test_images_from_before_grouping_existed_are_counted_separately():
    """They are not a fault, they are history -- and they are each their own
    group, which the reader has to know before trusting the group count."""
    known = {f"a{i}.jpg": f"cap_{i // 5}" for i in range(20)}
    note = cs.group_summary(known, images=25)
    assert "4 capture group" in note
    assert "5 ungrouped" in note


def test_a_well_grouped_dataset_gets_a_plain_line():
    known = {f"a{i}.jpg": f"cap_{i // 5}" for i in range(20)}
    note = cs.group_summary(known, images=20)
    assert "4 capture group(s) across 20 image(s)." == note


# --- the window --------------------------------------------------------------

pytest.importorskip("PySide6.QtWidgets")
from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


def test_a_new_group_can_be_started_without_stopping_the_preview():
    """Swapping the battery does not stop the camera, and it is the only
    moment the information exists."""
    win = _window()
    first = win.start_capture_group(announce=False)
    second = win.start_capture_group(announce=False)
    assert first and second and first != second

    assert "New Group" in {b.text() for b in win.findChildren(QPushButton)}


def test_opening_the_camera_starts_one():
    import inspect

    from label_detections.ui.main_window import MainWindow

    assert "start_capture_group" in inspect.getsource(MainWindow.open_camera)


def test_every_capture_path_records_its_group():
    """Three write into a dataset -- the capture button, a live frame kept by
    hand, and the reference photograph -- and a frame missed by one of them is
    a frame the split may separate."""
    import inspect

    from label_detections.ui.main_window import MainWindow

    for name in ("capture_frame", "_keep_frame", "_keep_reference_photo"):
        source = inspect.getsource(getattr(MainWindow, name))
        assert "_record_capture_group" in source, f"{name} does not record one"


def test_both_variants_of_one_capture_stay_together():
    """Raw and adjusted are the same battery in the same pose, so separating
    them would be the worst split available."""
    import inspect

    from label_detections.ui.main_window import MainWindow

    source = inspect.getsource(MainWindow.capture_frame)
    assert "_record_capture_group(raw_path, adj_path)" in source


def test_nothing_is_recorded_without_a_label_or_a_session(tmp_path):
    win = _window()
    win._capture_session = ""
    win._record_capture_group(tmp_path / "a.jpg")     # must not raise
    win._capture_session = "cap_1"
    saved, win.label_id = win.label_id, ""
    try:
        win._record_capture_group(tmp_path / "a.jpg")
    finally:
        win.label_id = saved
