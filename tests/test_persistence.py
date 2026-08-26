from __future__ import annotations

import json

from label_detections.core import persistence as io
from label_detections.core import storage
from label_detections.core.labels import CodeSpec, LabelDef, LabelLibrary


def test_library_round_trips_through_disk(tmp_path):
    label = LabelDef(label_id="sp", size_mm=[90, 60], reference_images=["a.png"])
    label.codes = [CodeSpec(role="serial", region_mm=[1, 2, 3, 4])]
    io.save_library(LabelLibrary([label]), tmp_path)
    back = io.load_library(tmp_path)
    assert back.get("sp").codes[0].region_mm == [1, 2, 3, 4]


def test_missing_library_loads_empty_so_the_app_still_launches(tmp_path):
    assert len(io.load_library(tmp_path)) == 0


def test_corrupt_library_loads_empty_rather_than_raising(tmp_path):
    (tmp_path / "labels.json").write_text("{ not json", encoding="utf-8")
    assert len(io.load_library(tmp_path)) == 0


def test_add_label_persists_and_refuses_a_duplicate(tmp_path):
    io.add_label(LabelDef(label_id="sp"), tmp_path)
    assert "sp" in io.load_library(tmp_path)
    try:
        io.add_label(LabelDef(label_id="sp"), tmp_path)
    except ValueError:
        pass
    else:
        raise AssertionError("a duplicate label id should be refused")
    io.add_label(LabelDef(label_id="sp", name="new"), tmp_path, replace=True)
    assert io.load_library(tmp_path).get("sp").name == "new"


def test_writes_are_atomic_and_leave_no_temp_file(tmp_path):
    io.save_library(LabelLibrary([LabelDef(label_id="sp")]), tmp_path)
    assert not list(tmp_path.glob("*.tmp"))


def test_annotation_sidecar_sits_beside_by_stem(tmp_path):
    io.save_annotation("sp", "frame_001.jpg", {"boxes": []}, tmp_path)
    assert (tmp_path / "sp" / "frame_001.json").is_file()
    assert io.load_annotation("sp", "frame_001.jpg", tmp_path) == {"boxes": []}


def test_dataset_statuses_report_images_that_have_no_sidecar(tmp_path):
    captures = tmp_path / "captures"
    labels = tmp_path / "labels"
    (captures / "sp").mkdir(parents=True)
    (captures / "sp" / "a.jpg").write_bytes(b"")
    (captures / "sp" / "b.jpg").write_bytes(b"")
    (captures / "sp" / "notes.txt").write_bytes(b"")
    io.save_annotation("sp", "a.jpg", {"boxes": [{"label_id": "sp"}]}, labels)
    statuses = io.dataset_statuses("sp", captures, labels)
    assert statuses == {"a.jpg": "needs_review", "b.jpg": "unlabeled"}


def test_list_images_ignores_non_images_and_hidden_files(tmp_path):
    folder = tmp_path / "sp"
    folder.mkdir()
    for name in ("a.jpg", "b.PNG", "c.txt", ".hidden.jpg"):
        (folder / name).write_bytes(b"")
    assert [p.name for p in storage.list_images("sp", tmp_path)] == ["a.jpg", "b.PNG"]
