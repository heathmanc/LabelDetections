"""The export diagnostics, read back from a dataset's own manifest."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import export_report as er

HEADER = "split,label_id,image,boxes,group"

DATA_YAML = """path: /x
train: images/train
val: images/val
nc: 3
names:
  0: battery_side
  1: spec_plate
  2: trace_tag
"""


def _write(tmp, manifest_lines=None, data_yaml=None, split_report=None):
    if manifest_lines is not None:
        (tmp / "manifest.csv").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    if data_yaml is not None:
        (tmp / "data.yaml").write_text(data_yaml, encoding="utf-8")
    if split_report is not None:
        (tmp / "split_report.txt").write_text(split_report, encoding="utf-8")
    return tmp


def test_class_names_parsed_in_order(tmp_path):
    _write(tmp_path, data_yaml=DATA_YAML)
    assert er.class_names(tmp_path) == ["battery_side", "spec_plate", "trace_tag"]


def test_class_names_missing_yaml(tmp_path):
    assert er.class_names(tmp_path) == []


def test_missing_manifest(tmp_path):
    assert "No manifest.csv" in er.count_summary(tmp_path)


def test_empty_manifest(tmp_path):
    _write(tmp_path, manifest_lines=[HEADER])
    assert "No labeled images" in er.count_summary(tmp_path)


def test_totals_come_from_the_manifest_the_export_actually_wrote(tmp_path):
    """The columns changed with the per-label export; reading the old ones
    reported zeros for everything while looking perfectly healthy."""
    rows = [
        HEADER,
        "train,spec_plate_31agm,spec_plate_31agm__a.jpg,2,s1",
        "train,spec_plate_31agm,spec_plate_31agm__b.jpg,1,s1",
        "val,trace_tag,trace_tag__c.jpg,3,s2",
    ]
    _write(tmp_path, manifest_lines=rows, data_yaml=DATA_YAML)
    out = er.count_summary(tmp_path)
    assert "Images written: 3  (train 2, val 1)" in out
    assert "Boxes written: 6" in out
    assert "spec_plate_31agm: 2" in out
    assert "trace_tag: 1" in out
    assert "Capture groups: 2" in out
    assert "Detector families (3): battery_side, spec_plate, trace_tag" in out


def test_backgrounds_are_counted_not_warned_about(tmp_path):
    """An empty label file is exactly how YOLO consumes a negative."""
    rows = [HEADER,
            "train,sp,sp__a.jpg,1,s1",
            "train,sp,sp__bg.jpg,0,s1"]
    _write(tmp_path, manifest_lines=rows)
    out = er.count_summary(tmp_path)
    assert "Images with no boxes (backgrounds): 1" in out


def test_split_warnings_are_surfaced_in_the_summary(tmp_path):
    """A validation set missing a label is worth seeing before training, not after."""
    rows = [HEADER, "train,sp,sp__a.jpg,1,s1"]
    _write(tmp_path, manifest_lines=rows,
           split_report="train: 1 images / 1 groups\n"
                        "WARNING: Label 'rare' has no validation images, so its "
                        "metrics mean nothing.\n")
    out = er.count_summary(tmp_path)
    assert "WARNING: Label 'rare' has no validation images" in out


def test_a_malformed_manifest_reports_rather_than_raising(tmp_path):
    (tmp_path / "manifest.csv").write_bytes(b"\xff\xfe not csv")
    assert "Could not read manifest.csv" in er.count_summary(tmp_path)


def test_rows_with_no_label_id_are_still_counted(tmp_path):
    rows = [HEADER, "train,,orphan.jpg,1,"]
    _write(tmp_path, manifest_lines=rows)
    out = er.count_summary(tmp_path)
    assert "Images written: 1" in out
    assert "(unknown): 1" in out
