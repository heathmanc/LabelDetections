"""The UI-to-exporter boundary, exercised the way the buttons exercise it.

This file exists because of a bug that shipped: main_window still called the
exporter with a class_mode argument the rewritten exporter had never accepted,
and nothing caught it because every test called the core functions directly
with the right arguments. A signature drift between the two halves is invisible
to a test that only ever looks at one of them.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-export-"))

import pytest

try:
    import cv2
    import numpy as np
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception as exc:  # pragma: no cover - depends on the environment
    HAVE_QT = False
    _WHY = exc

pytestmark = pytest.mark.skipif(not HAVE_QT, reason="PySide6/cv2 not available")

_win = None


def _window():
    global _win
    if _win is None:
        QApplication.instance() or QApplication([])
        from label_detections.ui.main_window import MainWindow
        _win = MainWindow()
    return _win


def _dataset(label_id, images=3):
    """A small, genuinely export-ready dataset for one label."""
    from label_detections.core import persistence, review, storage
    from label_detections.core.labels import LabelDef

    library = persistence.load_library()
    library.add(LabelDef(label_id=label_id, train_target=5), replace=True)
    persistence.save_library(library)

    folder = storage.dataset_folder(label_id)
    folder.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    for i in range(images):
        name = f"{label_id}_{i:03d}.jpg"
        cv2.imwrite(str(folder / name), frame)
        data = {
            "image": str(folder / name), "label_id": label_id,
            "width": 200, "height": 120,
            # A distinct capture group per image, so the split has something to
            # keep together rather than degrading to per-image shuffling.
            "session": f"{label_id}_s{i}",
            "boxes": [{
                "x": 10, "y": 10, "w": 100, "h": 60, "class_id": 1,
                "label": label_id, "label_id": label_id, "kind": "obb",
                "points": [[10, 10], [110, 10], [110, 70], [10, 70]],
            }],
        }
        review.stamp(data, review.make_review_record())
        persistence.save_annotation(label_id, name, data)
    return folder


def _silence_dialogs(monkeypatch):
    """Capture what the export would have told the operator."""
    import label_detections.ui.main_window as mw_mod

    shown = {}
    monkeypatch.setattr(mw_mod.QMessageBox, "information",
                        lambda parent, title, text, *a, **k: shown.update(
                            {"title": title, "text": text}))
    monkeypatch.setattr(mw_mod.QMessageBox, "warning",
                        lambda parent, title, text, *a, **k: shown.update(
                            {"warned": text}))
    return shown


def test_export_all_runs_the_way_the_button_runs_it(monkeypatch):
    """The exact call that raised: unexpected keyword argument 'class_mode'."""
    win = _window()
    _dataset("exp_all_a")
    _dataset("exp_all_b", )
    win.library = __import__(
        "label_detections.core.persistence", fromlist=["x"]).load_library()

    shown = _silence_dialogs(monkeypatch)
    win.export_all_yolo()

    assert "warned" not in shown, shown.get("warned")
    assert shown["title"] == "Export All complete"


def test_export_all_writes_a_trainable_dataset(monkeypatch):
    from label_detections.core.storage import EXPORT_DIR

    win = _window()
    _dataset("exp_written")
    _silence_dialogs(monkeypatch)
    win.export_all_yolo()

    out = EXPORT_DIR / "all_labels_obb"
    assert (out / "data.yaml").is_file()
    assert (out / "manifest.csv").is_file()
    assert (out / "split_report.txt").is_file()
    assert list((out / "images" / "train").glob("*.jpg"))
    assert list((out / "labels" / "train").glob("*.txt"))


def test_the_single_label_export_runs_too(monkeypatch):
    win = _window()
    _dataset("exp_single")
    win.set_active_label("exp_single")
    shown = _silence_dialogs(monkeypatch)
    win.export_yolo()
    assert "warned" not in shown, shown.get("warned")
    assert shown["title"] == "Export complete"


def test_the_detect_task_exports_as_well(monkeypatch):
    win = _window()
    _dataset("exp_detect")
    win.set_active_label("exp_detect")
    win.export_task_combo.setCurrentIndex(1)          # detect
    try:
        assert win._export_task() == "detect"
        shown = _silence_dialogs(monkeypatch)
        win.export_yolo()
        assert "warned" not in shown, shown.get("warned")
    finally:
        win.export_task_combo.setCurrentIndex(0)


def test_the_summary_reports_real_numbers_not_zeros(monkeypatch):
    """export_report read bung-era manifest columns and reported all zeros."""
    from label_detections.core.storage import EXPORT_DIR

    win = _window()
    _dataset("exp_summary", images=4)
    _silence_dialogs(monkeypatch)
    win.export_all_yolo()

    summary = win._export_count_summary(EXPORT_DIR / "all_labels_obb")
    assert "Images written:" in summary
    assert "Boxes written: 0" not in summary
    assert "exp_summary" in summary
    assert "Detector classes" in summary
    # The classes ARE the label ids, which is what makes the model's output
    # directly countable against a recipe written in ids.
    assert "exp_summary" in summary.split("Detector classes")[1]


def test_exporting_with_nothing_reviewed_explains_itself(monkeypatch):
    from label_detections.core import persistence
    from label_detections.core.labels import LabelDef

    win = _window()
    library = persistence.load_library()
    library.add(LabelDef(label_id="exp_empty", ), replace=True)
    persistence.save_library(library)
    win.library = persistence.load_library()
    win.set_active_label("exp_empty")

    shown = _silence_dialogs(monkeypatch)
    win.export_yolo()
    assert "warned" in shown
    assert "reviewed" in shown["warned"].lower()


def test_the_exported_classes_are_the_label_ids_the_recipe_uses(monkeypatch):
    """The whole point: what the model reports is countable against a recipe
    written in label ids, with nothing in between to resolve it."""
    from label_detections.core.storage import EXPORT_DIR

    win = _window()
    _dataset("exp_ids_a", images=3)
    _dataset("exp_ids_b", images=3)
    _silence_dialogs(monkeypatch)
    win.export_all_yolo()

    yaml = (EXPORT_DIR / "all_labels_obb" / "data.yaml").read_text()
    assert "exp_ids_a" in yaml and "exp_ids_b" in yaml


def test_the_battery_face_holds_class_zero(monkeypatch):
    """Class indices go into every label file. If a label could take index 0,
    adding one alphabetically early would re-point the whole dataset."""
    from label_detections.core import persistence, review, storage
    from label_detections.core.storage import EXPORT_DIR

    win = _window()
    folder = _dataset("aaa_sorts_first", images=3)
    # Give one image a battery_side box as well as the label's own.
    name = sorted(p.name for p in storage.list_images("aaa_sorts_first"))[0]
    data = persistence.load_annotation("aaa_sorts_first", name)
    data["boxes"].append({
        "label": "battery_side", "kind": "obb",
        "points": [[0, 0], [199, 0], [199, 119], [0, 119]]})
    review.stamp(data, review.make_review_record())
    persistence.save_annotation("aaa_sorts_first", name, data)

    _silence_dialogs(monkeypatch)
    win.export_all_yolo()
    yaml = (EXPORT_DIR / "all_labels_obb" / "data.yaml").read_text()
    assert "0: battery_side" in yaml
