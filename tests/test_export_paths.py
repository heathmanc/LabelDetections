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


# --- the two-stage pipeline: detect where, classify which --------------------

def test_the_two_stage_detector_learns_where_a_label_is_not_which(monkeypatch):
    """Under a crop pipeline the detector's job is location. Keeping one class
    per label there would put fine-grained identity back in the stage that has
    the fewest pixels to decide it with."""
    from label_detections.core import classify_export, storage

    _window()
    _dataset("ts_det_a", images=3)
    _dataset("ts_det_b", images=3)
    detect_dir, _ = classify_export.export_two_stage(
        out=storage.EXPORT_DIR / "ts_test_a")

    yaml = (detect_dir / "data.yaml").read_text()
    assert "0: battery_side" in yaml
    assert "1: label" in yaml
    assert "ts_det_a" not in yaml and "ts_det_b" not in yaml


def test_the_classifier_gets_one_folder_per_label_id(monkeypatch):
    from label_detections.core import classify_export, storage

    _window()
    _dataset("ts_cls_a", images=3)
    _, classify_dir = classify_export.export_two_stage(
        out=storage.EXPORT_DIR / "ts_test_b")

    classes = (classify_dir / "classes.txt").read_text().split()
    assert "ts_cls_a" in classes
    crops = list((classify_dir / "train" / "ts_cls_a").glob("*.jpg"))
    assert crops


def test_both_halves_hold_out_the_same_batteries(monkeypatch):
    """Split them separately and a crop of a battery the detector validates on
    can land in the classifier's training set. The measured accuracy then
    describes nothing."""
    from label_detections.core import classify_export, storage

    _window()
    for name in ("ts_split_a", "ts_split_b", "ts_split_c"):
        _dataset(name, images=4)
    detect_dir, classify_dir = classify_export.export_two_stage(
        out=storage.EXPORT_DIR / "ts_test_c")

    def groups(path, split_col=0, group_col=-1):
        out = {"train": set(), "val": set()}
        for row in path.read_text().splitlines()[1:]:
            parts = row.split(",")
            out.setdefault(parts[split_col], set()).add(parts[group_col])
        return out

    det = groups(detect_dir / "manifest.csv", group_col=4)
    cls = groups(classify_dir / "manifest.csv", group_col=3)
    assert det["val"] and cls["val"]
    assert not (cls["train"] & det["val"]), "a held-out battery leaked into training"
    assert det["train"] == cls["train"] and det["val"] == cls["val"]


def test_the_battery_face_is_never_offered_to_the_classifier():
    """battery_side is the face, not a label. A classifier taught to call it
    one would report it on every battery."""
    from label_detections.core import classify_export as ce

    data = {"boxes": [
        {"label": "battery_side", "points": [[0, 0], [9, 0], [9, 9], [0, 9]]},
        {"label": "sp", "label_id": "sp", "points": [[1, 1], [4, 1], [4, 4], [1, 4]]},
    ]}
    assert [lid for lid, _ in ce.crop_targets(data)] == ["sp"]


def test_a_box_with_no_identity_is_skipped_rather_than_guessed():
    from label_detections.core import classify_export as ce

    data = {"boxes": [{"kind": "obb", "points": [[0, 0], [9, 0], [9, 9], [0, 9]]}]}
    assert ce.crop_targets(data) == []


# --- label scale: does cropping actually help these labels? -----------------

def test_a_big_label_is_reported_as_losing_detail_to_a_small_crop():
    """The trap a fixed crop size sets. A label the detector already resolves
    to 500 px arrives at a 224 px classifier with less to go on, not more."""
    from label_detections.core import scale_report as sr

    scales = {"big": sr.LabelScale("big", [2000.0] * 5, [3840.0] * 5)}
    assert "LOSES" in sr.verdict(scales["big"], imgsz=640, crop=224)


def test_a_small_label_is_reported_as_gaining():
    from label_detections.core import scale_report as sr

    scales = sr.LabelScale("small", [300.0] * 5, [3840.0] * 5)
    assert "helps" in sr.verdict(scales, imgsz=640, crop=224)


def test_the_recommendation_is_sized_so_no_label_loses():
    """A mixed dataset cannot be served by a constant: the recommendation comes
    from the label the detector resolves best, because that is the one a crop
    can harm."""
    from label_detections.core import scale_report as sr

    scales = {
        "big": sr.LabelScale("big", [2000.0] * 4, [3840.0] * 4),
        "small": sr.LabelScale("small", [300.0] * 4, [3840.0] * 4),
    }
    crop = sr.recommend_crop(scales, imgsz=640)
    assert crop % sr.STRIDE == 0
    for scale in scales.values():
        assert "LOSES" not in sr.verdict(scale, imgsz=640, crop=crop)


def test_the_frame_each_box_came_from_is_paired_with_it():
    """400 px of label in a 1000 px frame and 400 px in a 4000 px frame reach
    the detector at completely different sizes. Averaging the frame widths
    separately would hide exactly that."""
    from label_detections.core import scale_report as sr

    scale = sr.LabelScale("mixed", [400.0, 400.0], [1000.0, 4000.0])
    assert scale.detector_px(640, "max") > scale.detector_px(640, "min") * 3


def test_the_battery_face_is_left_out_of_the_measurement():
    """battery_side is the frame's own size by definition, so including it
    would drag every statistic toward it."""
    from label_detections.core import scale_report as sr

    entries = [{"width": 400, "height": 240, "boxes": [
        {"label": "battery_side", "points": [[0, 0], [399, 0], [399, 239], [0, 239]]},
        {"label": "sp", "label_id": "sp", "points": [[10, 10], [90, 10], [90, 60], [10, 60]]},
    ]}]
    scales = sr.measure(entries)
    assert list(scales) == ["sp"]


def test_the_export_sizes_its_crops_from_the_data_not_a_default(monkeypatch):
    from label_detections.core import classify_export, storage
    import cv2

    _window()
    _dataset("scale_sized", images=3)
    _, classify_dir = classify_export.export_two_stage(
        out=storage.EXPORT_DIR / "ts_scaled", imgsz=640)
    crop = cv2.imread(str(next(classify_dir.rglob("*.jpg"))))
    assert crop is not None and crop.shape[0] == crop.shape[1]
    assert crop.shape[0] % 32 == 0


def test_the_crop_recommendation_is_never_silently_capped():
    """A capped number that reads like an answer is worse than a large one.
    At imgsz 1280 a 2000 px label needs 672, which used to clamp to 640 with
    nothing said."""
    from label_detections.core import scale_report as sr

    scales = {"big": sr.LabelScale("big", [2000.0] * 4, [3840.0] * 4)}
    assert sr.recommend_crop(scales, imgsz=1280) == 672
    assert sr.COSTLY_CROP < 672, "the costly-crop warning must still fire here"


def test_the_report_says_when_a_recommendation_is_expensive():
    from label_detections.core import scale_report as sr

    scales = {"big": sr.LabelScale("big", [2000.0] * 4, [3840.0] * 4),
              "small": sr.LabelScale("small", [300.0] * 4, [3840.0] * 4)}
    text = sr.report(scales, imgsz=1280, crop=224)
    assert "no longer cheap" in text
    assert "small" in text.split("crop stage is actually for")[0]


def test_labels_the_detector_already_resolves_are_not_the_ones_needing_a_crop():
    from label_detections.core import scale_report as sr

    scales = {"big": sr.LabelScale("big", [2000.0] * 4, [3840.0] * 4),
              "small": sr.LabelScale("small", [300.0] * 4, [3840.0] * 4)}
    assert sr.under_resolved(scales, imgsz=1280) == ["small"]


def test_a_region_is_measured_not_the_label_it_sits_on():
    """A 2000 px label whose revision block is 6% of it carries 120 px of
    deciding evidence, not 2000. Every conclusion from label width is off by
    that factor."""
    from label_detections.core import scale_report as sr
    from label_detections.core.labels import LabelDef, LabelLibrary, TextField

    lib = LabelLibrary([LabelDef(
        label_id="big", text_fields=[TextField(name="rev", region=[0.05, 0.4, 0.06, 0.08])])])
    scales = {"big": sr.LabelScale("big", [2000.0] * 4, [3840.0] * 4)}
    text = sr.region_report(scales, lib, imgsz=1280, crop=224)
    assert "6% of label" in text
    # 6% of 667 px at the detector; 6% of 224 after the crop -- the crop is worse.
    assert "40 px at detector" in text and "13 px after crop" in text


def test_a_code_is_checked_against_what_its_symbology_needs():
    from label_detections.core import scale_report as sr
    from label_detections.core.labels import CodeSpec, LabelDef, LabelLibrary

    lib = LabelLibrary([LabelDef(label_id="big", codes=[CodeSpec(
        role="part_number", symbology="code128", region=[0.05, 0.7, 0.30, 0.2],
        code_width_mm=40, x_dim_mm=0.33)])])
    scales = {"big": sr.LabelScale("big", [2000.0] * 4, [3840.0] * 4)}
    assert "NEITHER reaches" in sr.region_report(scales, lib, imgsz=1280, crop=224)


def test_no_read_regions_says_so_rather_than_reporting_nothing():
    from label_detections.core import scale_report as sr
    from label_detections.core.labels import LabelDef, LabelLibrary

    lib = LabelLibrary([LabelDef(label_id="plain")])
    scales = {"plain": sr.LabelScale("plain", [500.0] * 2, [3840.0] * 2)}
    assert "No read-regions defined yet" in sr.region_report(scales, lib, 1280, 224)


# --- region crops: the only stage that resolves fine print ------------------

def test_a_region_is_cropped_from_the_full_frame_not_the_detector_input():
    """The whole point: a 120 px revision block stays 120 px whatever imgsz the
    detector runs at, because nothing downscaled it."""
    from label_detections.core import classify_export as ce
    from label_detections.core.labels import LabelDef, LabelLibrary, TextField

    lib = LabelLibrary([LabelDef(
        label_id="rr", text_fields=[TextField(name="rev", region=[0.1, 0.4, 0.2, 0.1])])])
    data = {"boxes": [{"label": "rr", "label_id": "rr",
                       "points": [[0, 0], [1000, 0], [1000, 600], [0, 600]]}]}
    targets = ce.region_crop_targets(data, lib)
    assert len(targets) == 1
    label_id, name, quad = targets[0]
    assert label_id == "rr" and name == "text_rev"
    # 20% of a 1000 px label, placed by proportion.
    xs = [p[0] for p in quad]
    assert abs((max(xs) - min(xs)) - 200) < 2


def test_regions_are_foldered_by_label_so_ground_truth_is_free():
    """The label id already encodes which revision it is, so a classifier over
    these crops is trained without anyone labelling anything twice."""
    from label_detections.core import classify_export as ce, persistence, storage
    from label_detections.core.labels import LabelDef, TextField

    _window()
    for lid in ("rc_a", "rc_b"):
        _dataset(lid, images=3)
    # After _dataset, not before: it replaces the library row and would wipe
    # the regions back out.
    lib = persistence.load_library()
    for lid in ("rc_a", "rc_b"):
        label = lib.get(lid)
        label.text_fields = [TextField(name="rev", region=[0.1, 0.4, 0.3, 0.2])]
        lib.add(label, replace=True)
    persistence.save_library(lib)

    out = ce.export_region_crops(library=persistence.load_library(),
                                 out=storage.EXPORT_DIR / "rgn_test")
    classes = (out / "classes.txt").read_text().split()
    assert "rc_a" in classes and "rc_b" in classes
    assert list((out / "train" / "rc_a").glob("*.jpg"))


def test_no_regions_defined_explains_what_to_do():
    from label_detections.core import classify_export as ce
    from label_detections.core.labels import LabelDef, LabelLibrary

    lib = LabelLibrary([LabelDef(label_id="plain")])
    data = {"boxes": [{"label": "plain", "label_id": "plain",
                       "points": [[0, 0], [10, 0], [10, 10], [0, 10]]}]}
    assert ce.region_crop_targets(data, lib) == []


def test_the_advice_sizes_the_detector_from_the_smallest_label():
    """The smallest label runs out of pixels first, so it sets the floor --
    whichever branch the advice takes."""
    from label_detections.core import scale_report as sr

    scales = {"big": sr.LabelScale("big", [2000.0] * 4, [3840.0] * 4),
              "small": sr.LabelScale("small", [300.0] * 4, [3840.0] * 4)}
    text = sr.advise(scales, None, imgsz=1280)
    assert "small" in text
    assert str(sr.min_imgsz_for_identity(scales)) in text


def test_the_imgsz_requirement_is_never_silently_capped():
    """A 5472 px frame with 300 px labels honestly needs imgsz 2336. Returning
    a comfortable 2048 would have someone train a detector that still cannot
    see the label."""
    from label_detections.core import scale_report as sr

    scales = {"small": sr.LabelScale("small", [300.0] * 4, [5472.0] * 4)}
    # Derived from the floor rather than hard-coded, so tuning the threshold
    # does not silently turn this into a test of nothing.
    exact = sr.ADEQUATE_PX * 5472 / 300
    got = sr.min_imgsz_for_identity(scales)
    assert got >= exact and got - exact < sr.STRIDE
    assert got > sr.IMPRACTICAL_IMGSZ


def test_a_big_frame_pushes_the_advice_to_localise_then_crop():
    """The frame:imgsz ratio decides this, not preference. At 5472/1280 the
    detector throws away 4.3x and a small label arrives unidentifiable."""
    from label_detections.core import scale_report as sr

    scales = {n: sr.LabelScale(n, [w] * 4, [5472.0] * 4)
              for n, w in (("big", 2000), ("small", 300))}
    text = sr.advise(scales, None, imgsz=1280)
    assert "LOCALISE only" in text
    need = sr.min_imgsz_for_identity(scales)
    assert str(need) in text, "must state the requirement it is declining to meet"


def test_a_modest_frame_keeps_the_single_detector_advice():
    """The same code must not always say two-stage: where one detector can
    reach the identity floor, that is the simpler answer."""
    from label_detections.core import scale_report as sr

    scales = {n: sr.LabelScale(n, [w] * 4, [1920.0] * 4)
              for n, w in (("big", 900), ("small", 400))}
    text = sr.advise(scales, None, imgsz=1280)
    assert "one class per label" in text
    assert "LOCALISE only" not in text


def test_the_report_cannot_recommend_two_things_at_once():
    """It did. On real data section 1 said "single-stage, you are already
    there" and the working below said "a clean two-stage win" -- two
    thresholds, 128 and 256, answering one question in two places."""
    from label_detections.core import scale_report as sr

    scales = {"a": sr.LabelScale("a", [872.0] * 81, [5496.0] * 81),
              "b": sr.LabelScale("b", [3822.0], [5496.0])}
    assert sr.IDENTITY_FLOOR_PX == sr.ADEQUATE_PX
    weak = sr.under_resolved(scales, imgsz=1024)
    text = sr.advise(scales, None, imgsz=1024)
    # Whatever it concludes, both halves must agree that 'a' is the problem.
    assert weak == ["a"]
    assert "Under-resolved at this input: a" in text
    assert "you are already there" not in text.lower()


def test_a_single_capture_class_is_called_out_before_any_resolution_advice():
    """One box cannot be both trained and validated, and no amount of imgsz
    fixes that -- so it belongs above the resolution analysis, not after it."""
    from label_detections.core import scale_report as sr

    scales = {"many": sr.LabelScale("many", [872.0] * 81, [5496.0] * 81),
              "lonely": sr.LabelScale("lonely", [900.0], [5496.0])}
    issues = sr.data_health(scales, None)
    assert any("lonely: 1 box" in i for i in issues)
    text = sr.advise(scales, None, imgsz=1024)
    assert text.index("lonely: 1 box") < text.index("Frame 5496")


def test_a_box_covering_most_of_the_frame_is_questioned():
    """A battery face drawn as a label trains the detector to fire on every
    battery, and nothing else in the report would notice."""
    from label_detections.core import scale_report as sr

    scales = {"huge": sr.LabelScale("huge", [3822.0] * 30, [5496.0] * 30)}
    assert any("of the frame" in i for i in sr.data_health(scales, None))


def test_a_lopsided_class_balance_is_reported():
    from label_detections.core import scale_report as sr

    scales = {"big": sr.LabelScale("big", [800.0] * 81, [5496.0] * 81),
              "small": sr.LabelScale("small", [800.0] * 2, [5496.0] * 2)}
    assert any("Class balance" in i for i in sr.data_health(scales, None))


def test_a_healthy_dataset_is_not_nagged():
    from label_detections.core import scale_report as sr

    scales = {"a": sr.LabelScale("a", [800.0] * 60, [5496.0] * 60),
              "b": sr.LabelScale("b", [900.0] * 55, [5496.0] * 55)}
    assert sr.data_health(scales, None) == []


def test_a_large_library_is_advised_toward_the_generic_detector():
    """At hundreds of labels the deciding cost is onboarding, not pixels: one
    class per label means a full detector retrain for every new SKU."""
    from label_detections.core import scale_report as sr
    from label_detections.core.labels import LabelDef, LabelLibrary

    lib = LabelLibrary([LabelDef(label_id=f"lbl{i}") for i in range(300)])
    scales = {"a": sr.LabelScale("a", [872.0] * 81, [5496.0] * 81)}
    text = sr.advise(scales, lib, imgsz=1024)
    assert "B, at 300 labels" in text
    assert "onboarding, not pixels" in text


def test_a_small_library_is_still_advised_toward_the_simpler_option():
    """The same code must not always say B -- at two labels the second model
    is complexity bought for nothing."""
    from label_detections.core import scale_report as sr
    from label_detections.core.labels import LabelDef, LabelLibrary

    lib = LabelLibrary([LabelDef(label_id="a"), LabelDef(label_id="b")])
    scales = {"a": sr.LabelScale("a", [872.0] * 81, [5496.0] * 81)}
    text = sr.advise(scales, lib, imgsz=1024)
    assert "A is the simpler place to start" in text
