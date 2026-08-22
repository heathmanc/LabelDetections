"""Tests for background (negative) images.

These drive the real storage and export code against a temporary library, so
the round trip that matters is covered: mark an image background -> it survives
a save -> it reaches the exported dataset as an empty label file.

BUNGVISION_DATA_DIR is redirected before importing storage, because the module
resolves its data root at import time.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="bunglabel_bg_test_")
os.environ["BUNGVISION_DATA_DIR"] = _TMP

import numpy as np  # noqa: E402

from bung_labeler.core import dataset_health as dh  # noqa: E402
from bung_labeler.core import review as review_logic  # noqa: E402
from bung_labeler.core import storage  # noqa: E402
from bung_labeler.core import yolo_export  # noqa: E402


def _recipe(name: str = "Bg_Test") -> storage.Recipe:
    return storage.Recipe(group="Test", model=name)


def _frame(w: int = 64, h: int = 48) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _battery_box() -> dict:
    return {
        "label": "battery", "class_id": 0, "kind": "obb",
        "points": [[4, 4], [40, 4], [40, 40], [4, 40]],
    }


# --- the background flag itself -------------------------------------------

def test_no_boxes_alone_is_not_a_background():
    # Unfinished work must never be exported as a negative sample.
    assert not review_logic.is_background_annotation({"boxes": []})
    assert not review_logic.is_background_annotation(None)


def test_explicit_flag_is_a_background():
    assert review_logic.is_background_annotation({"boxes": [], "background": True})


def test_boxes_override_the_flag():
    data = {"boxes": [_battery_box()], "background": True}
    assert not review_logic.is_background_annotation(data)


def test_saving_boxes_clears_a_stale_background_flag():
    recipe = _recipe("clear_flag")
    img, _ = storage.save_capture(recipe, _frame())
    storage.save_annotations(img, 64, 48, [], [], background=True)
    assert review_logic.is_background_annotation(storage.load_annotations(img))

    # Operator changes their mind and labels the image after all.
    storage.save_annotations(img, 64, 48, [_battery_box()], ["battery"])
    data = storage.load_annotations(img)
    assert data["background"] is False
    assert not review_logic.is_background_annotation(data)


def test_background_flag_survives_a_later_save_that_does_not_mention_it():
    recipe = _recipe("survive")
    img, _ = storage.save_capture(recipe, _frame())
    storage.save_annotations(img, 64, 48, [], [], background=True)
    # background=None means "leave whatever is there alone".
    storage.save_annotations(img, 64, 48, [], [])
    assert review_logic.is_background_annotation(storage.load_annotations(img))


# --- health classification -------------------------------------------------

def test_background_status_is_distinct_from_empty():
    assert dh.annotation_status({"boxes": []}, 6) == "empty"
    assert dh.annotation_status({"boxes": [], "background": True}, 6) == "background"


def test_backgrounds_count_as_export_ready():
    tally = dh.tally_statuses(["background", "background", "empty"])
    assert dh.export_ready(tally) == 2


def test_backgrounds_are_not_counted_as_labeled():
    # They carry no boxes, so counting them as labeled would overstate the
    # dataset and hide that there is nothing to train on.
    tally = dh.tally_statuses(["background"])
    assert tally["labeled"] == 0


# --- import ---------------------------------------------------------------

def test_import_as_background_marks_every_image():
    recipe = _recipe("import_bg")
    src_dir = Path(tempfile.mkdtemp())
    import cv2
    srcs = []
    for i in range(3):
        p = src_dir / f"conveyor_{i}.png"
        cv2.imwrite(str(p), _frame())
        srcs.append(p)

    imported, errors, label_count = storage.import_images(recipe, srcs, as_background=True)
    assert not errors
    assert len(imported) == 3
    assert label_count == 3
    for p in imported:
        data = storage.load_annotations(p)
        assert review_logic.is_background_annotation(data)
        # Reviewed, so it is export-eligible without another operator pass.
        assert review_logic.annotation_reviewed(data)


# --- export ---------------------------------------------------------------

def _export_fixture(name: str) -> storage.Recipe:
    """One labeled battery image plus one background image."""
    recipe = _recipe(name)
    labeled, _ = storage.save_capture(recipe, _frame())
    storage.save_annotations(
        labeled, 64, 48, [_battery_box()], ["battery"],
        review=review_logic.make_review_record("test"),
    )
    background, _ = storage.save_capture(recipe, _frame())
    storage.save_annotations(
        background, 64, 48, [], [],
        review=review_logic.make_background_record(), background=True,
    )
    return recipe


def test_obb_export_includes_backgrounds_as_empty_label_files():
    recipe = _export_fixture("obb_export")
    out = yolo_export.export_recipe_obb(recipe.safe_name, split_train=0.5)

    labels = sorted((out / "labels").rglob("*.txt"))
    assert len(labels) == 2, "both the labeled and the background image should export"
    texts = sorted(p.read_text().strip() for p in labels)
    assert texts[0] == "", "the background must export as an empty label file"
    assert texts[1].startswith("0 "), "the labeled image must still export its box"

    images = sorted((out / "images").rglob("*.jpg"))
    assert len(images) == 2


def test_detect_export_includes_backgrounds_as_empty_label_files():
    recipe = _export_fixture("detect_export")
    out = yolo_export.export_recipe_yolo(recipe.safe_name, split_train=0.5)
    texts = sorted(p.read_text().strip() for p in (out / "labels").rglob("*.txt"))
    assert texts == ["", "0 0.343750 0.458333 0.562500 0.750000"]


def test_background_only_export_is_refused():
    # A dataset of nothing but negatives has no classes and cannot train.
    recipe = _recipe("bg_only")
    img, _ = storage.save_capture(recipe, _frame())
    storage.save_annotations(
        img, 64, 48, [], [],
        review=review_logic.make_background_record(), background=True,
    )
    for exporter in (yolo_export.export_recipe_obb, yolo_export.export_recipe_yolo):
        try:
            exporter(recipe.safe_name)
        except FileNotFoundError as e:
            assert "background" in str(e).lower()
        else:
            raise AssertionError(f"{exporter.__name__} should refuse a background-only export")


def test_unfinished_empty_annotations_still_stay_out_of_the_export():
    recipe = _recipe("unfinished")
    labeled, _ = storage.save_capture(recipe, _frame())
    storage.save_annotations(
        labeled, 64, 48, [_battery_box()], ["battery"],
        review=review_logic.make_review_record("test"),
    )
    blank, _ = storage.save_capture(recipe, _frame())
    # Reviewed but never flagged background: an accident, not a negative.
    storage.save_annotations(
        blank, 64, 48, [], [], review=review_logic.make_review_record("test"),
    )
    out = yolo_export.export_recipe_obb(recipe.safe_name, split_train=0.5)
    # A single entry is duplicated into both splits, so compare distinct names.
    names = {p.name for p in (out / "images").rglob("*.jpg")}
    assert names == {labeled.name}, f"only the labeled image should export, got {names}"


if __name__ == "__main__":
    import shutil
    import traceback
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception:
                failures += 1; print(f"FAIL {name}"); traceback.print_exc()
    shutil.rmtree(_TMP, ignore_errors=True)
    raise SystemExit(1 if failures else 0)
