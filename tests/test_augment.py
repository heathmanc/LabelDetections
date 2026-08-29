"""Regions of a label that change between units, and whether they actually do.

A date code, a serial, a lot number: the artwork around them holds still and
they do not. Collect two hundred images in one afternoon off one lot and the
code is IDENTICAL in every one, which makes it a perfectly good shortcut for
"this is a PC680" until next month's date breaks detection for reasons nobody
can see.

This measures that and says so. It used to also offer a fix -- grafting a real
date code from one image into another's label -- and that is gone. It was
careful about it, and it was still a stopgap: two hundred images off one lot
carry one date code however they are recombined. The number of real codes in
the training set does not go up. A second lot is the only thing that adds one.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-augment-"))

import pytest

from label_detections.core import augment
from label_detections.core.labels import LabelDef, LabelLibrary, TextField

try:
    import cv2
    import numpy as np
    HAVE_CV = True
except Exception:  # pragma: no cover - depends on the environment
    HAVE_CV = False


# --- policy, no pixels needed ----------------------------------------------

def test_variable_regions_are_the_drawn_text_fields():
    label = LabelDef(label_id="sp")
    label.text_fields = [
        TextField(name="date_code", region=[0.1, 0.7, 0.4, 0.2]),
        TextField(name="never_drawn"),                       # no region
        TextField(name="degenerate", region=[0.1, 0.1, 0, 0]),
    ]
    assert augment.variable_regions(label) == [("date_code", [0.1, 0.7, 0.4, 0.2])]


def test_a_constant_region_is_the_one_worth_acting_on():
    """A region that varies is already ignored by the network; a constant one
    is a shortcut waiting to be learned."""
    assert augment.needs_randomising(0.0) is True
    assert augment.needs_randomising(0.2) is False


def test_the_verdict_explains_the_failure_rather_than_printing_a_number():
    constant = augment.variance_verdict("date_code", 0.0, 47)
    assert "looks the same in all 47" in constant
    assert "shortcut" in constant
    varied = augment.variance_verdict("date_code", 0.3, 47)
    assert "nothing to do" in varied


def test_too_few_images_is_reported_as_unknown_not_as_constant():
    assert "not enough to tell" in augment.variance_verdict("date_code", 0.0, 1)


def test_a_two_image_sample_is_not_treated_as_at_risk():
    lonely = augment.RegionReport("sp", "date_code", [0, 0, 1, 1], 0.0, 1)
    assert lonely.at_risk is False


# --- pixels ----------------------------------------------------------------

pytestmark_cv = pytest.mark.skipif(not HAVE_CV, reason="cv2/numpy not available")

BOX = {"label": "spec_plate", "label_id": "sp", "kind": "obb",
       "points": [[20, 20], [220, 20], [220, 140], [20, 140]]}
DATE_REGION = [0.1, 0.6, 0.6, 0.25]


def _plate(text: str):
    """A label plate with a date-code line, on a 260x180 frame."""
    image = np.full((180, 260, 3), 40, np.uint8)
    cv2.rectangle(image, (20, 20), (220, 140), (215, 217, 222), -1)
    cv2.putText(image, "G31-950", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (30, 30, 34), 2)
    cv2.putText(image, text, (35, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (30, 30, 34), 2)
    return image


@pytestmark_cv
def test_a_region_that_never_changes_scores_near_zero():
    crops = [augment.rectify_region(_plate("2026-01-04"), BOX, DATE_REGION)
             for _ in range(5)]
    assert augment.region_variance(crops) < augment.NEAR_IDENTICAL


@pytestmark_cv
def test_a_region_with_real_variation_scores_well_above_it():
    dates = ["2026-01-04", "2026-03-19", "2026-07-28", "2026-11-02", "2027-02-15"]
    crops = [augment.rectify_region(_plate(d), BOX, DATE_REGION) for d in dates]
    assert augment.region_variance(crops) > augment.NEAR_IDENTICAL


@pytestmark_cv
def test_brightness_alone_is_not_counted_as_variation():
    """Two shots of the same date code under different light differ everywhere.
    Counting that as variation would call a constant region varied -- backwards."""
    base = _plate("2026-01-04")
    crops = [augment.rectify_region(cv2.convertScaleAbs(base, alpha=1.0, beta=b),
                                    BOX, DATE_REGION)
             for b in (-40, -20, 0, 20, 40)]
    assert augment.region_variance(crops) < augment.NEAR_IDENTICAL


@pytestmark_cv
def test_rectifying_deskews_so_crops_from_any_angle_compare():
    upright = _plate("2026-01-04")
    tilted_box = {**BOX, "points": [[25, 18], [222, 30], [216, 145], [19, 133]]}
    a = augment.rectify_region(upright, BOX, DATE_REGION)
    b = augment.rectify_region(upright, tilted_box, DATE_REGION)
    assert a.shape == b.shape == (augment.COMPARE_SIZE[1], augment.COMPARE_SIZE[0], 3)


@pytestmark_cv
@pytestmark_cv
@pytestmark_cv
@pytestmark_cv
@pytestmark_cv
def test_a_blank_region_scores_zero_rather_than_dividing_by_nothing():
    blank = [augment.rectify_region(np.full((180, 260, 3), 200, np.uint8),
                                    BOX, DATE_REGION) for _ in range(3)]
    assert augment.region_variance(blank) == 0.0


@pytestmark_cv
def test_the_score_does_not_depend_on_how_many_images_were_compared():
    """An absolute grey-level metric ranked a 20-image varied set BELOW a
    constant one. The ratio has to hold steady as the sample grows."""
    dates = ["2026-01-04", "2026-03-19", "2026-07-28", "2026-11-02",
             "2027-02-15", "2027-06-08", "2028-09-21", "2029-12-30"]
    two = augment.region_variance([augment.rectify_region(_plate(d), BOX, DATE_REGION)
                                   for d in dates[:2]])
    eight = augment.region_variance([augment.rectify_region(_plate(d), BOX, DATE_REGION)
                                     for d in dates])
    assert two > augment.NEAR_IDENTICAL and eight > augment.NEAR_IDENTICAL
    assert abs(two - eight) < 0.1


# --- through the exporter ---------------------------------------------------

def _library(vary_region=True):
    label = LabelDef(label_id="sp", train_target=5)
    if vary_region:
        label.text_fields = [TextField(name="date_code", region=DATE_REGION)]
    return LabelLibrary([label])


def _entries(tmp_path, dates):
    """A dataset of reviewed images, one per date string given."""
    from label_detections.core import dataset as ds

    entries = []
    for index, date in enumerate(dates):
        path = tmp_path / f"cap_{index:03d}.jpg"
        cv2.imwrite(str(path), _plate(date))
        entries.append(ds.Entry(
            label_id="sp", image=str(path), session=f"s{index}",
            annotation={"image": str(path), "label_id": "sp",
                        "width": 260, "height": 180, "boxes": [dict(BOX)]},
        ))
    return entries


CONSTANT = ["2026-01-04"] * 6
VARIED = ["2026-01-04", "2026-03-19", "2026-07-28", "2026-11-02",
          "2027-02-15", "2028-09-21"]


@pytestmark_cv
@pytestmark_cv
@pytestmark_cv
@pytestmark_cv
@pytestmark_cv
@pytestmark_cv
def test_the_check_ships_with_the_dataset_either_way(tmp_path):
    """A region that is the same picture in every image is worth knowing about
    before training, not after a month of drift."""
    from label_detections.core import yolo_export

    out = tmp_path / "ds"
    yolo_export.write_dataset(out, _entries(tmp_path, CONSTANT),
                              library=_library(), seed=1)
    report = (out / "variable_regions.txt").read_text(encoding="utf-8")
    assert "date_code" in report
    assert "looks the same" in report


@pytestmark_cv
def test_a_label_with_no_variable_regions_is_left_entirely_alone(tmp_path):
    from label_detections.core import yolo_export

    out = tmp_path / "ds"
    yolo_export.write_dataset(out, _entries(tmp_path, CONSTANT),
                              library=_library(vary_region=False), seed=1)
    assert not (out / "variable_regions.txt").exists()


@pytestmark_cv
def test_scan_text_names_what_to_do_about_it(tmp_path):
    reports = augment.scan_entries(_entries(tmp_path, CONSTANT), _library())
    text = augment.scan_text(reports)
    assert "date_code" in text
    assert "shortcut" in text
    assert "1 region(s) are constant enough" in text


@pytestmark_cv
def test_the_export_writes_no_image_that_was_not_photographed(tmp_path):
    """Every row in the manifest points at a real capture. The manifest used to
    carry an `augmented` column and rows ending in ,1 -- images composed out of
    two others, counted in the training set as though they were evidence."""
    from label_detections.core import yolo_export

    out = tmp_path / "ds"
    yolo_export.write_dataset(out, _entries(tmp_path, CONSTANT),
                              library=_library(), seed=1)
    lines = (out / "manifest.csv").read_text(encoding="utf-8").splitlines()
    assert "augmented" not in lines[0]

    written = sorted(p.name for p in (out / "images").rglob("*.jpg"))
    assert written and not [n for n in written if "__aug" in n]
