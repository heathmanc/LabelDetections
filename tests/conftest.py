"""Shared builders. Kept terse so the tests read as scenarios, not setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_detections.core import annotations as ann
from label_detections.core.labels import CodeSpec, LabelDef, LabelLibrary, TextField
from label_detections.core.recipes import CrossCheck, LabelRequirement, Recipe, ViewSpec

# Every test frame is 1000 x 500, so a normalised ROI of [0.1, 0.2, 0.2, 0.4]
# is pixels (100, 100)-(300, 300) and the expectations stay readable.
FRAME = (1000, 500)


def rect(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


@pytest.fixture
def library():
    plate = LabelDef(label_id="spec_plate_31agm", family="spec_plate",
                     size_mm=[90.0, 60.0], reference_images=["ref.png"],
                     confusable_with=["spec_plate_27agm"])
    plate.codes = [CodeSpec(role="serial", symbology="datamatrix",
                            policy="must_match_pattern", pattern=r"^SN\d{6}$",
                            region_mm=[10, 10, 30, 30], x_dim_mm=0.254)]
    plate.text_fields = [TextField(name="date_code", region_mm=[10, 45, 60, 10])]

    tag = LabelDef(label_id="trace_tag", family="trace_tag", size_mm=[50.0, 25.0],
                   reference_images=["tag.png"])
    tag.codes = [CodeSpec(role="serial", symbology="code128", policy="must_decode",
                          region_mm=[5, 5, 40, 12], x_dim_mm=0.33)]

    warn = LabelDef(label_id="warning_en", family="warning_label", size_mm=[40.0, 30.0],
                    reference_images=["warn.png"], rotation_policy="fixed",
                    rotation_tol_deg=8.0)
    other = LabelDef(label_id="spec_plate_27agm", family="spec_plate",
                     size_mm=[90.0, 60.0], reference_images=["other.png"])
    promo = LabelDef(label_id="promo", family="promo_label", size_mm=[30.0, 20.0],
                     reference_images=["promo.png"])
    return LabelLibrary([plate, tag, warn, other, promo])


@pytest.fixture
def recipe():
    """Two cameras, ROIs, a serial cross-check and one forbidden look-alike."""
    side_a = ViewSpec(
        view="side_a", camera="cam1", frame_size=list(FRAME),
        labels=[
            LabelRequirement("spec_plate_31agm", roi=[0.05, 0.1, 0.3, 0.4],
                             severity="fail", roi_tol=0.02),
            LabelRequirement("warning_en", roi=[0.5, 0.1, 0.2, 0.3], severity="fail"),
        ],
        forbidden=["spec_plate_27agm"],
    )
    side_b = ViewSpec(
        view="side_b", camera="cam2", frame_size=list(FRAME),
        labels=[LabelRequirement("trace_tag", roi=[0.1, 0.1, 0.4, 0.4], severity="fail")],
    )
    return Recipe(
        group="AGM", model="31-AGM-950", views=[side_a, side_b],
        cross_checks=[CrossCheck(type="equal",
                                 left="side_a.spec_plate_31agm.serial",
                                 right="side_b.trace_tag.serial")],
    )


def frame(view="side_a", boxes=None, label_id="", **meta):
    """A frame annotation: what one camera saw, or one training image."""
    data = ann.new_annotation(f"{view}.jpg", label_id, FRAME[0], FRAME[1], view=view, **meta)
    data["boxes"].extend(boxes or [])
    return data


def label_box(label_id, family, x, y, w, h, **kw):
    return ann.make_box(family, rect(x, y, w, h), label_id=label_id, **kw)


def code_region(role, decoded, ok=True, **kw):
    return ann.make_region("code", rect(0, 0, 10, 10), code_role=role,
                           decoded=decoded, decode_ok=ok, **kw)
