"""Shared builders. Kept terse so the tests read as scenarios, not setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from label_detections.core import annotations as ann
from label_detections.core.labels import CodeSpec, LabelDef, LabelLibrary, TextField

# Test frames are 1000 x 500 so pixel expectations stay readable.
FRAME = (1000, 500)


def rect(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


def reference_for(label_id: str) -> str:
    """A real reference image on disk for a test label.

    It has to exist rather than merely be named. A label with no artwork is
    refused everywhere now -- every region on it is a fraction of an outline
    drawn on a picture -- so a helper that pointed at a filename nobody wrote
    described a label the app would not open.
    """
    import numpy as np

    from label_detections.core.imageio import save_reference

    return str(save_reference(label_id, np.full((60, 90, 3), 220, np.uint8)))


@pytest.fixture
def library():
    plate = LabelDef(label_id="spec_plate_31agm", reference_images=["ref.png"],
                     confusable_with=["spec_plate_27agm"])
    plate.codes = [CodeSpec(role="serial", symbology="datamatrix",
                            policy="must_match_pattern", pattern=r"^SN\d{6}$",
                            region=[0.1, 0.1, 0.33, 0.5],
                            code_width_mm=30.0, x_dim_mm=0.254)]
    plate.text_fields = [TextField(name="date_code", region=[0.1, 0.75, 0.66, 0.16])]

    tag = LabelDef(label_id="trace_tag", reference_images=["tag.png"])
    tag.codes = [CodeSpec(role="serial", symbology="code128", policy="must_decode",
                          region=[0.1, 0.2, 0.8, 0.48],
                          code_width_mm=40.0, x_dim_mm=0.33)]

    warn = LabelDef(label_id="warning_en", reference_images=["warn.png"])
    other = LabelDef(label_id="spec_plate_27agm", reference_images=["other.png"])
    return LabelLibrary([plate, tag, warn, other])


def frame(label_id="spec_plate_31agm", boxes=None, **meta):
    """One training image's sidecar."""
    data = ann.new_annotation(f"{label_id}.jpg", label_id, FRAME[0], FRAME[1], **meta)
    data["boxes"].extend(boxes or [])
    return data


def label_box(label_id, family, x, y, w, h, **kw):
    return ann.make_box(family, rect(x, y, w, h), label_id=label_id, **kw)


def code_region(role, decoded, ok=True, **kw):
    return ann.make_region("code", rect(0, 0, 10, 10), code_role=role,
                           decoded=decoded, decode_ok=ok, **kw)
