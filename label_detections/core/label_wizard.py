"""The add-a-label questionnaire.

What gets asked here is the whole reason a new label SKU costs minutes instead
of a retraining cycle. Every answer either feeds identification (so the runtime
can name this label without a model that knows it), verification (so a code or
a date can be judged), or dataset work (so the label can be trained and
audited).

The questions that earn their keep, and why:

* **physical size** -- gives every detection a millimetre scale, so a box that
  swallowed two labels is caught as a misdetection instead of being reported
  as a defect on the battery.
* **variable data / anchor** -- a label carrying a per-unit serial matches
  against its unchanging artwork only. Without this every unit looks like a
  mismatch.
* **code region on the artwork** -- the runtime crops straight to the barcode
  from the full-resolution frame instead of searching for it, which is both
  faster and the difference between decoding a 10-mil DataMatrix and not.
* **surface** -- gloss and foil decide whether reference matching works at all
  and whether the line needs cross-polarisation.
* **severity** -- what missing this label should do. Asked once, here, rather
  than re-litigated in every recipe.
"""
from __future__ import annotations

import re
from typing import Any

from .labels import (
    CODE_POLICIES, CODE_ROLES, DEFAULT_FAMILIES, ROTATION_POLICIES, SEVERITIES,
    SHAPES, SURFACES, SYMBOLOGIES, CodeSpec, LabelDef, TextField,
)
from .storage import safe_token
from .wizard import Flow, Page, Question


def _valid_regex(value: Any, _answers: dict[str, Any]) -> str:
    try:
        re.compile(str(value))
    except re.error as exc:
        return f"Not a valid regular expression: {exc}"
    return ""


def _positive_size(value: Any, _answers: dict[str, Any]) -> str:
    try:
        w, h = float(value[0]), float(value[1])
    except Exception:
        return "Size must be width and height in mm."
    if w <= 0 or h <= 0:
        return ("Physical size must be greater than zero -- it is the scale sanity "
                "check that separates a misdetection from a defect.")
    return ""


def _rect(value: Any, _answers: dict[str, Any]) -> str:
    if not value:
        return ""
    try:
        x, y, w, h = (float(v) for v in list(value)[:4])
    except Exception:
        return "Region must be x, y, width, height in mm."
    if w <= 0 or h <= 0:
        return "Region width and height must be greater than zero."
    if x < 0 or y < 0:
        return "Region origin cannot be negative -- it is measured from the label's top-left."
    return ""


CODE_COLUMNS = [
    Question("role", "Role", "choice", choices=CODE_ROLES, default="serial",
             help="What the code means. Cross-checks between labels refer to this name."),
    Question("symbology", "Symbology", "choice", choices=SYMBOLOGIES, default="datamatrix"),
    Question("policy", "Policy", "choice", choices=CODE_POLICIES, default="must_decode",
             help="present = the code must be there; decode = it must be readable."),
    Question("region_mm", "Region on artwork (mm)", "rect_mm", validator=_rect,
             help="Draw it on the reference image. Lets the runtime crop straight to the code."),
    Question("x_dim_mm", "X-dimension / cell (mm)", "float", default=0.254,
             help="Narrow-bar width, or 2D cell size. Decides whether your camera can read it."),
    Question("quiet_zone_mm", "Quiet zone (mm)", "float", default=2.54),
    Question("pattern", "Content pattern", "text", validator=_valid_regex,
             placeholder="^SN[0-9]{10}$"),
    Question("grade", "Print-grade it", "bool", default=False,
             help="ISO 15415/15416 grading. Slower; use where a customer demands it."),
]

TEXT_COLUMNS = [
    Question("name", "Field name", "text", required=True, placeholder="date_code"),
    Question("region_mm", "Region on artwork (mm)", "rect_mm", validator=_rect),
    Question("policy", "Policy", "choice",
             choices=["ignore", "must_be_present", "must_match_pattern"],
             default="must_be_present"),
    Question("pattern", "Pattern", "text", validator=_valid_regex,
             placeholder=r"^\d{4}-\d{2}-\d{2}$"),
    Question("max_age_days", "Max age (days)", "int", default=0,
             help="Date codes only. 0 disables the age check."),
]


PAGES = [
    Page(
        "identity", "Identity",
        blurb="What this label is called and which revision of it this is.",
        questions=[
            Question("label_id", "Label ID", "text", required=True,
                     placeholder="spec_plate_31agm",
                     help="Machine-safe id used by recipes and reports. Cannot change later."),
            Question("name", "Description", "text", required=True,
                     placeholder="31-AGM spec plate, English"),
            Question("part_number", "Label stock part number", "text",
                     help="The purchased label's own part number, not the battery's."),
            Question("vendor", "Vendor", "text"),
            Question("revision", "Artwork revision", "text", placeholder="C"),
            Question("effective_date", "Revision effective from", "text",
                     placeholder="2026-01-15",
                     help="Lets a changeover accept the old and new revision for a period."),
            Question("supersedes", "Supersedes label", "label_picker",
                     help="The label this one replaces, if any."),
        ],
    ),
    Page(
        "appearance", "Appearance",
        blurb="How the label looks, so it can be found and identified.",
        questions=[
            Question("family", "Detector family", "choice", choices=DEFAULT_FAMILIES,
                     default="spec_plate", required=True,
                     help="The coarse class the model is trained on. Adding a label to an "
                          "existing family needs no retraining."),
            Question("reference_images", "Reference images", "paths", required=True,
                     help="Capture three or more under production lighting. Artwork files "
                          "alone match poorly against a real, slightly glared label."),
            Question("size_mm", "Physical size W x H (mm)", "size_mm", required=True,
                     validator=_positive_size,
                     help="The scale check: a detection far off this size is a misdetection."),
            Question("shape", "Shape", "choice", choices=SHAPES, default="rectangle"),
            Question("surface", "Surface", "choice", choices=SURFACES, default="matte",
                     help="Gloss and foil glare. They decide lighting and whether matching works."),
            Question("color_significant", "Colour distinguishes this label", "bool",
                     default=False,
                     help="Set when another label has identical artwork in a different colour."),
        ],
    ),
    Page(
        "orientation", "Orientation and variable data",
        questions=[
            Question("rotation_policy", "Rotation allowed", "choice",
                     choices=ROTATION_POLICIES, default="fixed",
                     help="fixed = one way up; flip_ok = 180 degrees is acceptable."),
            Question("rotation_tol_deg", "Rotation tolerance (deg)", "float", default=8.0,
                     visible_when={"rotation_policy": ["fixed", "flip_ok"]}),
            Question("variable_data", "Artwork changes per unit", "bool", default=False,
                     help="Serial, lot or date printed per battery."),
            Question("anchor_region_mm", "Static anchor region (mm)", "rect_mm",
                     visible_when={"variable_data": True}, validator=_rect,
                     help="The part that never changes. Matching scores against this only."),
        ],
    ),
    Page(
        "codes", "Barcodes and 2D codes",
        blurb="One row per code on the label. Leave empty if it carries none.",
        questions=[
            Question("codes", "Codes", "table", columns=CODE_COLUMNS),
        ],
    ),
    Page(
        "text", "Text fields",
        blurb="Printed text that inspection must read. Leave empty to skip OCR.",
        questions=[
            Question("text_fields", "Text fields", "table", columns=TEXT_COLUMNS),
        ],
    ),
    Page(
        "behaviour", "Inspection behaviour",
        questions=[
            Question("default_severity", "If this label is missing", "choice",
                     choices=SEVERITIES, default="fail",
                     help="The default a recipe inherits. A recipe can override it."),
            Question("min_confidence", "Minimum detection confidence", "float", default=0.5),
            Question("notes", "Notes", "textarea"),
        ],
    ),
    Page(
        "training", "Training this label",
        blurb="This label gets its own dataset and is trained on its own schedule. "
              "Nothing here affects any other label.",
        questions=[
            Question("train_target", "Images to gather before training", "int", default=150,
                     help="A working target, not a law. Distinctive artwork on a stable "
                          "line trains on far fewer; a foil label in changing light needs "
                          "far more."),
            Question("confusable_with", "Looks like these labels", "multichoice",
                     choices_from="__library_ids",
                     help="The look-alikes. Their images become hard negatives when this "
                          "label is trained, which is what stops the two being swapped -- "
                          "and they are the labels worth putting on a recipe's forbidden "
                          "list."),
            Question("synthetic_ok", "Generate synthetic training samples", "bool",
                     default=True,
                     help="Composite the artwork at random perspective, lighting and blur "
                          "onto real backgrounds. Bootstraps a label long before the line "
                          "has produced enough real examples of it."),
        ],
    ),
]


def build_label(answers: dict[str, Any]) -> LabelDef:
    """Turn wizard answers into a ``LabelDef``."""
    codes = [
        CodeSpec(
            role=str(row.get("role", "other")),
            symbology=str(row.get("symbology", "datamatrix")),
            policy=str(row.get("policy", "must_decode")),
            pattern=str(row.get("pattern", "") or ""),
            region_mm=[float(v) for v in (row.get("region_mm") or [])[:4]],
            x_dim_mm=float(row.get("x_dim_mm", 0.0) or 0.0),
            quiet_zone_mm=float(row.get("quiet_zone_mm", 0.0) or 0.0),
            grade=bool(row.get("grade", False)),
        )
        for row in answers.get("codes", []) or []
    ]
    text_fields = [
        TextField(
            name=str(row.get("name", "") or ""),
            region_mm=[float(v) for v in (row.get("region_mm") or [])[:4]],
            pattern=str(row.get("pattern", "") or ""),
            policy=str(row.get("policy", "must_be_present")),
            max_age_days=int(row.get("max_age_days", 0) or 0),
        )
        for row in answers.get("text_fields", []) or []
    ]
    size = answers.get("size_mm") or [0.0, 0.0]
    label = LabelDef(
        label_id=safe_token(answers.get("label_id", ""), "unnamed_label"),
        name=str(answers.get("name", "") or ""),
        family=str(answers.get("family", "spec_plate")),
        revision=str(answers.get("revision", "") or ""),
        effective_date=str(answers.get("effective_date", "") or ""),
        supersedes=str(answers.get("supersedes", "") or ""),
        part_number=str(answers.get("part_number", "") or ""),
        vendor=str(answers.get("vendor", "") or ""),
        size_mm=[float(size[0]), float(size[1])],
        shape=str(answers.get("shape", "rectangle")),
        surface=str(answers.get("surface", "matte")),
        color_significant=bool(answers.get("color_significant", False)),
        rotation_policy=str(answers.get("rotation_policy", "fixed")),
        rotation_tol_deg=float(answers.get("rotation_tol_deg", 8.0) or 0.0),
        variable_data=bool(answers.get("variable_data", False)),
        anchor_region_mm=[float(v) for v in (answers.get("anchor_region_mm") or [])[:4]],
        reference_images=[str(p) for p in answers.get("reference_images", []) or []],
        default_severity=str(answers.get("default_severity", "fail")),
        min_confidence=float(answers.get("min_confidence", 0.5) or 0.0),
        synthetic_ok=bool(answers.get("synthetic_ok", True)),
        train_target=int(answers.get("train_target", 150) or 0),
        confusable_with=[str(v) for v in answers.get("confusable_with", []) or []],
        notes=str(answers.get("notes", "") or ""),
    )
    label.codes = codes
    label.text_fields = text_fields
    return label


def review_answers(answers: dict[str, Any]) -> list[str]:
    """Cross-question warnings shown on the wizard's summary page.

    These are the mistakes that pass every field validator and still produce a
    label that quietly under-inspects.
    """
    warnings: list[str] = []
    label = build_label(answers)

    if label.variable_data and len(label.anchor_region_mm) < 4:
        warnings.append(
            "This label changes per unit but has no anchor region, so matching will "
            "score against text that is different on every battery."
        )
    if label.surface in ("gloss", "foil", "holographic") and not label.codes:
        warnings.append(
            f"A {label.surface} surface glares badly. Confirm the line has diffuse or "
            "cross-polarised lighting before relying on artwork matching."
        )
    for i, code in enumerate(label.codes, start=1):
        if code.policy in ("must_decode", "must_match_pattern") and not code.region_mm:
            warnings.append(
                f"Code {i} ({code.role}) must decode but has no region on the artwork, "
                "so the runtime has to search the whole label instead of cropping to it."
            )
        needed = code.min_pixels_needed()
        if needed:
            warnings.append(
                f"Code {i} ({code.role}, {code.symbology}) needs about {needed:.0f} px "
                f"across {code.region_mm[2]:.0f} mm to decode reliably -- check the "
                "camera resolution over its field of view."
            )
    if label.family == "battery_side":
        warnings.append(
            "'battery_side' is the whole battery face, not a label. Pick the family "
            "this label actually belongs to."
        )
    if not label.confusable_with:
        warnings.append(
            "No look-alikes named. If another label resembles this one, listing it here "
            "makes it a hard negative during training and a candidate for a recipe's "
            "forbidden list -- which is what catches a wrong label, since a wrong label "
            "is present and correct-looking."
        )
    return warnings


FLOW = Flow(
    key="add_label",
    title="Add a label to the library",
    pages=PAGES,
    build=build_label,
    review=review_answers,
)
