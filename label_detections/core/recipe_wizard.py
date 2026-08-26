"""The build-a-recipe questionnaire.

A recipe is the vision program's bill of labels: per camera, which labels must
be present and **where to look for each one**. The wizard walks that in the
order an operator can answer it -- name the model, declare the cameras, fill
each camera's bill from the label library with an ROI per label, forbid the
look-alikes, then wire up the cross-checks.

The labeling tool never reads this. Labels are trained one at a time against
their own datasets; the recipe only assembles already-trained labels into an
inspection, which is why a new recipe costs minutes and needs no model work at
all when its labels already exist.

Three things here are worth more than they look:

* **Cameras before labels.** Every later page derives its choices from the
  views declared on page two, so a bill can never name a camera that does not
  exist, and a cross-check can never reference one.
* **The forbidden list.** Most wrong-label escapes are not a missing label --
  they are the *neighbouring model's* label, which is present, correct-looking
  and completely wrong. Nothing in the required bill notices it.
* **Cross-checks.** With one camera per side, no single image ever sees two
  labels that must agree. The check has to live at the battery, and this is
  where it gets written.
"""
from __future__ import annotations

import re
from typing import Any

from .labels import SEVERITIES
from .recipes import (
    CROSS_CHECK_TYPES, DEFAULT_CATEGORY, CrossCheck, LabelRequirement, Recipe,
    ViewSpec, count_spec_error, normalise_roi, parse_count, rois_overlap,
    validate_recipe,
)
from .wizard import Flow, Page, Question


def _valid_count(value: Any, _answers: dict[str, Any]) -> str:
    error = count_spec_error(value)
    if error:
        return error
    lo, hi = parse_count(value)
    if lo == 0 and hi == 0:
        return "A required count of zero belongs on the forbidden list instead."
    return ""


def _valid_regex(value: Any, _answers: dict[str, Any]) -> str:
    try:
        re.compile(str(value))
    except re.error as exc:
        return f"Not a valid regular expression: {exc}"
    return ""


def _valid_ref(value: Any, _answers: dict[str, Any]) -> str:
    parts = [p for p in str(value or "").split(".") if p]
    if len(parts) != 3:
        return f"'{value}' must be view.label_id.role, e.g. side_a.spec_plate.serial"
    return ""


def _valid_frame(value: Any, _answers: dict[str, Any]) -> str:
    if not value or not any(value):
        return ""
    try:
        w, h = int(value[0]), int(value[1])
    except Exception:
        return "Frame size must be width and height in pixels."
    if w <= 0 or h <= 0:
        return "Frame size must be greater than zero."
    return ""


def _valid_roi(value: Any, _answers: dict[str, Any]) -> str:
    """ROIs are fractions of the frame, which is the part people get wrong first."""
    if not value:
        return ""
    roi = normalise_roi(value)
    if not roi:
        return "ROI must be x, y, width, height with a width and height above zero."
    x, y, w, h = roi
    if max(x, y, w, h) > 1.0001:
        return (f"ROI values are fractions of the frame, 0 to 1 -- got "
                f"({x:g}, {y:g}, {w:g}, {h:g}). Draw it on a reference image rather "
                "than typing pixels.")
    if x < 0 or y < 0 or x + w > 1.0001 or y + h > 1.0001:
        return "ROI runs outside the frame."
    return ""


VIEW_COLUMNS = [
    Question("view", "View name", "text", required=True, placeholder="side_a",
             help="Used in file paths, reports and cross-check references."),
    Question("camera", "Camera", "text", placeholder="basler_01"),
    Question("frame_size", "Frame size W x H (px)", "frame_size",
             validator=_valid_frame,
             help="This camera's resolution. Only used to draw ROIs and to convert "
                  "them to pixels; the ROIs themselves are stored as fractions, so "
                  "a camera swap does not invalidate them."),
    Question("unexpected_severity", "Unlisted label found", "choice",
             choices=SEVERITIES, default="warn",
             help="What to do about a real label the bill never asked for."),
]

BILL_COLUMNS = [
    Question("view", "View", "choice", required=True,
             choices_from="views", choices_field="view"),
    Question("label_id", "Label", "label_picker", required=True),
    Question("count", "Count", "count", default=1, validator=_valid_count,
             help="1, or a range like 0..1 or 1..*"),
    Question("severity", "If wrong", "choice", choices=SEVERITIES, default="fail"),
    Question("roi", "ROI", "roi", validator=_valid_roi,
             help="Drag it on a reference photo from this camera. Scopes the search "
                  "and locates the result. Empty means anywhere in the frame."),
    Question("roi_tol", "ROI slack", "float", default=0.02,
             help="Fraction of the frame allowed outside the ROI, for fixture play."),
    Question("rotation_tol_deg", "Rotation tolerance (deg)", "float", default=0.0,
             help="0 inherits the tolerance set on the label itself."),
    Question("notes", "Notes", "text"),
]

FORBIDDEN_COLUMNS = [
    Question("view", "View", "choice", required=True,
             choices_from="views", choices_field="view"),
    Question("label_id", "Label that must not appear", "label_picker", required=True),
]

CROSS_COLUMNS = [
    Question("type", "Check", "choice", choices=CROSS_CHECK_TYPES, default="equal"),
    Question("left", "Value", "text", required=True, validator=_valid_ref,
             placeholder="side_a.spec_plate_31agm.serial"),
    Question("right", "Must equal", "text", validator=_valid_ref,
             visible_when={"type": ["equal", "not_equal"]},
             placeholder="side_b.trace_tag.serial"),
    Question("pattern", "Pattern", "text", validator=_valid_regex,
             visible_when={"type": "pattern"}, placeholder="^SN[0-9]{10}$"),
    Question("severity", "If it fails", "choice", choices=SEVERITIES, default="fail"),
]


PAGES = [
    Page(
        "identity", "Battery model",
        blurb="Which product this recipe inspects.",
        questions=[
            Question("category", "Category", "text", default=DEFAULT_CATEGORY,
                     help="Broad equipment grouping, so one install can hold several lines."),
            Question("group", "Group", "text", required=True, placeholder="AGM"),
            Question("model", "Model", "text", required=True, placeholder="31-AGM-950"),
            Question("revision", "Recipe revision", "text", placeholder="C",
                     help="Logged into every inspection report. Change it when the bill changes."),
            Question("constrained", "Enforce the bill of labels", "bool", default=True,
                     help="Off makes this a free-form recipe: any labels pass. Use it for "
                          "background and conveyor captures."),
            Question("notes", "Notes", "textarea"),
        ],
    ),
    Page(
        "views", "Cameras",
        blurb="One row per camera. Every later page draws its choices from these.",
        visible_when={"constrained": True},
        questions=[
            Question("views", "Views", "table", columns=VIEW_COLUMNS, required=True),
        ],
    ),
    Page(
        "bill", "Bill of labels",
        blurb="What each camera must see. One row per label per view.",
        visible_when={"constrained": True},
        questions=[
            Question("bill", "Required labels", "table", columns=BILL_COLUMNS, required=True),
        ],
    ),
    Page(
        "forbidden", "Look-alikes",
        blurb="Labels that must NOT appear -- usually the neighbouring model's. "
              "This is the check that catches a wrong label, since a wrong label is "
              "present and correct-looking; nothing else will flag it.",
        visible_when={"constrained": True},
        questions=[
            Question("forbidden", "Forbidden labels", "table", columns=FORBIDDEN_COLUMNS),
        ],
    ),
    Page(
        "cross_checks", "Cross-checks",
        blurb="Values that must agree across labels and cameras, such as the serial "
              "on the spec plate matching the one on the traceability tag.",
        visible_when={"constrained": True},
        questions=[
            Question("cross_checks", "Checks", "table", columns=CROSS_COLUMNS),
        ],
    ),
]


def build_recipe(answers: dict[str, Any]) -> Recipe:
    """Turn wizard answers into a ``Recipe``."""
    views: list[ViewSpec] = []
    for row in answers.get("views", []) or []:
        name = str(row.get("view", "") or "").strip()
        if not name:
            continue
        size = row.get("frame_size") or []
        views.append(ViewSpec(
            view=name,
            camera=str(row.get("camera", "") or ""),
            frame_size=[int(v) for v in size[:2]] if len(size) >= 2 and any(size[:2]) else [],
            unexpected_severity=str(row.get("unexpected_severity", "warn")),
        ))

    by_name = {v.view: v for v in views}
    for row in answers.get("bill", []) or []:
        view = by_name.get(str(row.get("view", "")))
        label_id = str(row.get("label_id", "") or "").strip()
        if view is None or not label_id:
            continue
        view.labels.append(LabelRequirement(
            label_id=label_id,
            roi=normalise_roi(row.get("roi")),
            count=row.get("count", 1),
            severity=str(row.get("severity", "fail")),
            roi_tol=float(row.get("roi_tol", 0.02) or 0.0),
            rotation_tol_deg=float(row.get("rotation_tol_deg", 0.0) or 0.0),
            notes=str(row.get("notes", "") or ""),
        ))

    for row in answers.get("forbidden", []) or []:
        view = by_name.get(str(row.get("view", "")))
        label_id = str(row.get("label_id", "") or "").strip()
        if view is not None and label_id and label_id not in view.forbidden:
            view.forbidden.append(label_id)

    checks = [
        CrossCheck(
            type=str(row.get("type", "equal")),
            left=str(row.get("left", "") or ""),
            right=str(row.get("right", "") or ""),
            pattern=str(row.get("pattern", "") or ""),
            severity=str(row.get("severity", "fail")),
        )
        for row in answers.get("cross_checks", []) or []
        if str(row.get("left", "") or "").strip()
    ]

    return Recipe(
        group=str(answers.get("group", "") or "Default").strip(),
        model=str(answers.get("model", "") or "Model").strip(),
        category=str(answers.get("category", DEFAULT_CATEGORY) or DEFAULT_CATEGORY).strip(),
        revision=str(answers.get("revision", "") or ""),
        constrained=bool(answers.get("constrained", True)),
        views=views,
        cross_checks=checks,
        notes=str(answers.get("notes", "") or ""),
    )


def answers_from_recipe(recipe: Recipe) -> dict[str, Any]:
    """Seed the wizard from an existing recipe, so editing reuses the same flow.

    A separate edit dialog would drift from the wizard within two releases;
    round-tripping through the same questions keeps them honest.
    """
    answers: dict[str, Any] = {
        "category": recipe.category,
        "group": recipe.group,
        "model": recipe.model,
        "revision": recipe.revision,
        "constrained": recipe.constrained,
        "notes": recipe.notes,
        "views": [],
        "bill": [],
        "forbidden": [],
        "cross_checks": [],
    }
    for view in recipe.views:
        answers["views"].append({
            "view": view.view,
            "camera": view.camera,
            "frame_size": list(view.frame_size) or [0, 0],
            "unexpected_severity": view.unexpected_severity,
        })
        for req in view.labels:
            answers["bill"].append({
                "view": view.view,
                "label_id": req.label_id,
                "roi": list(req.roi),
                "count": req.count,
                "severity": req.severity,
                "roi_tol": req.roi_tol,
                "rotation_tol_deg": req.rotation_tol_deg,
                "notes": req.notes,
            })
        for label_id in view.forbidden:
            answers["forbidden"].append({"view": view.view, "label_id": label_id})
    for check in recipe.cross_checks:
        answers["cross_checks"].append({
            "type": check.type, "left": check.left, "right": check.right,
            "pattern": check.pattern, "severity": check.severity,
        })
    return answers


def ref_options(answers: dict[str, Any], library=None) -> list[str]:
    """Every ``view.label_id.role`` a cross-check could legally point at.

    Offered as a dropdown rather than a free-text field, because a typo in a
    cross-check reference does not fail loudly -- it silently never matches,
    and an inspection that never runs looks exactly like one that always
    passes.
    """
    options: list[str] = []
    for row in answers.get("bill", []) or []:
        view = str(row.get("view", "") or "")
        label_id = str(row.get("label_id", "") or "")
        if not view or not label_id:
            continue
        label = library.get(label_id) if library is not None else None
        if label is None:
            continue
        for code in label.codes:
            options.append(f"{view}.{label_id}.{code.role}")
        for field_ in label.text_fields:
            if field_.name:
                options.append(f"{view}.{label_id}.{field_.name}")
    return sorted(set(options))


def review_answers(answers: dict[str, Any], library=None) -> list[str]:
    """Warnings for the summary page: what this recipe will and will not catch."""
    recipe = build_recipe(answers)
    warnings = list(validate_recipe(recipe, library))
    if not recipe.constrained:
        return warnings

    for view in recipe.views:
        if not view.labels:
            warnings.append(f"{view.view}: no labels required, so this camera checks nothing.")
        for req in view.labels:
            if not req.roi:
                warnings.append(
                    f"{view.view}/{req.label_id}: no ROI, so this label is accepted "
                    "anywhere in the frame and a misplaced one will pass."
                )
        for i, a in enumerate(view.labels):
            for b in view.labels[i + 1:]:
                if a.roi and b.roi and rois_overlap(a.roi, b.roi):
                    warnings.append(
                        f"{view.view}: the ROIs for '{a.label_id}' and '{b.label_id}' "
                        "overlap, so one label can satisfy the other's placement check."
                    )
        if not view.forbidden:
            warnings.append(
                f"{view.view}: no forbidden labels. A label from a similar model would "
                "be reported only as unexpected ({0}), not as a failure.".format(
                    view.unexpected_severity)
            )
    if library is not None:
        untrained = sorted(i for i in recipe.label_ids() if i not in library)
        if untrained:
            warnings.append(
                "These labels are not in the library yet, so nothing has been trained "
                "to find them: " + ", ".join(untrained)
            )
        for view in recipe.views:
            for req in view.labels:
                label = library.get(req.label_id)
                if label is None:
                    continue
                decoding = [c for c in label.codes
                            if req.code_policy.get(c.role, c.policy) in
                            ("must_decode", "must_match_pattern")]
                if decoding and not recipe.cross_checks:
                    warnings.append(
                        f"{view.view}/{req.label_id} decodes {decoding[0].role} but no "
                        "cross-check uses it. Consider checking it against another label."
                    )
                    break
    return warnings


FLOW = Flow(
    key="new_recipe",
    title="New recipe",
    pages=PAGES,
    build=build_recipe,
    review=lambda answers: review_answers(answers, None),
)
