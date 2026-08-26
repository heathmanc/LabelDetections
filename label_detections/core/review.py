"""Operator review markers and the labeling tool's per-image gate.

The marker discipline is carried over wholesale from BungVision Label Studio,
because it was learned the hard way: **only a marker this tool wrote counts as
reviewed**. Runtime and third-party JSON is full of generic ``reviewed: true``
and ``review_status: ok`` fields that mean something else entirely, and
letting those into a training export is how unchecked data ends up teaching
the model.

The gate is per image and per label, because a dataset here is one label's
images. An image is export-ready when it carries a properly annotated instance
of the label it was collected for, and an operator has said so. Nothing in this
module reads a recipe -- recipes belong to the vision program, and a labeling
session should never be blocked by one.
"""
from __future__ import annotations

import time
from typing import Any

from . import annotations as ann
from . import geometry as geo
from ..version import APP_TITLE, REVIEW_SOURCE

_MARKER_KEYS = ("source", "tool", "review_source", "reviewed_by", "reviewer", "app")

# Why an operator knowingly kept a unit that does not match its recipe. Required
# on force-review so the defect library is queryable: "show me every
# wrong_revision example" is the question that builds a real FAIL test set, and
# it is unanswerable if every forced image just says "mismatch".
DEFECT_REASONS = [
    "missing_label",
    "wrong_label",
    "wrong_revision",
    "rotated",
    "misplaced",
    "torn_or_wrinkled",
    "smeared_code",
    "unreadable_code",
    "duplicate_label",
    "other",
]


def is_review_marker(review: dict | None) -> bool:
    """True only for a review stamp this application wrote."""
    if not isinstance(review, dict) or not bool(review.get("reviewed", False)):
        return False
    text = " ".join(str(review.get(k, "")) for k in _MARKER_KEYS).lower()
    return REVIEW_SOURCE in text or "labelvision studio" in text


def annotation_reviewed(data: dict | None) -> bool:
    """True when an operator approved this image inside this tool.

    Generic imported fields deliberately do not qualify. This is the single
    rule that keeps unvetted data out of training, and it is worth its
    strictness.
    """
    if not isinstance(data, dict):
        return False
    if is_review_marker(data.get("review")):
        return True
    if bool(data.get("reviewed", False)):
        return is_review_marker({
            "reviewed": True,
            "source": data.get("review_source") or data.get("source"),
            "tool": data.get("review_tool") or data.get("tool"),
            "reviewed_by": data.get("reviewed_by"),
        })
    return False


def annotation_force_reviewed(data: dict | None) -> bool:
    """True when an image was kept on purpose despite failing its recipe."""
    if not annotation_reviewed(data):
        return False
    review = data.get("review")
    if isinstance(review, dict):
        if bool(review.get("forced_review", False)):
            return True
        if str(review.get("review_status", "")).lower() == "forced_reviewed":
            return True
    return bool(data.get("forced_review", False))


def is_background_annotation(data: dict | None) -> bool:
    """True for an image deliberately marked as holding nothing to detect.

    The flag is explicit rather than inferred from "no boxes": an annotation
    with no boxes is usually unfinished work, and exporting that as a negative
    teaches the model to ignore real labels.
    """
    if not isinstance(data, dict) or data.get("boxes"):
        return False
    return bool(data.get("background", False))


def make_review_record(reason: str = "operator_review", *, force: bool = False,
                       defect_reason: str = "", verdict: str = "",
                       findings: list[str] | None = None) -> dict[str, Any]:
    """The stamp written into a sidecar when an operator approves it."""
    record: dict[str, Any] = {
        "reviewed": True,
        "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reviewed_by": APP_TITLE,
        "source": REVIEW_SOURCE,
        "tool": APP_TITLE.split(" v")[0],
        "reason": reason,
    }
    if force:
        record.update({
            "forced_review": True,
            "review_status": "forced_reviewed",
            # Recorded verbatim so the defect library can be filtered on it.
            "defect_reason": defect_reason or "other",
            "recipe_verdict": verdict,
            "findings": list(findings or []),
            "warning": (
                "Operator kept this unit even though it does not match its recipe. "
                "It is a deliberate defect example."
            ),
        })
    return record


def clear_review(data: dict) -> dict:
    """Strip review markers after an edit that changed what was reviewed.

    Without this an operator can review a unit, edit its labels into a
    mismatch, save, and leave a stale approval behind -- the exact bug that
    cost BungVision a release. Editing is not approving.
    """
    for key in ("review", "reviewed", "review_source", "review_tool",
                "review_status", "reviewed_by", "forced_review"):
        data.pop(key, None)
    return data


def stamp(data: dict, record: dict[str, Any]) -> dict:
    """Attach a review record, keeping the flat mirror fields other tools read."""
    data["review"] = record
    data["reviewed"] = True
    data["review_source"] = record.get("source", REVIEW_SOURCE)
    data["review_tool"] = record.get("tool", "")
    data["review_status"] = record.get("review_status", "reviewed")
    return data


# --- the gate --------------------------------------------------------------

# Per-image states, in the order a dataset health dashboard shows them.
IMAGE_STATUSES = ("unlabeled", "background", "empty", "needs_review",
                  "problem", "forced", "ready")


def annotation_status(data: dict | None, label_id: str) -> str:
    """Classify one training image in ``label_id``'s dataset.

    unlabeled    no sidecar at all
    background   deliberately marked as a negative for this label
    empty        a sidecar exists but nothing was drawn yet
    needs_review annotated, not yet approved
    forced       approved as a deliberate defect example
    problem      approved, but carries no instance of the label it was collected
                 for -- a stale approval left behind by a later edit
    ready        approved and usable
    """
    if data is None:
        return "unlabeled"
    if is_background_annotation(data):
        return "background"
    if not ann.boxes(data):
        return "empty"
    if not annotation_reviewed(data):
        return "needs_review"
    if annotation_force_reviewed(data):
        return "forced"
    return "ready" if ann.boxes_for(data, label_id) else "problem"


def export_ready(status: str) -> bool:
    """Statuses whose images may enter a training export.

    ``forced`` is included on purpose -- a torn label or a smeared code is some
    of the most valuable training data there is. ``problem`` is excluded: it
    means an approval that no longer matches the labels underneath it.
    """
    return status in ("ready", "forced", "background")


def validate_boxes(data: dict | None, label_id: str, *, min_side: float = 3.0,
                   overlap_iou: float = 0.6) -> list[str]:
    """Label-quality problems the review gate cannot see. Empty means clean.

    Advisory, never blocking: these are the geometry mistakes that quietly
    poison training -- a degenerate box, a box hanging off the edge of the
    image, the same label drawn twice -- and an operator who cannot save
    because of a warning will stop using the tool.
    """
    issues: list[str] = []
    if data is None:
        return ["No annotation for this image."]

    width = int(data.get("width", 0) or 0)
    height = int(data.get("height", 0) or 0)
    boxes = ann.boxes(data)

    for index, box in enumerate(boxes, start=1):
        name = str(box.get("label_id") or box.get("label") or "box")
        bx, by, bw, bh = geo.quad_bounds(ann.box_polygon(box))
        if bw < min_side or bh < min_side:
            issues.append(f"{name} #{index}: degenerate or too small ({bw:.0f}x{bh:.0f} px)")
        if width and height:
            if bx < -1 or by < -1 or bx + bw > width + 1 or by + bh > height + 1:
                issues.append(f"{name} #{index}: extends outside the image")
        for region in ann.regions(box):
            rx, ry, rw, rh = geo.quad_bounds(region.get("points") or [])
            if rw < 1 or rh < 1:
                issues.append(f"{name} #{index}: a sub-region is degenerate")
            elif not _mostly_inside((rx, ry, rw, rh), (bx, by, bw, bh)):
                issues.append(
                    f"{name} #{index}: a {region.get('role', 'sub')} region sits outside "
                    "its label -- the artwork geometry may be wrong"
                )

    same = ann.boxes_for(data, label_id)
    bounds = [geo.quad_bounds(ann.box_polygon(b)) for b in same]
    for i in range(len(same)):
        for j in range(i + 1, len(same)):
            if geo.rect_iou(bounds[i], bounds[j]) > overlap_iou:
                issues.append(f"{label_id} #{i + 1} and #{j + 1} overlap heavily (duplicate?)")

    if not same and not is_background_annotation(data):
        issues.append(
            f"No '{label_id}' is labeled, but this image is in that label's dataset. "
            "Draw it, or mark the image a background."
        )
    return issues


def _mostly_inside(inner: tuple[float, float, float, float],
                   outer: tuple[float, float, float, float]) -> bool:
    """Is ``inner`` substantially within ``outer``? Tolerant of a few px of slop."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    pad = 2.0
    return (ix >= ox - pad and iy >= oy - pad
            and ix + iw <= ox + ow + pad and iy + ih <= oy + oh + pad)


def dataset_counts(statuses: list[str]) -> dict[str, int]:
    """Tally of image statuses for one label's dataset."""
    counts = {status: 0 for status in IMAGE_STATUSES}
    for status in statuses:
        counts[status] = counts.get(status, 0) + 1
    counts["export_ready"] = sum(n for s, n in counts.items() if export_ready(s))
    return counts


def dataset_summary(label_id: str, statuses: list[str], *, want: int = 150) -> str:
    """One-line readiness summary for a label's dataset.

    ``want`` is a working default, not a law: a label with distinctive artwork
    on a stable line trains on far fewer, and a foil label in changing light
    needs far more.
    """
    counts = dataset_counts(statuses)
    ready = counts["export_ready"]
    parts = [f"{label_id}: {ready} export-ready of {len(statuses)} images"]
    pending = counts["needs_review"] + counts["empty"] + counts["unlabeled"]
    if pending:
        parts.append(f"{pending} still to label or review")
    if counts["problem"]:
        parts.append(f"{counts['problem']} with stale approvals")
    if counts["forced"]:
        parts.append(f"{counts['forced']} defect examples")
    if ready < want:
        parts.append(f"below the {want}-image working target")
    return "; ".join(parts)
