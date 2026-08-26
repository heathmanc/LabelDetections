"""Pure helpers for the dataset health dashboard.

Given saved annotation dicts these functions classify each image and tally the
totals the dashboard shows. No Qt, no OpenCV, no file IO, so they stay unit
testable; the UI walks the dataset folders and feeds the dicts in.

The classification itself lives in ``review.annotation_status`` -- there is one
definition of "ready", and both the gate and this dashboard read it, so they
cannot drift.
"""
from __future__ import annotations

from . import review as review_logic

# Per-image statuses that carry drawn boxes.
LABELED_STATUSES = ("ready", "forced", "problem", "needs_review")
# "background" is annotated work carrying no boxes, so it sits outside
# LABELED_STATUSES while still counting toward export_ready().
ALL_STATUSES = LABELED_STATUSES + ("background", "empty", "unlabeled")


def new_tally() -> dict[str, int]:
    tally = {"images": 0, "labeled": 0}
    tally.update({status: 0 for status in ALL_STATUSES})
    return tally


def add_image(tally: dict[str, int], data: dict | None, label_id: str) -> str:
    """Fold one image into a tally and return the status it was counted as."""
    status = review_logic.annotation_status(data, label_id)
    tally["images"] += 1
    tally[status] = tally.get(status, 0) + 1
    if status in LABELED_STATUSES:
        tally["labeled"] += 1
    return status


def merge_tally(into: dict[str, int], other: dict[str, int]) -> None:
    for key, value in other.items():
        into[key] = into.get(key, 0) + value


def export_ready(tally: dict[str, int]) -> int:
    """Images a reviewed-only export would actually include."""
    return sum(tally.get(status, 0) for status in ALL_STATUSES
               if review_logic.export_ready(status))


def readiness(tally: dict[str, int], target: int) -> float:
    """Fraction of the way to a label's training target, clamped to 1.0."""
    if target <= 0:
        return 1.0
    return min(1.0, export_ready(tally) / float(target))


def blockers(tally: dict[str, int]) -> list[str]:
    """What is standing between this dataset and a training run.

    Ordered by how much attention each deserves: stale approvals first, because
    they are the only category that is silently wrong rather than merely
    unfinished.
    """
    out: list[str] = []
    if tally.get("problem"):
        out.append(f"{tally['problem']} approved image(s) no longer carry the label")
    if tally.get("needs_review"):
        out.append(f"{tally['needs_review']} labeled image(s) awaiting review")
    if tally.get("empty"):
        out.append(f"{tally['empty']} image(s) opened but never labeled")
    if tally.get("unlabeled"):
        out.append(f"{tally['unlabeled']} image(s) with no annotation at all")
    return out
