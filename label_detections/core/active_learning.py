"""Active-learning prioritisation: which unlabeled image to work on next.

Pure scoring (no Qt, no OpenCV) for ranking a label's unlabeled images by how
much the current model struggles with them. Adapted from BungVision's
recipe-count version: the question there was "does the detected bung layout
match the recipe", and here there is no recipe -- the question is "does the
model already find this label confidently?".

Images where it does teach almost nothing. Images where it finds nothing, finds
too many, or is unsure are where the next hour of labeling actually moves the
model, so they float to the top.
"""
from __future__ import annotations

from dataclasses import dataclass

# A miss is the strongest signal an image needs a human: the model cannot see
# the label at all here, which is exactly the case more data fixes.
MISS_PENALTY = 10.0
# Extra detections of the label's family are near-misses -- a look-alike, or one
# label split into two boxes. Worth less than a miss, more than low confidence.
EXTRA_WEIGHT = 1.5
LOW_CONF_WEIGHT = 2.0


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def disagreement_score(
    found: int,
    expected: int = 1,
    total_detections: int = 0,
    avg_conf: float | None = None,
) -> float:
    """Higher = the model handles this image worse = label it sooner.

    ``found`` is how many detections matched the active label's family and
    ``expected`` how many the operator expects on a typical image (almost
    always one). ``total_detections`` covers the case where the model fires on
    everything, which is its own kind of wrong.
    """
    score = 0.0
    if found <= 0:
        score += MISS_PENALTY
    else:
        score += EXTRA_WEIGHT * max(0, int(found) - int(expected))
    # A frame full of detections when one label was expected is noisy in a way
    # a plain miss is not, so it earns a smaller, separate nudge.
    score += 0.25 * max(0, int(total_detections) - max(1, int(expected)))
    if avg_conf is not None:
        score += LOW_CONF_WEIGHT * (1.0 - _clamp01(float(avg_conf)))
    return score


@dataclass
class QueueItem:
    key: str
    score: float


def rank_items(items: list[QueueItem]) -> list[QueueItem]:
    """Sort highest-disagreement first; ties keep a stable key order."""
    return sorted(items, key=lambda it: (-float(it.score), str(it.key)))
