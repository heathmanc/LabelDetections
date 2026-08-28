"""Which way up a label was presented, decided by looking at it.

The detector's oriented box reports an angle, and the angle cannot answer this.
Ultralytics regularises it: ground-truth corners go through ``cv2.minAreaRect``
at training time, which discards which corner the operator drew first, and
predictions come back through ``regularize_rboxes``, which forces the angle
into ``[0, pi/2)`` with the longer side as the width. So the model is never
shown which end of a label is its top and could not report it if it were. Four
corners plus an angle in a quarter-turn range is a statement about tilt, not
about orientation.

Proportion narrows it to two readings -- a landscape label read as portrait is
a quarter turn out, and ``geometry.candidate_turns`` rules that out. The two
that survive differ by a half turn and are the same shape, so nothing
geometric can separate them.

What can separate them is the artwork. Every label is required to have a
reference image, and that image is the label the right way up by definition:
somebody drew its outline. Flattening the detection each way round and scoring
it against that picture answers the question directly, in about a millisecond,
without reading anything -- which matters, because the print is exactly what is
unreadable on the frames where this goes wrong.

Reading the label is still the better evidence where it works. This is what
answers when it does not.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo
from . import reference as reference_logic
from .imageio import rectify_quad

# Longest side of the comparison. Orientation is a question about layout --
# which end is dark, where the block of print sits -- and layout survives being
# shrunk to a thumbnail. Small also means the four warps cost nothing.
SIZE = 128

# How much better the winner has to be than the runner-up before it is
# believed. A label whose two ends look alike scores nearly the same both ways,
# and a coin toss dressed as a measurement is worse than admitting it.
MIN_MARGIN = 0.05

# And how well the winner has to match at all. Below this the crop is not this
# label -- a bad box, a different part, motion blur -- and its best-of-four
# means nothing.
MIN_SCORE = 0.15

_cache: dict[str, tuple[float, np.ndarray]] = {}


def load(label) -> np.ndarray | None:
    """A label's reference artwork as pixels, or None.

    Cached against the file's timestamp: this is asked once per box per frame,
    and re-decoding the same PNG each time would be the expensive part of an
    otherwise cheap check. A replaced reference has a new timestamp, so the
    cache cannot serve artwork the label no longer has.
    """
    path = reference_logic.reference_path(label)
    if not path:
        return None
    try:
        stamp = Path(path).stat().st_mtime
    except OSError:
        return None
    hit = _cache.get(path)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    image = cv2.imread(path)
    if image is None or image.size == 0:
        return None
    _cache[path] = (stamp, image)
    return image


def _sub(image: np.ndarray, rect) -> np.ndarray:
    """The part of a flattened label a region names, as fractions of it."""
    if len(rect or ()) < 4:
        return image
    h, w = image.shape[:2]
    x, y, rw, rh = (float(v) for v in rect[:4])
    x0, y0 = max(0, int(x * w)), max(0, int(y * h))
    x1, y1 = min(w, int((x + rw) * w)), min(h, int((y + rh) * h))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return image
    return image[y0:y1, x0:x1]


def _prepared(image: np.ndarray, size: tuple[int, int]) -> np.ndarray | None:
    """Greyscale, at a common size, as floats ready to correlate."""
    if image is None or getattr(image, "size", 0) == 0:
        return None
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(grey, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Zero-mean normalised cross-correlation of two prepared images, -1 to 1.

    Zero-mean and unit-norm so a brighter or more contrasty capture scores the
    same as the artwork it matches: the question is where the light and dark
    parts of the label are, not how bright the room was.
    """
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    x = a - a.mean()
    y = b - b.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x.ravel(), y.ravel()) / denom) if denom > 1e-6 else 0.0


def _target_size(artwork: np.ndarray) -> tuple[int, int]:
    h, w = artwork.shape[:2]
    longest = max(w, h) or 1
    scale = SIZE / float(longest)
    return (max(8, int(round(w * scale))), max(8, int(round(h * scale))))


def score_turns(frame, quad, artwork, *, turns=(0, 1, 2, 3),
                region=()) -> dict[int, float]:
    """How well each quarter turn of ``quad`` matches the artwork.

    ``region`` scores a part of the label rather than all of it -- the anchor,
    for a label carrying a serial or a date that is different on every unit and
    would otherwise count against every reading equally.
    """
    if frame is None or artwork is None or quad is None or len(quad) < 4:
        return {}
    size = _target_size(artwork)
    want = _prepared(_sub(artwork, region), size)
    if want is None:
        return {}

    settled = geo.order_quad(quad)
    out: dict[int, float] = {}
    for turn in turns:
        crop = rectify_quad(frame, geo.turn_quad(settled, turn),
                            max_side=SIZE * 3, orient=False)
        if crop is None or crop.size == 0:
            continue
        out[int(turn)] = similarity(_prepared(_sub(crop, region), size), want)
    return out


def best_turn(frame, quad, artwork, *, aspect: float = 0.0,
              region=()) -> tuple[int, float, float]:
    """``(turn, score, margin)`` for the reading that matches the artwork best.

    ``margin`` is how far clear of the runner-up it came, which is the number
    worth acting on: a high score both ways up means a label whose two ends
    look alike, and picking the winner of that is guessing.

    ``(0, 0.0, 0.0)`` when there is nothing to compare, which reads as "no
    opinion" against the thresholds and leaves the caller on its fallback.
    """
    turns = geo.candidate_turns(quad, aspect)
    scores = score_turns(frame, quad, artwork, turns=turns, region=region)
    if not scores:
        return (0, 0.0, 0.0)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, top = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else -1.0
    return (best, top, top - runner)


def settled(score: float, margin: float) -> bool:
    """Is a match good enough, and clear enough, to act on?"""
    return score >= MIN_SCORE and margin >= MIN_MARGIN


def note(turn: int, score: float, margin: float) -> str:
    """One line about what the comparison decided, for a diagnostic."""
    if not score and not margin:
        return "no reference artwork to compare against"
    quarter = {0: "as detected", 1: "a quarter turn", 2: "upside down",
               3: "three quarters"}.get(int(turn) % 4, "")
    if not settled(score, margin):
        return (f"artwork match is not conclusive ({score:.2f}, only "
                f"{margin:.2f} clear of the next reading)")
    return f"artwork says {quarter} ({score:.2f}, {margin:.2f} clear)"
