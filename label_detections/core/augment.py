"""Region randomisation for label areas that change between units.

A date code, a serial, a lot number: the artwork around them holds still and
they do not. That variance is usually harmless -- a region that differs across
the training set carries no signal for the class, so the network stops using
it. The problem is the opposite case. Collect two hundred images in one
afternoon off one lot and the date code is *identical* in every one, which
makes it a perfectly good shortcut for "this is a spec plate" until next
month's date breaks detection for reasons nobody can see.

What lives here is the check that says whether a label has that problem, and
nothing that pretends to fix it.

There used to be a fix: cross-grafting, which took a real date code from one
image and put it into another's label. It was careful -- real pixels, real
lighting, real perspective, and it only fired where the check measured a region
as genuinely constant. It was still a stopgap for a data-collection shortcut,
and the code said so in as many words. Two hundred images off one lot have one
date code however they are recombined; the number of real date codes in the
training set does not go up. Photographing units from a second lot is not a
harder fix, it is the only one that adds information.

So the check reports, and what it asks for is more data rather than more
copies of the data.

Policy is stdlib and testable bare. OpenCV is imported inside the functions
that actually touch pixels, so importing this module costs nothing.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import geometry as geo

# The score is how much a region changes between images relative to how much
# structure it has at all -- cross-image difference over within-image contrast.
# A ratio rather than a grey-level count on purpose: an absolute threshold
# depends on how much of the region the text happens to cover, and measured
# against real data it ranked a well-varied region *below* a constant one.
#
# Measured against synthetic plates: identical content, brightness-only
# differences and blank regions all score exactly 0; any genuine difference in
# printed content scores 0.066 or above, stable whether there are 2 images or
# 20. Camera noise on a lit fixture lands around 0.05, which is what the
# threshold has to clear.
NEAR_IDENTICAL = 0.06
# Above this there is enough variation that the network has nothing to latch
# on to and randomising the region buys nothing.
CLEARLY_VARIED = 0.15

# Crops are rectified to this before being compared, so images taken at
# different distances and angles are still measured against each other.
COMPARE_SIZE = (192, 96)

DEFAULT_COPIES = 2


# --- policy ----------------------------------------------------------------

def variable_regions(label_def) -> list[tuple[str, list[float]]]:
    """``(name, rect)`` for every area of a label expected to change per unit.

    Text fields are the explicit statement of "this changes"; on a label marked
    variable, everything outside the anchor changes too, but only the drawn
    fields are precise enough to graft.
    """
    out: list[tuple[str, list[float]]] = []
    for field in getattr(label_def, "text_fields", []) or []:
        rect = list(getattr(field, "region", []) or [])
        if len(rect) >= 4 and rect[2] > 0 and rect[3] > 0:
            out.append((str(getattr(field, "name", "") or "text"), rect))
    return out


def needs_randomising(score: float) -> bool:
    """True when a region is constant enough to become a shortcut."""
    return float(score) < NEAR_IDENTICAL


def variance_verdict(name: str, score: float, images: int) -> str:
    """One line an operator can act on, rather than a number to interpret."""
    if images < 2:
        return (f"'{name}': only {images} image(s) -- not enough to tell whether "
                "it varies.")
    if score < NEAR_IDENTICAL:
        return (f"'{name}': looks the same in all {images} images "
                f"(variation {score:.2f}). Check whether they all came from one "
                "lot -- if so the model can use it as a shortcut for the class, "
                "and a new value would then break detection. Capture across more "
                "lots.")
    if score < CLEARLY_VARIED:
        return (f"'{name}': varies a little across {images} images "
                f"(variation {score:.2f}). Worth more spread if these came from "
                "one session.")
    return (f"'{name}': varies across {images} images (variation {score:.2f}) -- "
            "nothing to do.")


@dataclass
class RegionReport:
    label_id: str
    name: str
    rect: list[float]
    score: float
    images: int

    @property
    def at_risk(self) -> bool:
        return self.images >= 2 and needs_randomising(self.score)

    def text(self) -> str:
        return variance_verdict(self.name, self.score, self.images)


def _cv2():
    import cv2  # imported here so the policy above stays importable bare
    return cv2


def _np():
    import numpy as np
    return np


def region_quad(box: dict, rect: list[float]) -> list[list[float]] | None:
    """Where a label-space region lands on a labeled box, in image pixels."""
    points = box.get("points") or box.get("obb") or []
    if len(points) < 4:
        return None
    quad = [[float(x), float(y)] for x, y in points[:4]]
    return geo.place_unit_rect(quad, rect)


def rectify_region(image, box: dict, rect: list[float], size=COMPARE_SIZE):
    """Pull a region out of an image, deskewed to a fixed size.

    Rectifying is what makes two crops comparable and graftable at all: the same
    region photographed at a different angle or distance comes out the same
    shape, so alignment falls out of the geometry rather than needing a search.
    """
    cv2, np = _cv2(), _np()
    quad = region_quad(box, rect)
    if quad is None or image is None:
        return None
    width, height = int(size[0]), int(size[1])
    src = np.array(quad, dtype=np.float32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    try:
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image, matrix, (width, height))
    except Exception:
        return None


# Comparing every pair is quadratic and pointless past a few dozen; the mean
# stops moving long before then.
MAX_PAIRS = 60


def region_variance(crops: list) -> float:
    """How much a region's content changes between images, 0 upward.

    Cross-image difference divided by within-image contrast, so the answer does
    not depend on how much of the region the printing happens to cover. Each
    crop has its own mean removed first: two shots of the same date code under
    different light differ in brightness everywhere, and counting that as
    variation would call a constant region varied -- exactly backwards.

    A region with no structure at all scores 0. So does a constant one, which is
    the right answer for both: neither gives the network anything to latch on to
    that a new value would then break.
    """
    import itertools

    np = _np()
    cv2 = _cv2()
    usable = [c for c in crops if c is not None and getattr(c, "size", 0)]
    if len(usable) < 2:
        return 0.0

    greys = []
    for crop in usable:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        grey = grey.astype(np.float32)
        greys.append(grey - float(grey.mean()))

    within = float(np.mean([g.std() for g in greys]))
    if within < 1.0:
        # Nothing printed there to vary.
        return 0.0

    pairs = list(itertools.combinations(range(len(greys)), 2))[:MAX_PAIRS]
    cross = float(np.mean([np.abs(greys[i] - greys[j]).mean() for i, j in pairs]))
    return cross / within


def _label_boxes(entry, label_id: str) -> list[dict]:
    return [b for b in (entry.annotation.get("boxes") or [])
            if str(b.get("label_id", "")) == str(label_id)]


def _read(path: str):
    cv2 = _cv2()
    return cv2.imread(str(path))


def scan_entries(entries, library, *, read=None) -> list[RegionReport]:
    """Measure how much each label's variable regions really differ.

    This is the check that decides whether the augmentation is worth turning on
    at all. A region that already varies needs nothing; one that is the same
    picture in every image is a shortcut waiting to be learned.
    """
    read = read or _read
    reports: list[RegionReport] = []
    by_label: dict[str, list] = {}
    for entry in entries:
        by_label.setdefault(str(entry.label_id), []).append(entry)

    for label_id, group in sorted(by_label.items()):
        label = library.get(label_id) if library is not None else None
        if label is None:
            continue
        for name, rect in variable_regions(label):
            crops = []
            for entry in group:
                image = None
                for box in _label_boxes(entry, label_id):
                    if image is None:
                        image = read(entry.image)
                    if image is None:
                        break
                    crop = rectify_region(image, box, rect)
                    if crop is not None:
                        crops.append(crop)
            reports.append(RegionReport(
                label_id=label_id, name=name, rect=list(rect),
                score=region_variance(crops), images=len(crops),
            ))
    return reports


def scan_text(reports: list[RegionReport]) -> str:
    """The check, as something to read rather than a table of numbers."""
    if not reports:
        return "No variable regions are defined, so there is nothing to check."
    lines = ["Variable-region check:"]
    for report in sorted(reports, key=lambda r: (r.label_id, r.name)):
        lines.append(f"  {report.label_id} -- {report.text()}")
    at_risk = [r for r in reports if r.at_risk]
    if at_risk:
        lines.append("")
        lines.append(
            f"{len(at_risk)} region(s) are constant enough to be learned as a "
            "shortcut. The fix is captures from another lot -- recombining the "
            "ones already collected cannot add a date code that was never "
            "photographed.")
    return "\n".join(lines)
