"""Region randomisation for label areas that change between units.

A date code, a serial, a lot number: the artwork around them holds still and
they do not. That variance is usually harmless -- a region that differs across
the training set carries no signal for the class, so the network stops using
it. The problem is the opposite case. Collect two hundred images in one
afternoon off one lot and the date code is *identical* in every one, which
makes it a perfectly good shortcut for "this is a spec plate" until next
month's date breaks detection for reasons nobody can see.

Two things live here, and they belong together: a check that says whether a
label actually has that problem, and the fix for when it does.

The fix is **cross-grafting**, not synthesis. Fifty images already carry fifty
real date codes; grafting one image's date-code region into another's label
produces training data with a genuine, correctly-lit, correctly-perspectived
code that simply belongs to a different battery. Painting noise or fake text
there instead would put something in the training set that never occurs at
runtime, which is a domain gap built on purpose.

Policy is stdlib and testable bare. OpenCV is imported inside the functions
that actually touch pixels, so importing this module costs nothing.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

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
                "lots, or turn on variable-region copies at export.")
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


def plan_copies(reports: Iterable[RegionReport], requested: int) -> int:
    """How many extra copies to write, given what the check found.

    Zero when nothing is at risk: recombining a region that already varies adds
    training images that teach nothing, and they dilute the real ones.
    """
    requested = max(0, int(requested))
    if not requested:
        return 0
    return requested if any(r.at_risk for r in reports) else 0


# --- pixels ----------------------------------------------------------------

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


def match_levels(patch, reference):
    """Shift a patch onto the reference's label stock.

    The donor came from a different battery under slightly different light, so
    its stock sits at a different grey. Pasted as-is that shows as a faint
    rectangle -- an artifact occurring nowhere at runtime, which a network will
    happily learn as the thing marking a training image.

    Shifted by the difference of medians, and deliberately not rescaled. Median
    rather than mean because the two crops carry different amounts of text, and
    a mean is dragged around by how much ink happens to be in each. Contrast is
    left alone because both were printed by the same process on the same stock:
    rescaling it would only distort the donor's glyphs to match the target's
    text coverage, which is not a thing they should agree on.
    """
    np = _np()
    if patch is None or reference is None or not getattr(reference, "size", 0):
        return patch
    shift = float(np.median(reference.astype(np.float32))) - \
        float(np.median(patch.astype(np.float32)))
    return np.clip(patch.astype(np.float32) + shift, 0, 255).astype(patch.dtype)


# One pixel, not more. Feathering was meant to hide a seam, but once the patch
# is levelled onto the same stock there is no seam to hide -- and a wide ramp
# blends two different values together, which shows up as exactly the ghosted
# rectangle it was supposed to prevent. Measured across 0/1/3 px on real
# grafts: 0 and 1 are clean, 3 is visibly outlined.
DEFAULT_FEATHER = 1


def graft_region(image, box: dict, rect: list[float], patch,
                 feather: int = DEFAULT_FEATHER):
    """Paste a rectified patch into a region of an image.

    The patch is levelled to the target region first, then feathered, so the
    result carries a different value printed in the same ink under the same
    light -- which is the only thing that makes it usable as training data.
    """
    cv2, np = _cv2(), _np()
    quad = region_quad(box, rect)
    if quad is None or patch is None or image is None:
        return image

    existing = rectify_region(image, box, rect,
                              size=(patch.shape[1], patch.shape[0]))
    patch = match_levels(patch, existing)

    height, width = patch.shape[:2]
    src = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    dst = np.array(quad, dtype=np.float32)
    try:
        matrix = cv2.getPerspectiveTransform(src, dst)
    except Exception:
        return image

    shape = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(patch, matrix, shape)
    mask = np.full((height, width), 255, dtype=np.uint8)
    if feather > 0:
        mask[:feather, :] = 0
        mask[-feather:, :] = 0
        mask[:, :feather] = 0
        mask[:, -feather:] = 0
    warped_mask = cv2.warpPerspective(mask, matrix, shape)
    if feather > 0:
        blur = max(3, feather * 2 + 1)
        warped_mask = cv2.GaussianBlur(warped_mask, (blur, blur), 0)

    alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, None]
    blended = image.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha
    return blended.astype(image.dtype)


def shuffle_patch(crop, rng: random.Random, tiles: int = 6):
    """Scramble a crop's own tiles. The fallback when there is no donor.

    Keeps the local texture -- ink on label stock, the same gloss and the same
    lighting -- while destroying the glyphs, so it still reads as something
    printed rather than as noise.
    """
    np = _np()
    if crop is None or not getattr(crop, "size", 0):
        return crop
    height, width = crop.shape[:2]
    step_y = max(1, height // 2)
    step_x = max(1, width // max(1, tiles))
    blocks = []
    for y in range(0, height - step_y + 1, step_y):
        for x in range(0, width - step_x + 1, step_x):
            blocks.append(((y, x), crop[y:y + step_y, x:x + step_x].copy()))
    if len(blocks) < 2:
        return crop
    contents = [b for _pos, b in blocks]
    rng.shuffle(contents)
    out = crop.copy()
    for ((y, x), _original), content in zip(blocks, contents):
        out[y:y + content.shape[0], x:x + content.shape[1]] = content
    return out


# --- dataset-level ---------------------------------------------------------

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
            "shortcut. Capturing across more lots is the real fix; variable-region "
            "copies at export are the stopgap.")
    return "\n".join(lines)


def augmented_variants(entry, label, copies: int, donors: dict, rng: random.Random,
                       *, read=None) -> list:
    """Extra training images for one entry, with its variable regions replaced.

    Donor content comes from other images of the same label -- real values,
    correctly lit, simply belonging to a different battery. Only when no donor
    exists does it fall back to scrambling the region's own tiles.

    The annotation is untouched: same box, same class, same everything. Only
    pixels inside the regions change, which is the whole point.
    """
    read = read or _read
    regions = variable_regions(label)
    boxes = _label_boxes(entry, str(entry.label_id))
    if not regions or not boxes or copies <= 0:
        return []

    base = read(entry.image)
    if base is None:
        return []

    out = []
    for _ in range(int(copies)):
        image = base.copy()
        changed = False
        for box in boxes:
            for name, rect in regions:
                pool = [c for src, c in donors.get(name, []) if src != entry.image]
                if pool:
                    patch = pool[rng.randrange(len(pool))]
                else:
                    own = rectify_region(base, box, rect)
                    patch = shuffle_patch(own, rng) if own is not None else None
                if patch is None:
                    continue
                image = graft_region(image, box, rect, patch)
                changed = True
        if changed:
            out.append(image)
    return out


def donor_pool(entries, label, *, read=None) -> dict:
    """``{region name: [(source image, crop), ...]}`` across a label's images."""
    read = read or _read
    pool: dict[str, list] = {}
    for name, rect in variable_regions(label):
        collected = []
        for entry in entries:
            image = None
            for box in _label_boxes(entry, str(entry.label_id)):
                if image is None:
                    image = read(entry.image)
                if image is None:
                    break
                crop = rectify_region(image, box, rect)
                if crop is not None:
                    collected.append((entry.image, crop))
        pool[name] = collected
    return pool
