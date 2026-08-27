"""How big each label actually is, in pixels, across the collected data.

The single-stage / two-stage choice, and the crop size if two-stage wins, are
both decided by one number nobody can estimate reliably by eye: how many pixels
wide a label really is in a captured frame. So this measures it from the boxes
already drawn rather than asking anyone to guess.

Two things fall out of that measurement.

**Whether cropping helps at all.** A detector scales the whole frame to its
input size, so a label of L px in a W px frame reaches it as ``L * imgsz / W``.
A classifier gets the crop resized to its own input, so it sees roughly the
crop size. Cropping is a gain only where the second number beats the first --
which is to say, only for labels that are *small in frame*.

**What the crop size has to be.** This is the part a default gets wrong. A
fixed 224 px crop is a large gain for a 300 px label and an outright loss for a
2000 px one, which arrives at the detector already resolved to 500 px and gets
thrown away down to 224. With a wide spread of label sizes the crop has to be
sized from the *largest* label that needs fine discrimination, or the two-stage
pipeline is worse than single-stage at the top of the range while being better
at the bottom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import annotations as ann
from . import geometry as geo
from . import labels as labels_mod

# Classifier inputs are conventionally multiples of 32.
STRIDE = 32
# Above this a "classifier" is costing detector money, and the crop is no
# longer the cheap stage it was supposed to be.
MAX_SENSIBLE_CROP = 640
DEFAULT_IMGSZ = 640


@dataclass
class LabelScale:
    """Measured pixel sizes for one label across its dataset."""
    label_id: str
    long_sides: list[float] = field(default_factory=list)
    frame_sides: list[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.long_sides)

    def _pct(self, values: list[float], q: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))]

    @property
    def median_px(self) -> float:
        return self._pct(self.long_sides, 0.5)

    @property
    def min_px(self) -> float:
        return min(self.long_sides) if self.long_sides else 0.0

    @property
    def max_px(self) -> float:
        return max(self.long_sides) if self.long_sides else 0.0

    def detector_px(self, imgsz: int, which: str = "median") -> float:
        """How many pixels of this label the detector actually receives.

        Paired per image rather than averaged separately: a 400 px label in a
        1000 px frame and a 400 px label in a 4000 px frame reach the detector
        at wildly different sizes, and mixing their frame widths into one mean
        would hide exactly that.
        """
        if not self.long_sides:
            return 0.0
        ratios = [side * imgsz / frame if frame else 0.0
                  for side, frame in zip(self.long_sides, self.frame_sides)]
        if which == "min":
            return min(ratios)
        if which == "max":
            return max(ratios)
        return self._pct(ratios, 0.5)


def measure(entries) -> dict[str, LabelScale]:
    """``{label_id: LabelScale}`` from annotation sidecars.

    Structural classes are skipped: ``battery_side`` is the frame's own size by
    definition and would drag every statistic toward it.
    """
    out: dict[str, LabelScale] = {}
    for entry in entries:
        data = getattr(entry, "annotation", entry) or {}
        frame = max(float(data.get("width", 0) or 0), float(data.get("height", 0) or 0))
        if not frame:
            continue
        for box in ann.boxes(data) or []:
            label_id = str(box.get("label_id", "") or box.get("label", "") or "").strip()
            if not label_id or label_id in labels_mod.STRUCTURAL_CLASSES:
                continue
            quad = ann.box_polygon(box)
            if len(quad) < 4:
                continue
            w, h = geo.quad_size(quad)
            if not max(w, h):
                continue
            scale = out.setdefault(label_id, LabelScale(label_id))
            scale.long_sides.append(max(w, h))
            scale.frame_sides.append(frame)
    return out


def recommend_crop(scales: dict[str, LabelScale], imgsz: int = DEFAULT_IMGSZ) -> int:
    """The smallest crop size at which no label is worse off than single-stage.

    Sized from the label the detector already resolves best, because that is
    the one a crop can harm. Anything smaller trades the top of the range away
    to help the bottom, which is a trade worth making deliberately and not by
    accepting a default.
    """
    worst = max((s.detector_px(imgsz, "median") for s in scales.values()), default=0.0)
    if worst <= 0:
        return 224
    return min(MAX_SENSIBLE_CROP, int(math.ceil(worst / STRIDE) * STRIDE))


def verdict(scale: LabelScale, imgsz: int, crop: int) -> str:
    """What cropping does for one label: gain, loss, or neither."""
    det = scale.detector_px(imgsz, "median")
    cls = min(float(crop), scale.median_px)
    if det <= 0:
        return "no data"
    ratio = cls / det
    if ratio >= 1.15:
        return f"crop helps {ratio:.1f}x"
    if ratio <= 1 / 1.15:
        return f"crop LOSES {1 / ratio:.1f}x"
    return "about the same"


def report(scales: dict[str, LabelScale], imgsz: int = DEFAULT_IMGSZ,
           crop: int | None = None) -> str:
    """A human-readable answer to 'single-stage or two-stage, and at what size'."""
    if not scales:
        return ("No boxes measured yet. Draw and save some labels, then run this "
                "again -- it reads the boxes, not the images.")
    crop = int(crop or recommend_crop(scales, imgsz))

    lines = [f"Measured from {sum(s.count for s in scales.values())} drawn box(es) "
             f"across {len(scales)} label(s).",
             f"Detector input {imgsz} px; classifier crop {crop} px.",
             ""]
    width = max(len(l) for l in scales)
    lines.append(f"{'label'.ljust(width)}  {'boxes':>5}  {'label px (min/med/max)':>24}  "
                 f"{'-> detector':>11}  {'-> crop':>7}  verdict")
    helps = loses = 0
    for label_id in sorted(scales):
        s = scales[label_id]
        det = s.detector_px(imgsz, "median")
        v = verdict(s, imgsz, crop)
        helps += v.startswith("crop helps")
        loses += v.startswith("crop LOSES")
        span = f"{s.min_px:.0f} / {s.median_px:.0f} / {s.max_px:.0f}"
        lines.append(f"{label_id.ljust(width)}  {s.count:>5}  {span:>24}  "
                     f"{det:>11.0f}  {min(float(crop), s.median_px):>7.0f}  {v}")

    lines.append("")
    if loses:
        needed = recommend_crop(scales, imgsz)
        lines.append(
            f"{loses} label(s) would arrive at the classifier with LESS detail than the "
            f"detector already had. A crop of {needed} px fixes that -- below it you are "
            f"trading your biggest labels away to help your smallest.")
    if helps and not loses:
        lines.append(
            f"Every label is at least as well resolved after cropping, and {helps} "
            f"materially better. Two-stage is the stronger choice here.")
    if not helps and not loses:
        lines.append(
            "Cropping changes little either way at this size, so single-stage is the "
            "simpler answer: one model, one pass, nothing to keep in step.")
    lines.append(
        "Detail is only half of it: a classifier is also easier to retrain for a new "
        "label (no boxes to draw, no detector retrain) and its softmax gives a real "
        "'not recognised' threshold. Weigh those too.")
    return "\n".join(lines)
