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
# Above this a "classifier" costs detector money and the crop stops being the
# cheap second stage it was sold as. It is a warning threshold, not a silent
# ceiling: the recommendation goes above it when the data demands and says so,
# because a capped number that reads like an answer is worse than a large one.
COSTLY_CROP = 512
# A hard ceiling only to stop a pathological dataset asking for something absurd.
MAX_CROP = 1024
DEFAULT_IMGSZ = 640

# Roughly the label width at which fine-grained identity stops being limited by
# resolution and starts being limited by the model. A rule of thumb, and the
# reason the report prefers per-region numbers wherever read-regions exist.
ADEQUATE_PX = 256


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
    return min(MAX_CROP, int(math.ceil(worst / STRIDE) * STRIDE))


def under_resolved(scales: dict[str, LabelScale], imgsz: int,
                   adequate: float = ADEQUATE_PX) -> list[str]:
    """Labels the detector alone does not resolve well enough for identity.

    These are the labels a crop stage is actually for. A label the detector
    already receives at 600 px does not need one, and saying so matters: sizing
    the crop to protect labels that never needed protecting is what pushes the
    classifier up into detector-sized inputs.
    """
    return sorted(k for k, s in scales.items()
                  if 0 < s.detector_px(imgsz, "median") < adequate)


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
    weak = under_resolved(scales, imgsz)
    if weak:
        lines.append(
            f"Under-resolved by the detector alone (< {ADEQUATE_PX:.0f} px of label): "
            + ", ".join(weak))
        lines.append("These are the labels a crop stage is actually for.")
    else:
        lines.append(
            f"Every label reaches the detector at {ADEQUATE_PX:.0f} px or more. Nothing "
            f"here is resolution-starved, so single-stage is the simpler answer -- one "
            f"model, one pass, nothing to keep in step.")
    lines.append("")

    if loses:
        needed = recommend_crop(scales, imgsz)
        lines.append(
            f"{loses} label(s) would reach the classifier with LESS detail than the "
            f"detector already had. A crop of {needed} px removes that entirely.")
        if needed > COSTLY_CROP:
            lines.append(
                f"But {needed} px is a big classifier -- at that size the 'cheap second "
                f"stage' is no longer cheap. Consider whether your LARGEST labels need "
                f"the crop at all: a label the detector already sees at 600 px is not "
                f"the problem, and sizing the crop to protect it is what forced this "
                f"number up. A smaller crop that serves {', '.join(weak) or 'the small ones'} "
                f"may be the better trade, made knowingly.")
    elif helps:
        lines.append(
            f"No label loses detail at {crop} px, and {helps} gain materially. This is "
            f"a clean two-stage win.")

    lines.append("")
    lines.append(
        "Caveat on all of the above: this measures the whole LABEL. What decides "
        "identity is often one small part of it -- a revision letter, a language "
        "line. Where a label has read-regions defined, the per-region numbers below "
        "are the ones that actually matter.")
    lines.append(
        "Detail is only half of it either way: a classifier is easier to retrain for a "
        "new label (no boxes to draw, no detector retrain) and its softmax gives a real "
        "'not recognised' threshold.")
    return "\n".join(lines)


# --- what the deciding detail gets, rather than the whole label -------------

def region_report(scales: dict[str, LabelScale], library, imgsz: int,
                  crop: int) -> str:
    """Pixels across each read-region, at each stage.

    The label-level numbers above answer "how much of the label does each stage
    see". This answers the question that actually decides identity: how much of
    the part that *differs* does each stage see. A 2000 px label whose only
    distinguishing mark is a 4% wide revision block is carrying 80 px of
    evidence, not 2000, and every conclusion drawn from the label width is off
    by that factor.

    For codes there is a real threshold to check against rather than a feeling:
    CodeSpec.min_pixels_needed() is what the symbology needs to decode.
    """
    if library is None:
        return ""
    rows: list[str] = []
    for label_id in sorted(scales):
        label = library.get(label_id)
        if label is None:
            continue
        regions: list[tuple[str, float, float]] = []
        for i, code in enumerate(getattr(label, "codes", []) or []):
            if len(getattr(code, "region", []) or []) >= 4:
                regions.append((f"code[{code.role}]", float(code.region[2]),
                                code.min_pixels_needed()))
        for field_ in getattr(label, "text_fields", []) or []:
            if len(getattr(field_, "region", []) or []) >= 4:
                regions.append((f"text[{field_.name}]", float(field_.region[2]), 0.0))
        if not regions:
            continue
        scale = scales[label_id]
        det_label = scale.detector_px(imgsz, "median")
        for name, frac, needed in regions:
            det_px = frac * det_label
            crop_px = frac * min(float(crop), scale.median_px)
            note = ""
            if needed:
                note = ("decodes after crop" if crop_px >= needed >= det_px else
                        "decodes at both" if det_px >= needed else
                        f"NEITHER reaches the {needed:.0f} px this symbology needs")
            rows.append(f"  {label_id} {name}: {frac:.0%} of label -> "
                        f"{det_px:.0f} px at detector, {crop_px:.0f} px after crop"
                        + (f"   [{note}]" if note else ""))
    if not rows:
        return ("\nNo read-regions defined yet. Draw them (Define Regions) and this "
                "report can size the crop from the detail that actually decides "
                "identity instead of from the whole label.")
    return "\nRead-regions -- the detail identity actually turns on:\n" + "\n".join(rows)


def full_report(scales: dict[str, LabelScale], library=None,
                imgsz: int = DEFAULT_IMGSZ, crop: int | None = None) -> str:
    crop = int(crop or recommend_crop(scales, imgsz))
    return report(scales, imgsz, crop) + "\n" + region_report(scales, library, imgsz, crop)
