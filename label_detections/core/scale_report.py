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

# The label width below which identity is limited by resolution rather than by
# the model. A rule of thumb -- the honest number depends on how alike two
# labels are, which is why the report prefers per-region figures wherever
# read-regions exist.
#
# ONE constant, deliberately. There used to be two -- 128 here and 256 in the
# sizing helper -- answering the same question in two places, and on a real
# dataset they reached opposite conclusions inside a single report: "you are
# already there, single-stage" above "a clean two-stage win" below. A threshold
# worth arguing about is worth arguing about once.
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

    # Observations only. The recommendation is advise()'s job -- when both
    # concluded independently they contradicted each other in one document.
    if loses:
        needed = recommend_crop(scales, imgsz)
        lines.append(
            f"{loses} label(s) would reach the classifier with LESS detail than the "
            f"detector already had; a {needed} px crop removes that."
            + (f" That is a large classifier, so it is a real cost."
               if needed > COSTLY_CROP else ""))
    elif helps:
        lines.append(
            f"At {crop} px no label loses detail and {helps} gain materially.")

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


# --- what to actually build ------------------------------------------------
#
# Inverting the question -- "what does each approach REQUIRE?" instead of "is
# this approach good?" -- gives a different answer than either option on the
# table, and the numbers are not close.
#
# A deciding region needs roughly 64 px across to be read at all. For a 6% wide
# revision block on a 2000 px plate that means imgsz 2048 single-stage, or a
# 1067 px crop two-stage. For 25% of a 300 px cert mark: imgsz 3277, or a 256 px
# crop. Not only is each expensive, no single setting serves both -- the
# requirements move in opposite directions with label size.
#
# The third option is the one that works, and it was sitting in the schema
# already: crop the READ-REGION itself out of the full-resolution frame. Its
# pixels are never downscaled at all, so a 120 px revision block stays 120 px
# whatever imgsz the detector runs at, and the crop is tiny.
#
# Which leaves a clean division of labour:
#
#   detector (per-label classes)  -> WHICH label, from gross artwork. Needs the
#                                    label at ~128 px, not its fine print.
#   read-region at native res     -> revision, language, codes. The only stage
#                                    that ever resolves those.
#
# A whole-label classifier sits between the two and is worse than both: too
# coarse to read fine print on a large label, and unnecessary for the gross
# identity the detector already has.

IDENTITY_FLOOR_PX = ADEQUATE_PX
# A text region narrower than this cannot be read by anything. Rule of thumb;
# codes have an exact figure via CodeSpec.min_pixels_needed().
REGION_FLOOR_PX = 64


def min_imgsz_for_identity(scales: dict[str, LabelScale],
                           floor: float = IDENTITY_FLOOR_PX) -> int:
    """Smallest detector input at which every label clears the identity floor.

    Sized from the *smallest* label, which is the one that runs out of pixels
    first. This is about telling a warning label from a spec plate, not about
    reading a revision letter -- nothing at detector resolution does that.
    """
    need = 0.0
    for s in scales.values():
        if s.median_px <= 0:
            continue
        frame = s.frame_sides[0] if s.frame_sides else 0.0
        ratios = [f / side for side, f in zip(s.long_sides, s.frame_sides) if side]
        if not ratios:
            continue
        need = max(need, floor * max(ratios))
    if need <= 0:
        return DEFAULT_IMGSZ
    # Uncapped on purpose. The previous version clamped to 2048 and returned a
    # number that read like an answer -- on a 5472 px frame with 300 px labels
    # the honest requirement is 2336, and quietly reporting 2048 would have had
    # someone train a detector that still could not see the label. When the
    # requirement is impractical the caller must be told the requirement, not a
    # comfortable number, and IMPRACTICAL_IMGSZ is what makes that sayable.
    return int(math.ceil(need / STRIDE) * STRIDE)


# Past this, a detector is slow enough and memory-hungry enough that localising
# cheaply and identifying from a crop is the better trade. Not a hard limit --
# a static station with time to spare can go higher.
IMPRACTICAL_IMGSZ = 2048

# Above this many labels, how a new one is ONBOARDED dominates every
# resolution argument. Under one class per label, adding label N+1 means
# drawing boxes on its images and retraining a detector that now has N+1
# classes -- then re-checking the other N did not regress. Under a generic
# detector the detector never changes: it already finds label-shaped things it
# has never seen, so a new label is crops and a classifier retrain, and the
# crops can come from the detector itself rather than from anyone drawing
# boxes. That difference compounds with library size; resolution does not.
MANY_LABELS = 20


def advise(scales: dict[str, LabelScale], library=None,
           imgsz: int = DEFAULT_IMGSZ) -> str:
    """The recommendation, from the measurements rather than from taste.

    Presents a fork where one genuinely exists. Raising the detector's input
    and cropping are both real answers to an under-resolved label, with
    different costs, and picking one silently hides a decision that belongs to
    whoever runs the line.
    """
    if not scales:
        return "Nothing measured yet -- draw and save some boxes first."

    need = min_imgsz_for_identity(scales)
    smallest = min(scales.values(), key=lambda s: s.median_px)
    frame = smallest.frame_sides[0] if smallest.frame_sides else 0.0
    at_now = smallest.detector_px(imgsz, "median")
    crop = recommend_crop(scales, imgsz)
    weak = under_resolved(scales, imgsz)

    lines = ["RECOMMENDATION", ""]

    health = data_health(scales, library)
    if health:
        lines.append("FIRST -- the data, which decides more than any of the below:")
        lines += [f"  * {h}" for h in health]
        lines.append("")

    lines += [
        f"Frame {frame:.0f} px, detector input {imgsz} px -- a "
        f"{frame / max(imgsz, 1):.1f}x reduction before the model sees anything.",
        f"Your smallest label ({smallest.label_id}) is {smallest.median_px:.0f} px "
        f"in frame and reaches the detector at {at_now:.0f} px "
        f"(the floor for identity is about {ADEQUATE_PX:.0f}).",
        "",
    ]

    if not weak:
        lines += [
            f"Every label clears the floor at imgsz {imgsz}. Single-stage: one "
            f"detector, one class per label, one pass, reporting the id the "
            f"recipe is written in.",
            "A crop stage would hand the classifier less than the detector "
            "already has, so it would cost accuracy rather than add it.",
        ]
    elif need > IMPRACTICAL_IMGSZ:
        lines += [
            f"Under-resolved at this input: {', '.join(weak)}.",
            f"One detector doing identity too would need imgsz {need}, which is "
            f"slow and memory-hungry enough not to be worth it.",
            "",
            f"So: detector at imgsz {imgsz} trained to LOCALISE only (one "
            f"generic `label` class), then identity from a "
            f"full-resolution crop at {crop} px. Finding a "
            f"{at_now:.0f} px label is easy; identifying one is not.",
        ]
    else:
        count = len(library.all()) if library is not None else len(scales)
        lines += [
            f"Under-resolved at this input: {', '.join(weak)}. Two ways to fix "
            f"it, and they are a real choice:",
            "",
            f"A) Single-stage at imgsz {need}. One model, one pass, no second "
            f"thing to keep in step. Costs roughly "
            f"{(need / max(imgsz, 1)) ** 2:.1f}x the inference time of {imgsz} "
            f"and more memory to train.",
            "",
            f"B) Two-stage: detector stays at {imgsz} and only localises, "
            f"identity comes from a {crop} px crop of the full-resolution "
            f"frame. Two models to train and keep matched.",
            "",
        ]
        if count >= MANY_LABELS:
            lines += [
                f"B, at {count} labels -- and the reason is onboarding, not "
                f"pixels.",
                f"Under A, label {count + 1} means drawing boxes on its images, "
                f"retraining a detector that now carries {count + 1} classes, "
                f"and re-checking the other {count} did not regress. Under B "
                f"the detector never changes: it already finds label-shaped "
                f"things it has never seen, so a new label is crops plus a "
                f"classifier retrain -- and the crops come from the detector "
                f"itself instead of from anyone drawing boxes.",
                "That cost compounds with every label added. Resolution does "
                "not.",
                "Classifiers also carry hundreds of classes far more happily "
                "than a detection head, which is splitting its capacity "
                "between where and which at every anchor.",
            ]
        else:
            lines += [
                f"At {count} label(s), A is the simpler place to start. B earns "
                f"its complexity past about {MANY_LABELS} labels, where "
                f"retraining the detector for each new one starts to dominate, "
                f"or sooner if A confuses two labels.",
            ]

    lines.append("")
    fine: list[str] = []
    if library is not None:
        for label_id in sorted(scales):
            label = library.get(label_id)
            if label is None:
                continue
            fine += [f"{label_id}:{c.role}" for c in getattr(label, "codes", []) or []
                     if len(getattr(c, "region", []) or []) >= 4]
            fine += [f"{label_id}:{t.name}" for t in getattr(label, "text_fields", []) or []
                     if len(getattr(t, "region", []) or []) >= 4]
    if fine:
        lines += [
            "Check the crop resolves your finest deciding region: "
            + ", ".join(fine[:6]) + (" ..." if len(fine) > 6 else ""),
            "Whatever separates two labels has to survive into whichever stage "
            "makes the call. The per-region numbers below say whether it does.",
        ]
    else:
        lines += [
            "No read-regions defined. Draw one over whatever distinguishes any "
            "two similar labels (Define Regions) -- it is the only way this "
            "report can tell you whether the difference survives, and with two "
            "labels that look nothing alike it may simply not matter yet.",
        ]
    return "\n".join(lines)


# --- a dump of the dataset, for someone else to look at --------------------

def dataset_details(scales: dict[str, LabelScale], library=None,
                    imgsz: int = DEFAULT_IMGSZ, crop: int | None = None,
                    extra: dict | None = None) -> str:
    """Everything measurable about the collected data, as plain text.

    Written to be pasted somewhere and read by someone who does not have the
    machine: raw distributions rather than conclusions, so the reader can
    disagree with the conclusions. Every number in the recommendation is
    derivable from what is here.
    """
    crop = int(crop or recommend_crop(scales, imgsz))
    out: list[str] = ["LABEL DETECTIONS - DATASET DETAILS", ""]
    for key, value in (extra or {}).items():
        out.append(f"{key}: {value}")
    if extra:
        out.append("")

    frames: dict[str, int] = {}
    for scale in scales.values():
        for side in scale.frame_sides:
            frames[f"{side:.0f}"] = frames.get(f"{side:.0f}", 0) + 1
    out += ["Frame long side (px): "
            + ", ".join(f"{k} x{v}" for k, v in sorted(frames.items(), key=lambda i: -i[1])),
            f"Detector input assumed: {imgsz}",
            f"Classifier crop assumed: {crop}",
            ""]

    out.append("PER LABEL")
    out.append(f"{'label':<24} {'boxes':>6} {'min':>7} {'med':>7} {'max':>7} "
               f"{'frac of frame':>14} {'-> detector':>12}")
    for label_id in sorted(scales):
        sc = scales[label_id]
        frac = (sc.median_px / sc.frame_sides[0]) if sc.frame_sides else 0.0
        out.append(f"{label_id:<24} {sc.count:>6} {sc.min_px:>7.0f} "
                   f"{sc.median_px:>7.0f} {sc.max_px:>7.0f} {frac:>13.1%} "
                   f"{sc.detector_px(imgsz, 'median'):>12.0f}")
    out.append("")

    if library is not None:
        out.append("PER LABEL DEFINITION")
        for label_id in sorted(scales):
            label = library.get(label_id)
            if label is None:
                out.append(f"{label_id}: NOT IN LIBRARY (orphaned dataset)")
                continue
            bits = [f"rev={label.revision or '-'}",
                    f"part={label.part_number or '-'}",
                    f"vendor={label.vendor or '-'}",
                    f"variable_data={label.variable_data}",
                    f"target={label.train_target}"]
            if label.confusable_with:
                bits.append("confusable_with=" + ",".join(label.confusable_with))
            out.append(f"{label_id}: " + "  ".join(bits))
            for role, name, rect in label.regions():
                width_px = float(rect[2]) * scales[label_id].median_px
                out.append(f"    {role}:{name} {float(rect[2]):.0%} wide "
                           f"-> {width_px:.0f} px native, "
                           f"{float(rect[2]) * crop:.0f} px in a {crop} crop")
            for code in label.codes:
                needed = code.min_pixels_needed()
                if needed:
                    out.append(f"    code:{code.role} {code.symbology} needs "
                               f"{needed:.0f} px to decode")
        out.append("")

    health = data_health(scales, library)
    out.append("DATA HEALTH")
    out += ([f"  * {h}" for h in health] if health
            else ["  nothing flagged"])
    out.append("")

    out.append("WHAT THE TOOL CONCLUDES")
    out.append("")
    out.append(advise(scales, library, imgsz))
    out.append("")
    out.append("WORKING")
    out.append("")
    out.append(report(scales, imgsz, crop))
    out.append(region_report(scales, library, imgsz, crop))
    return "\n".join(out)


# --- is there enough data to train this at all? ----------------------------
#
# Every number above is about resolution, and none of it matters for a class
# with three images. This is the check that catches the dataset problems
# resolution analysis walks straight past.

# Below this a class is not learned so much as memorised. Not a hard floor --
# a visually distinctive label may do fine on fewer -- but reporting nothing
# until training fails is worse than reporting a rough number now.
THIN_CLASS_IMAGES = 20
# A box covering more of the frame than this is more likely a mis-draw -- the
# battery face drawn as a label -- than a genuinely enormous label.
SUSPICIOUS_FRAME_FRACTION = 0.60


def data_health(scales: dict[str, LabelScale], library=None) -> list[str]:
    """Problems with the collected data itself, worst first.

    Separate from the resolution report because they fail differently: a
    resolution problem makes a model worse, a data problem makes it untrained.
    """
    issues: list[str] = []
    for label_id in sorted(scales):
        sc = scales[label_id]
        target = 0
        if library is not None:
            label = library.get(label_id)
            target = int(getattr(label, "train_target", 0) or 0)

        if sc.count <= 1:
            issues.append(
                f"{label_id}: {sc.count} box. It cannot be both trained and "
                f"validated -- the split puts it on one side, so the class is "
                f"either never learned or never checked. Whichever way it "
                f"falls, the metrics will not mention it.")
        elif sc.count < THIN_CLASS_IMAGES:
            issues.append(
                f"{label_id}: {sc.count} boxes, under the ~{THIN_CLASS_IMAGES} "
                f"where a class starts being learned rather than memorised.")
        elif target and sc.count < target:
            issues.append(f"{label_id}: {sc.count} of {target} toward its target.")

        if sc.frame_sides:
            frac = sc.median_px / sc.frame_sides[0]
            if frac > SUSPICIOUS_FRAME_FRACTION:
                issues.append(
                    f"{label_id}: covers {frac:.0%} of the frame. Worth opening "
                    f"one to check it is the label and not the battery face -- "
                    f"a face drawn as a label trains the detector to fire on "
                    f"every battery.")

    if library is not None:
        known = {l.label_id for l in library.all()}
        for label_id in sorted(scales):
            if label_id not in known:
                issues.insert(0, (
                    f"{label_id}: has images on disk but no library row -- deleted "
                    f"or renamed. It is excluded from export, so it will not become "
                    f"a class, but the folder is still there. Delete it, or re-add "
                    f"the id if the removal was a mistake."))

    counts = [s.count for s in scales.values()]
    if len(counts) > 1 and max(counts) >= 10 * max(min(counts), 1):
        issues.append(
            f"Class balance: {max(counts)} boxes for the largest class against "
            f"{min(counts)} for the smallest. At that ratio the small class "
            f"contributes almost nothing to the loss and the model can score "
            f"well while never predicting it.")
    return issues
