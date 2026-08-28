"""Getting a barcode off a detected label, at the resolution it was printed at.

The deciding half is ``core/codes``. This half finds the pixels and hands them
to a decoder.

Two things matter here and both are about resolution. First, the code is read
from the FULL-RESOLUTION frame, never from the crop stage 2 was given -- that
crop is 320 px of a label so a classifier can see it, and a barcode that
occupies a tenth of the label arrives as 32 px, which is below what any decoder
can resolve. Second, the crop is the code's own region rather than the whole
label, which is both faster (warping costs what its destination costs) and more
reliable (nothing else in frame to misread).

The region comes free. The library stores where the code sits as fractions of
the label, drawn once on the artwork, and ``geometry.place_unit_rect`` maps
that onto whatever quad the detector just produced -- at any distance, any
angle, no calibration. A label with no region drawn falls back to searching the
whole label, which works and costs more.
"""
from __future__ import annotations

from . import codes as logic
from .geometry import oriented, place_unit_rect
from .imageio import rectify_quad

# The whole-label fallback, capped. Only used when no region was drawn: a
# 2000 px warp costs real milliseconds per detection and there may be five in
# frame, so this is the slow road and the region is the fast one.
FALLBACK_MAX_SIDE = 1600

# A little context around the symbol. Decoders want the quiet zone, and a
# region drawn tight to the printed bars does not include one. This is the
# floor, used when the print spec was never entered; a label that declares its
# quiet zone gets the real number instead. See region_margin.
REGION_MARGIN = 0.25

# However wrong a typed number is, the crop stays a crop. A quiet zone entered
# in the wrong units would otherwise swallow the whole label and read whatever
# else is printed on it.
MAX_REGION_MARGIN = 1.5

_BACKEND = None
_REASON = ""


def backend():
    """The decoder, imported once. ``(module, reason)`` -- module is None on failure."""
    global _BACKEND, _REASON
    if _BACKEND is None and not _REASON:
        try:
            import zxingcpp

            _BACKEND = zxingcpp
        except Exception as exc:
            _REASON = (f"zxing-cpp is not installed, so codes cannot be read "
                       f"({exc}). pip install zxing-cpp")
    return _BACKEND, _REASON


def available() -> tuple[bool, str]:
    module, reason = backend()
    return module is not None, reason


def _read_once(module, image, **options) -> list[logic.Read]:
    """One pass of the decoder, with everything that can go wrong swallowed."""
    try:
        results = module.read_barcodes(image, **options)
    except Exception:
        return []
    out = []
    for result in results or []:
        text = str(getattr(result, "text", "") or "")
        if not text:
            continue
        # zxing reports validity per symbol; a checksum failure is not a read.
        if getattr(result, "valid", True) is False:
            continue
        out.append(logic.Read(text=text,
                              symbology=str(getattr(result, "format", "")),
                              box=_position_of(result)))
    return out


def _position_of(result) -> list[float]:
    """``[x0, y0, x1, y1]`` of a decoded symbol, or ``[]``.

    The decoder knows where it found the code. That turns "is the region in the
    right place" from an argument into a measurement -- and one the operator can
    act on, because the answer is the numbers to type into the region.
    """
    pos = getattr(result, "position", None)
    if pos is None:
        return []
    xs, ys = [], []
    for corner in ("top_left", "top_right", "bottom_right", "bottom_left"):
        point = getattr(pos, corner, None)
        try:
            xs.append(float(point.x))
            ys.append(float(point.y))
        except Exception:
            return []
    return [min(xs), min(ys), max(xs), max(ys)] if xs else []


def _ladder(module, image):
    """The images and options to try, cheapest and most likely first.

    Measured on rendered UPC-A symbols put through what this pipeline does to
    them -- rectified from an angle, blurred by that resampling, and carrying
    the colour fringing a Bayer sensor leaves on fine vertical bars. Each rung
    earns its place by decoding cases the ones above it cannot:

      * **as taken** reads every clean code, so a good part costs one call.
      * **greyscale** is the big one. Handing zxing a BGR array lets it do its
        own luminance conversion, and chroma noise across the bars survives
        that; converting first, with proper luma weights, does not. Fringing
        that fails outright on BGR reads on grey.
      * **a fixed threshold** suits a cropped code. The default local-average
        binarizer is built to find a code somewhere in a large scene with
        uneven light, and on a crop that is almost entirely one code it adapts
        to the bars themselves.
      * **upscaling** gives the scanline more samples per bar. It invents no
        detail, and it recovers blurred codes the binarizers miss.
      * **one colour channel** is the last resort, for fringing bad enough that
        even a luma mix carries it. A single channel never had the artefact.

    Nothing here loosens what counts as a read: every rung still has to satisfy
    the symbology's own checksum.
    """
    import cv2

    fixed = {}
    try:
        fixed = {"binarizer": module.Binarizer.FixedThreshold}
    except Exception:
        pass

    yield "as taken", image, {}
    if getattr(image, "ndim", 2) != 3:
        yield "fixed threshold", image, fixed
        return

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    yield "greyscale, fixed threshold", grey, fixed
    yield "greyscale", grey, {}
    try:
        bigger = cv2.resize(grey, None, fx=2.0, fy=2.0,
                            interpolation=cv2.INTER_CUBIC)
        yield "greyscale, upscaled", bigger, fixed
    except Exception:
        pass
    # Downscaling, which sounds backwards and is not. INTER_AREA averages, so
    # it low-passes away sensor noise and the colour moire that sits at the bar
    # pitch, while a code only needs 2-3 px per module to be read. Observed on
    # a real label: the whole-label crop decoded at 2.5 px per module having
    # been shrunk to fit a cap, while the native-resolution crop of the same
    # bars at 3.3 px per module did not.
    for factor in (0.6, 0.4):
        try:
            smaller = cv2.resize(grey, None, fx=factor, fy=factor,
                                 interpolation=cv2.INTER_AREA)
            if min(smaller.shape[:2]) >= 12:
                yield f"greyscale, downscaled x{factor:g}", smaller, fixed
                yield f"downscaled x{factor:g}", smaller, {}
        except Exception:
            pass
    yield "red channel", image[:, :, 2], fixed


def decode(image, detail: bool = False):
    """Every symbol in one image, trying harder as each attempt comes back empty.

    See ``_ladder`` for what is tried and why each rung is there. Cost is paid
    only on failure: a code that reads on the first attempt costs one call.

    ``detail`` returns ``(reads, how)`` so a diagnostic can say which rung it
    took. That is worth surfacing -- a code that only reads on the last one is
    readable today and the first thing to fail when the print, the focus or the
    lighting drifts.
    """
    module, _reason = backend()
    if module is None or image is None or getattr(image, "size", 0) == 0:
        return ([], "") if detail else []

    try:
        rungs = list(_ladder(module, image))
    except Exception:
        rungs = [("as taken", image, {})]

    for how, candidate, options in rungs:
        reads = _read_once(module, candidate, **options)
        if reads:
            return (reads, how) if detail else reads
    return ([], "") if detail else []


class Spec:
    """A frozen copy of one CodeSpec: the fields the runtime reads, and no more.

    The worker runs off the GUI thread and the library is edited on it, so what
    crosses between them is a snapshot rather than a live reference.
    """

    __slots__ = ("role", "symbology", "policy", "pattern", "region",
                 "quiet_zone_mm", "code_width_mm", "rotation_policy")

    def __init__(self, role="", symbology="", policy="", pattern="", region=(),
                 quiet_zone_mm=0.0, code_width_mm=0.0):
        self.role = role
        self.symbology = symbology
        self.policy = policy
        self.pattern = pattern
        self.region = list(region)
        self.quiet_zone_mm = float(quiet_zone_mm or 0.0)
        self.code_width_mm = float(code_width_mm or 0.0)
        self.rotation_policy = ""


def specs_from(label) -> list[Spec]:
    """One label's code specs, frozen."""
    return [
        Spec(role=str(getattr(spec, "role", "") or ""),
             symbology=str(getattr(spec, "symbology", "") or ""),
             policy=str(getattr(spec, "policy", "") or ""),
             pattern=str(getattr(spec, "pattern", "") or ""),
             region=[float(v) for v in (getattr(spec, "region", None) or [])],
             quiet_zone_mm=getattr(spec, "quiet_zone_mm", 0.0),
             code_width_mm=getattr(spec, "code_width_mm", 0.0))
        for spec in (getattr(label, "codes", None) or [])
    ]


def region_margin(spec) -> float:
    """How far to grow a drawn region so the decoder gets its quiet zone.

    The drawn box follows the printed bars, because that is what an operator
    can see to drag around. Decoders want the blank margin either side of them,
    and how much is not a matter of taste -- the symbology specifies it, and
    for a small code it can be most of the symbol's own width again. A fixed
    guess is fine for a Code 128 across a battery face and far too tight for an
    8 mm DataMatrix, where cropping inside the quiet zone loses the read.

    So when the label declares its quiet zone and printed width, use them.
    ``_expand`` splits the margin either side of the region, so each side gets
    ``width * margin / 2`` -- which makes the margin twice the quiet zone as a
    fraction of the width.

    Never tighter than the guess it replaces, so a label whose spec says less
    than the old fixed value keeps behaving as it did.
    """
    quiet = float(getattr(spec, "quiet_zone_mm", 0.0) or 0.0)
    width = float(getattr(spec, "code_width_mm", 0.0) or 0.0)
    if quiet <= 0 or width <= 0:
        return REGION_MARGIN
    return min(MAX_REGION_MARGIN, max(REGION_MARGIN, 2.0 * quiet / width))


def _expand(rect: list[float], margin: float) -> list[float]:
    """Grow a unit rect about its centre, clamped inside the label."""
    x, y, w, h = (float(v) for v in rect[:4])
    x -= w * margin / 2.0
    y -= h * margin / 2.0
    w *= 1.0 + margin
    h *= 1.0 + margin
    x, y = max(0.0, x), max(0.0, y)
    return [x, y, min(w, 1.0 - x), min(h, 1.0 - y)]


# Which way up a label was presented cannot be read off four corners: an
# upside-down label produces corners in the same slots as an upright one, so
# every region measured from the top-left lands at the diagonally opposite end.
# The only thing that settles it is reading the label and seeing which reading
# produces the code that belongs there -- so where the library says a label may
# arrive turned over, both are tried and the reads are pooled. A wrong-way-up
# crop almost never decodes, and if it does it still has to satisfy a checksum
# and match an enrolled pattern before it means anything.
FLIPPABLE = ("flip_ok", "any")


def orientations(rotation_policy: str) -> list[bool]:
    """Which readings of a quad to try, in order. False is upright."""
    return [False, True] if str(rotation_policy or "") in FLIPPABLE else [False]


def read_label(frame, quad, specs, rotation_policy: str = "") -> list[logic.Read]:
    """Decode the codes of one detected label out of the full-resolution frame.

    Tries each declared region first and stops as soon as something decodes --
    one good read answers the question, and the remaining warps are work for an
    answer already in hand. Falls back to the whole label only when no region
    decoded, because that is the expensive path and it exists for labels whose
    artwork was never marked up.
    """
    module, _reason = backend()
    if module is None or frame is None or quad is None or len(quad) < 4:
        return []

    wanted = logic.demanded(specs)
    tries = [(flipped, oriented(quad, flipped))
             for flipped in orientations(rotation_policy)]
    for spec in wanted:
        region = list(getattr(spec, "region", None) or [])
        if len(region) < 4:
            continue
        for _flipped, settled in tries:
            placed = place_unit_rect(settled, _expand(region, region_margin(spec)),
                                     orient=False)
            if not placed:
                continue
            # No max_side: this is the one crop in the pipeline that must not
            # be shrunk. Everything else here downsamples for a model; a
            # decoder wants every printed module it can get.
            reads = decode(rectify_quad(frame, placed))
            if reads:
                return reads

    if any(len(list(getattr(s, "region", None) or [])) >= 4 for s in wanted):
        # Regions were drawn and none of them decoded. Searching the whole
        # label would usually find the same nothing for five times the cost.
        return []
    return decode(rectify_quad(frame, quad, max_side=FALLBACK_MAX_SIDE))


def diagnose(frame, quad, specs) -> dict:
    """Why a code did not read, in a form somebody can look at.

    "no code" on the plate is the end of a chain with several places to fail
    and no way to tell them apart from the outside: the region can land in the
    wrong place, the crop can be too few pixels, the print can be out of focus,
    or the decoder can simply not be there. Guessing between those from a
    screenshot is not a method.

    So this hands back the actual pixels each step used, plus the one
    comparison that splits the chain in half: the whole label decoded as well.
    A region that reads nothing while the whole label reads fine is a region
    landing in the wrong place -- almost always because the runtime maps it
    onto the detector's box, and the detector's box is not exactly the outline
    the artwork was flattened from. Both empty is a picture problem, not a
    placement one.
    """
    ok, reason = available()
    out = {"ok": ok, "reason": reason, "regions": [], "whole": None}
    if not ok or frame is None or quad is None or len(quad) < 4:
        return out

    for spec in logic.demanded(specs):
        region = list(getattr(spec, "region", None) or [])
        entry = {"spec": spec, "crop": None, "reads": [], "note": "",
                 "how": "", "placed_right": None}
        if len(region) < 4:
            entry["note"] = "no region drawn, so the whole label is searched"
            out["regions"].append(entry)
            continue
        placed = place_unit_rect(quad, _expand(region, region_margin(spec)))
        if not placed:
            entry["note"] = "the region could not be placed on this box"
            out["regions"].append(entry)
            continue
        crop = rectify_quad(frame, placed)
        entry["crop"] = crop
        entry["reads"], entry["how"] = decode(crop, detail=True)
        out["regions"].append(entry)

    whole = rectify_quad(frame, quad, max_side=FALLBACK_MAX_SIDE)
    whole_reads, whole_how = decode(whole, detail=True)
    out["whole"] = {"crop": whole, "reads": whole_reads, "how": whole_how}
    # How wide the label and the frame are in the ORIGINAL pixels, which is
    # what any advice about the camera has to be measured in. The whole-label
    # crop is capped, so its own width says nothing about either.
    try:
        xs = [float(p[0]) for p in quad[:4]]
        out["label_px"] = max(xs) - min(xs)
        out["frame_px"] = float(frame.shape[1])
    except Exception:
        pass
    return out


def diagnosis_text(report: dict) -> str:
    """What the crops say, in the order that narrows it down fastest."""
    if not report.get("ok"):
        return (f"No decoder: {report.get('reason', '')}\n\n"
                f"Nothing below could have read anything.")

    lines = []
    regions = report.get("regions") or []
    whole = report.get("whole") or {}
    got_region = any(r["reads"] for r in regions)
    got_whole = bool(whole.get("reads"))

    for entry in regions:
        crop = entry.get("crop")
        size = f"{crop.shape[1]}x{crop.shape[0]} px" if crop is not None else "no crop"
        role = getattr(entry["spec"], "role", "?")
        if entry["reads"]:
            found = ", ".join(f"{r.text} [{r.symbology}]" for r in entry["reads"])
            how = entry.get("how") or ""
            lines.append(f"Region '{role}' ({size}): {found}"
                         + (f"  [{how}]" if how and how != "as taken" else ""))
        else:
            lines.append(f"Region '{role}' ({size}): nothing decoded"
                         + (f" -- {entry['note']}" if entry["note"] else ""))
        # The one number that says whether this could ever have worked. The
        # crop carries the region plus its margin, so the symbol itself is the
        # declared width as a share of that.
        declared = list(getattr(entry["spec"], "region", None) or [])
        if crop is not None and len(declared) >= 4 and declared[2] > 0:
            grown = declared[2] * (1.0 + region_margin(entry["spec"]))
            symbology = getattr(entry["spec"], "symbology", "")
            note = logic.resolution_note(
                symbology, crop.shape[1] * declared[2] / grown)
            if note:
                lines.append(f"   {note}")
            framing = logic.framing_note(symbology, declared[2],
                                         report.get("label_px", 0.0),
                                         report.get("frame_px", 0.0))
            if framing:
                lines.append(f"   {framing}")

    if whole.get("crop") is not None:
        crop = whole["crop"]
        size = f"{crop.shape[1]}x{crop.shape[0]} px"
        if got_whole:
            found = ", ".join(f"{r.text} [{r.symbology}]" for r in whole["reads"])
            lines.append(f"Whole label ({size}): {found}")
        else:
            lines.append(f"Whole label ({size}): nothing decoded")

    # Did the whole label find THIS code, or just some other symbol on the
    # label? A QR next to the barcode reads happily while the barcode does not,
    # and treating that as "the code is legible" sends somebody to redraw a
    # region that was already right. It is the distinction the first version of
    # this got wrong.
    wanted = [str(getattr(e["spec"], "pattern", "") or "") for e in regions]
    wanted = [w for w in wanted if w]
    whole_is_the_same_code = bool(wanted) and any(
        logic.matches(pattern, text)
        for r in (whole.get("reads") or []) for text in r.candidates()
        for pattern in wanted)
    # Where the code actually is, as fractions of the label. The whole-label
    # crop IS the label, so the decoder's own coordinates convert straight
    # across -- which settles the placement question by measurement instead of
    # argument, and hands back the numbers to type into the region.
    measured = ""
    if whole_is_the_same_code and whole.get("crop") is not None:
        crop = whole["crop"]
        height, width = crop.shape[:2]
        for read in whole.get("reads") or []:
            if not read.box or not any(
                    logic.matches(pattern, text)
                    for text in read.candidates() for pattern in wanted):
                continue
            x0, y0, x1, y1 = read.box
            measured = (f"{x0 / width:.3f}, {y0 / height:.3f}, "
                        f"{(x1 - x0) / width:.3f}, {(y1 - y0) / height:.3f}")
            for entry in regions:
                declared = list(getattr(entry["spec"], "region", None) or [])
                if len(declared) < 4:
                    continue
                entry["placed_right"] = (
                    declared[0] < x1 / width
                    and declared[0] + declared[2] > x0 / width
                    and declared[1] < y1 / height
                    and declared[1] + declared[3] > y0 / height)
            break

    if wanted and got_whole and not whole_is_the_same_code:
        got_whole = False
        found = ", ".join(f"{r.text} [{r.symbology}]"
                          for r in whole.get("reads") or [])
        lines.append("")
        lines.append(f"NOTE: the whole label decoded {found}, which is a "
                     f"different symbol -- not the code this label\'s pattern "
                     f"describes. So it is no evidence that the code in "
                     f"question is legible.")

    lines.append("")
    if got_region:
        lines.append("READS. Whatever comes back here is what the pattern is "
                     "matched against -- compare it character for character "
                     "with what you typed.")
        first = next((r for e in regions for r in e["reads"]), None)
        if first is not None and len(first.candidates()) > 1:
            lines.append(
                f"A UPC-A and an EAN-13 share an encoding, and decoders "
                f"commonly return the 13-digit form. A pattern written for "
                f"either spelling matches: {' or '.join(first.candidates())}.")
        rung = next((e.get("how") for e in regions if e["reads"] and e.get("how")), "")
        if rung and rung != "as taken":
            lines.append(
                f"It only read on the '{rung}' attempt, which means this crop is "
                f"marginal -- readable today, and the first thing to fail when "
                f"the print, the focus or the lighting drifts. More pixels "
                f"across the bars is the durable fix.")
    elif got_whole and any(e.get("placed_right") for e in regions):
        lines.append(
            "THE REGION IS IN THE RIGHT PLACE, AND ITS CROP WILL NOT READ.\n\n"
            f"The decoder found this code at {measured} (x, y, w, h as "
            f"fractions of the label), which overlaps the region as drawn. So "
            f"redrawing it is not the fix.\n\n"
            "What differs between the two crops is how each was made. The whole "
            "label is capped in size, so it was shrunk on the way out -- and "
            "shrinking averages pixels, which quietly removes sensor noise and "
            "the colour moire that sits at the bar pitch. The region is cropped "
            "at full resolution, which keeps all of it. More pixels is not "
            "always a better picture of a barcode, and the decoder now tries "
            "shrinking the region too.")
    elif got_whole:
        lines.append(
            "THE REGION IS LANDING IN THE WRONG PLACE. The code is legible in "
            "this frame -- the whole label read it -- but the crop taken from "
            "the region did not contain it."
            + (f"\n\nThe decoder found this code at {measured} (x, y, w, h as "
               f"fractions of the label). Put those numbers in the region and it "
               f"will crop the right area." if measured else "")
            + "\n\nThe region is a fraction of the label, and at runtime it is "
              "mapped onto the DETECTOR's box rather than the outline the "
              "artwork was flattened from. Where those differ, the region lands "
              "somewhere else on the label.")
    else:
        lines.append(
            "THIS CODE IS NOT DECODING ANYWHERE -- not from the region, and not "
            "from the whole label. That makes it a picture problem rather than "
            "a placement one, so redrawing the region will not help.\n\n"
            "Every fallback was tried: greyscale, a fixed threshold, upscaling, "
            "downscaling, and a single colour channel. What is left is the "
            "picture. Read the pixels-per-module line above first -- if it says "
            "MARGINAL, that is the answer and no decoder setting fixes it. The "
            "label needs to fill more of the frame, through a longer lens or a "
            "closer camera. Otherwise: focus, glare across the symbol, or an "
            "angle steep enough to close the bars up.")
    return "\n".join(lines)


def library_snapshot(library) -> tuple[dict, dict]:
    """``({label_id: [spec, ...]}, {label_id: [pattern, ...]})`` for the worker."""
    specs, patterns = {}, {}
    for label in (library.all() if library is not None else []):
        label_id = str(getattr(label, "label_id", ""))
        frozen = specs_from(label)
        # Carried on the specs rather than in a parallel map: the policy
        # belongs to the label and every spec on it is read the same way up.
        policy = str(getattr(label, "rotation_policy", "") or "")
        for spec in frozen:
            spec.rotation_policy = policy
        if frozen:
            specs[label_id] = frozen
        found = [s.pattern for s in logic.demanded(frozen) if s.pattern]
        if found:
            patterns[label_id] = found
    return specs, patterns
