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

import numpy as np

from . import codes as logic
from .geometry import place_unit_rect
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


def decode(image) -> list[logic.Read]:
    """Every symbol in one image.

    Returns an empty list for anything that goes wrong, deliberately. A frame
    where the decoder threw and a frame with no code in it are the same fact as
    far as the verdict is concerned -- nothing was read -- and the policy above
    already treats that as a failure to verify rather than as a pass.
    """
    module, _reason = backend()
    if module is None or image is None or getattr(image, "size", 0) == 0:
        return []
    try:
        results = module.read_barcodes(image)
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
        symbology = getattr(result, "format", "")
        out.append(logic.Read(text=text, symbology=str(symbology)))
    return out


class Spec:
    """A frozen copy of one CodeSpec: the fields the runtime reads, and no more.

    The worker runs off the GUI thread and the library is edited on it, so what
    crosses between them is a snapshot rather than a live reference.
    """

    __slots__ = ("role", "symbology", "policy", "pattern", "region",
                 "quiet_zone_mm", "code_width_mm")

    def __init__(self, role="", symbology="", policy="", pattern="", region=(),
                 quiet_zone_mm=0.0, code_width_mm=0.0):
        self.role = role
        self.symbology = symbology
        self.policy = policy
        self.pattern = pattern
        self.region = list(region)
        self.quiet_zone_mm = float(quiet_zone_mm or 0.0)
        self.code_width_mm = float(code_width_mm or 0.0)


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
    can see to drag around. Decoders need the blank margin either side of them,
    and how much is not a matter of taste -- the symbology specifies it, and
    for a small code it can be most of the symbol's own width again. A fixed
    guess is fine for a Code 128 across a battery face and far too tight for an
    8 mm DataMatrix, where cropping inside the quiet zone means a good part
    reading "no code".

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


def read_label(frame, quad, specs) -> list[logic.Read]:
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
    for spec in wanted:
        region = list(getattr(spec, "region", None) or [])
        if len(region) < 4:
            continue
        placed = place_unit_rect(quad, _expand(region, region_margin(spec)))
        if not placed:
            continue
        # No max_side: this is the one crop in the pipeline that must not be
        # shrunk. Everything else here downsamples for a model; a decoder wants
        # every printed module it can get.
        patch = rectify_quad(frame, placed)
        reads = decode(patch)
        if reads:
            return reads

    if any(len(list(getattr(s, "region", None) or [])) >= 4 for s in wanted):
        # Regions were drawn and none of them decoded. Searching the whole
        # label would usually find the same nothing for five times the cost.
        return []
    return decode(rectify_quad(frame, quad, max_side=FALLBACK_MAX_SIDE))


def library_snapshot(library) -> tuple[dict, dict]:
    """``({label_id: [spec, ...]}, {label_id: [pattern, ...]})`` for the worker."""
    specs, patterns = {}, {}
    for label in (library.all() if library is not None else []):
        label_id = str(getattr(label, "label_id", ""))
        frozen = specs_from(label)
        if frozen:
            specs[label_id] = frozen
        found = [s.pattern for s in logic.demanded(frozen) if s.pattern]
        if found:
            patterns[label_id] = found
    return specs, patterns
