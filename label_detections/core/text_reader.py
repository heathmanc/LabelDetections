"""Reading the printed text off a detected label.

The deciding half is ``core/text_read``. This half finds the pixels and runs
the recogniser.

Two things matter and both are about the region being known in advance. The
crop is taken from the FULL-RESOLUTION frame, for the same reason the code
reader does it: stage 2's crop is a few hundred pixels of whole label, and a
part number inside that is a dozen pixels tall. And because the library already
says where the text sits, the recogniser is run WITHOUT its text-detection
stage -- measured at 15 ms against 1113 ms for detect-then-recognise, which is
the difference between a check that fits inside a frame budget and one that
does not. Detection is asking "where is there text on this label", and that
question was answered once, on the artwork, by an operator drawing a box.
"""
from __future__ import annotations

from . import text_read as logic
from .geometry import oriented, place_unit_rect
from .imageio import rectify_quad

# A little context around the drawn box. Recognisers are trained on crops with
# some margin, and a box drawn tight to the glyphs clips ascenders.
REGION_MARGIN = 0.18

# Recognition models want a text line around this tall. Their own preprocessing
# will resize to it, but doing it here with a good interpolation beats letting
# a nearest-neighbour resize eat a 20 px cap height.
TARGET_LINE_PX = 48

_BACKEND = None
_REASON = ""


def backend():
    """The recogniser, built once. ``(engine, reason)`` -- engine is None on failure."""
    global _BACKEND, _REASON
    if _BACKEND is None and not _REASON:
        try:
            from rapidocr_onnxruntime import RapidOCR

            _BACKEND = RapidOCR()
        except Exception as exc:
            _REASON = (f"rapidocr-onnxruntime is not installed, so printed text "
                       f"cannot be read ({exc}). pip install rapidocr-onnxruntime")
    return _BACKEND, _REASON


def available() -> tuple[bool, str]:
    engine, reason = backend()
    return engine is not None, reason


def _scaled(image):
    """A crop at the height the recogniser wants, never shrunk below it."""
    import cv2

    height = image.shape[0]
    if height <= 0 or height >= TARGET_LINE_PX:
        return image
    factor = TARGET_LINE_PX / float(height)
    return cv2.resize(image, None, fx=factor, fy=factor,
                      interpolation=cv2.INTER_CUBIC)


def recognise(image, detect: bool = False) -> list[logic.Read]:
    """The text in one crop.

    ``detect=False`` treats the whole crop as a single line, which is what the
    drawn region promises and what keeps this at 15 ms. ``detect=True`` runs the
    full pipeline for the diagnostic, where a second is affordable and the
    question is what is on the label at all rather than what one line says.
    """
    engine, _reason = backend()
    if engine is None or image is None or getattr(image, "size", 0) == 0:
        return []
    try:
        if detect:
            result, _elapsed = engine(image)
            rows = [(row[1], row[2]) for row in (result or [])]
        else:
            result, _elapsed = engine(_scaled(image), use_det=False,
                                      use_cls=False, use_rec=True)
            rows = [(row[0], row[1]) for row in (result or [])]
    except Exception:
        return []
    out = []
    for text, confidence in rows:
        text = str(text or "").strip()
        if text:
            out.append(logic.Read(text=text, confidence=float(confidence or 0.0)))
    return out


def _expand(rect: list[float], margin: float) -> list[float]:
    x, y, w, h = (float(v) for v in rect[:4])
    x -= w * margin / 2.0
    y -= h * margin / 2.0
    w *= 1.0 + margin
    h *= 1.0 + margin
    x, y = max(0.0, x), max(0.0, y)
    return [x, y, min(w, 1.0 - x), min(h, 1.0 - y)]


def crop_field(frame, quad, spec, flipped: bool = False):
    """The pixels of one text field, out of the full-resolution frame."""
    region = list(getattr(spec, "region", None) or [])
    if frame is None or quad is None or len(quad) < 4 or len(region) < 4:
        return None
    placed = place_unit_rect(oriented(quad, flipped),
                             _expand(region, REGION_MARGIN), orient=False)
    if not placed:
        return None
    return rectify_quad(frame, placed)


def read_label(frame, quad, fields, rotation_policy: str = "") -> list[logic.Read]:
    """Read every text field this label demands, out of one frame.

    Both ways up where the library says the label may arrive turned over. Four
    corners cannot say which way up printing is, so an upside-down label puts
    every region at the diagonally opposite end -- and the only thing that
    settles it is reading and seeing which reading says what it should.

    The reads are pooled rather than the orientation being chosen here: it is
    the pattern match that decides, and a crop of the wrong end of the label
    almost never produces text that matches an enrolled part number.
    """
    from .code_reader import orientations

    engine, _reason = backend()
    if engine is None:
        return []
    out: list[logic.Read] = []
    for spec in logic.demanded(fields):
        for flipped in orientations(rotation_policy):
            crop = crop_field(frame, quad, spec, flipped)
            if crop is not None:
                out.extend(recognise(crop))
    return out


class Field:
    """A frozen copy of one TextField, safe to hand to the worker thread."""

    __slots__ = ("name", "policy", "pattern", "region", "rotation_policy")

    def __init__(self, name="", policy="", pattern="", region=()):
        self.name = name
        self.policy = policy
        self.pattern = pattern
        self.region = list(region)
        self.rotation_policy = ""


def fields_from(label) -> list[Field]:
    return [
        Field(name=str(getattr(f, "name", "") or ""),
              policy=str(getattr(f, "policy", "") or ""),
              pattern=str(getattr(f, "pattern", "") or ""),
              region=[float(v) for v in (getattr(f, "region", None) or [])])
        for f in (getattr(label, "text_fields", None) or [])
    ]


def library_snapshot(library) -> tuple[dict, dict]:
    """``({label_id: [field, ...]}, {label_id: [string, ...]})`` for the worker."""
    fields, expected = {}, {}
    for label in (library.all() if library is not None else []):
        label_id = str(getattr(label, "label_id", ""))
        frozen = fields_from(label)
        policy = str(getattr(label, "rotation_policy", "") or "")
        for spec in frozen:
            spec.rotation_policy = policy
        if frozen:
            fields[label_id] = frozen
        wanted = [f.pattern.lstrip("^").rstrip("$")
                  for f in logic.demanded(frozen) if f.pattern]
        if wanted:
            expected[label_id] = wanted
    return fields, expected


def diagnose(frame, quad, fields) -> dict:
    """Why a field did not read, in a form somebody can look at."""
    ok, reason = available()
    out = {"ok": ok, "reason": reason, "fields": [], "whole": None}
    if not ok or frame is None or quad is None or len(quad) < 4:
        return out

    for spec in logic.demanded(fields):
        entry = {"spec": spec, "crop": None, "reads": [], "note": ""}
        crop = crop_field(frame, quad, spec)
        if crop is None:
            entry["note"] = "no region drawn for this field"
        else:
            entry["crop"] = crop
            entry["reads"] = recognise(crop)
        out["fields"].append(entry)

    # The whole label with detection on, which answers a different question:
    # not "what does this line say" but "what text is on this label at all".
    # A second is affordable here and nowhere near the live path.
    whole = rectify_quad(frame, quad, max_side=1600)
    out["whole"] = {"crop": whole, "reads": recognise(whole, detect=True)}
    return out


def diagnosis_text(report: dict, expected: dict | None = None) -> str:
    """What the crops say, in the order that narrows it down fastest."""
    if not report.get("ok"):
        return (f"No recogniser: {report.get('reason', '')}\n\n"
                f"Nothing below could have been read.")

    lines = []
    fields = report.get("fields") or []
    whole = report.get("whole") or {}
    for entry in fields:
        crop = entry.get("crop")
        size = f"{crop.shape[1]}x{crop.shape[0]} px" if crop is not None else "no crop"
        name = getattr(entry["spec"], "name", "?") or "?"
        if entry["reads"]:
            found = ", ".join(f"'{r.text}' ({r.confidence:.0%})"
                              for r in entry["reads"])
            lines.append(f"Field '{name}' ({size}): {found}")
        else:
            lines.append(f"Field '{name}' ({size}): nothing read"
                         + (f" -- {entry['note']}" if entry["note"] else ""))

    if whole.get("crop") is not None:
        found = ", ".join(f"'{r.text}'" for r in whole.get("reads") or [])
        lines.append(f"Whole label: {found or 'nothing read'}")

    lines.append("")
    got_field = any(e["reads"] for e in fields)
    if got_field and expected:
        joined = " ".join(r.text for e in fields for r in e["reads"])
        label_id, score, runner_up = logic.best_match(joined, expected)
        lines.append(f"Closest enrolled label: {label_id or '-'} at {score:.0%}"
                     + (f", next {runner_up:.0%}" if runner_up else ""))
        if score < logic.MIN_SIMILARITY:
            lines.append(
                f"Under the {logic.MIN_SIMILARITY:.0%} needed to call it that, "
                f"so this reads as unknown. Either the label is genuinely not "
                f"enrolled -- which is the point -- or the expected text for it "
                f"is not what is actually printed.")
        else:
            lines.append("Enough to identify it.")
    elif got_field:
        lines.append("Read, but no label declares what text to expect, so there "
                     "is nothing to check it against.")
    elif whole.get("reads"):
        lines.append(
            "THE FIELD REGION IS LANDING IN THE WRONG PLACE. Text is legible on "
            "this label -- the whole-label pass read some -- but the crop taken "
            "from the field did not contain any. Compare what the whole label "
            "found against where the region is drawn.")
    else:
        lines.append(
            "NO TEXT READ ANYWHERE. That makes it the picture rather than the "
            "region: a cap height under about 20 px, out of focus, glare, or an "
            "angle steep enough to close the letters up.")
    return "\n".join(lines)
