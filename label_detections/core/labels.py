"""The label library: what a label *is*, independent of any recipe.

The central design decision of this tool lives here. A label's identity is
**data**, not a trained class. The detector learns a handful of coarse
families (``spec_plate``, ``warning_label``, ...) that stay stable for years;
which exact label a detection *is* -- artwork, revision, part number -- is
resolved afterwards by decoding its barcode and matching its reference
artwork. Adding a new label SKU is therefore a row in this library, not a
retraining cycle.

Two coordinate facts make the rest of the system work:

* ``size_mm`` gives every label a real-world size, so a detection whose
  rectified size is wildly off is a misdetection rather than a defect.
* code and text regions are stored in **label space** (millimetres within the
  label's own artwork), so once an operator draws the label's four corners the
  barcode's position follows from the library instead of being drawn by hand.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Coarse detector classes. Deliberately few and visually distinct: this list is
# the thing that costs a retrain to change, so it holds *kinds* of label, never
# individual SKUs.
DEFAULT_FAMILIES = [
    "battery_side",   # the whole side; drives rectification, always labeled
    "spec_plate",
    "warning_label",
    "cert_mark",
    "trace_tag",
    "promo_label",
    "code_patch",
]

SYMBOLOGIES = [
    "code128", "code39", "itf14", "ean13", "upca", "gs1_128",
    "qr", "datamatrix", "pdf417", "aztec",
]

# What the inspection must be able to do with a code for the label to pass.
CODE_POLICIES = ["ignore", "must_be_present", "must_decode", "must_match_pattern"]

# What a code means, so cross-checks between labels and views can name it.
CODE_ROLES = ["serial", "part_number", "date_code", "lot", "revision", "other"]

SEVERITIES = ["fail", "warn", "info"]

ROTATION_POLICIES = ["fixed", "flip_ok", "any"]

SURFACES = ["matte", "gloss", "foil", "holographic", "clear_on_metal"]

SHAPES = ["rectangle", "rounded", "die_cut", "wrap_around"]


@dataclass
class CodeSpec:
    """One barcode or 2D code carried by a label."""
    role: str = "other"
    symbology: str = "datamatrix"
    policy: str = "must_decode"
    pattern: str = ""              # regex the decoded text must match
    # Where the code sits within the label artwork, [x, y, w, h] in mm.
    # Present means the runtime never has to *find* the code -- it crops it.
    region_mm: list[float] = field(default_factory=list)
    x_dim_mm: float = 0.0          # narrow-bar width (1D) or cell size (2D)
    quiet_zone_mm: float = 0.0
    grade: bool = False            # run ISO 15415/15416 print grading

    def min_pixels_needed(self) -> float:
        """Pixels across the whole code needed to hit the decode threshold.

        A 1D symbol wants ~2 px per narrow bar and a 2D symbol ~3 px per cell
        before a decoder gets reliable. Multiplied out against ``region_mm``
        this becomes the concrete "your camera is not sharp enough" number
        that the dataset health report shows, instead of an argument about
        whether the model is at fault.
        """
        if not self.x_dim_mm or not self.region_mm:
            return 0.0
        per_module = 3.0 if self.symbology in ("qr", "datamatrix", "aztec") else 2.0
        modules = float(self.region_mm[2]) / float(self.x_dim_mm)
        return modules * per_module


@dataclass
class TextField:
    """A human-readable field on the label that inspection must verify."""
    name: str = ""
    region_mm: list[float] = field(default_factory=list)
    pattern: str = ""
    policy: str = "must_be_present"     # ignore | must_be_present | must_match_pattern
    # For date codes: reject when the read date is older than this many days.
    max_age_days: int = 0


@dataclass
class LabelDef:
    """Everything the add-label wizard asks about one label."""
    label_id: str
    name: str = ""
    family: str = "spec_plate"          # detector class this label belongs to
    revision: str = ""
    effective_date: str = ""            # ISO date this revision became valid
    supersedes: str = ""                # label_id this replaces, for changeover
    part_number: str = ""               # part number of the label stock itself
    vendor: str = ""

    size_mm: list[float] = field(default_factory=lambda: [0.0, 0.0])
    shape: str = "rectangle"
    surface: str = "matte"
    color_significant: bool = False     # two labels differing only by colour
    rotation_policy: str = "fixed"
    rotation_tol_deg: float = 8.0

    # Variable-data labels (serial, lot, date printed per unit) must be matched
    # on their unchanging artwork only, or every unit looks like a mismatch.
    variable_data: bool = False
    anchor_region_mm: list[float] = field(default_factory=list)

    reference_images: list[str] = field(default_factory=list)
    codes: list[CodeSpec] = field(default_factory=list)
    text_fields: list[TextField] = field(default_factory=list)

    default_severity: str = "fail"      # if this label is missing
    min_confidence: float = 0.5

    # Training. This label owns its dataset and is trained on its own schedule,
    # so its training settings live with it rather than in a recipe.
    synthetic_ok: bool = True           # composite artwork into training data
    train_target: int = 150             # images to gather before training
    # Labels this one is genuinely mistakable for. Their images are the hard
    # negatives that stop the two being swapped, and they are the obvious
    # candidates for a recipe's forbidden list.
    confusable_with: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabelDef":
        data = dict(data or {})
        codes = [CodeSpec(**c) for c in data.pop("codes", []) if isinstance(c, dict)]
        texts = [TextField(**t) for t in data.pop("text_fields", []) if isinstance(t, dict)]
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in data.items() if k in known}
        clean.setdefault("label_id", "unnamed")
        obj = cls(**clean)
        obj.codes = codes
        obj.text_fields = texts
        return obj

    def code_by_role(self, role: str) -> CodeSpec | None:
        for c in self.codes:
            if c.role == role:
                return c
        return None

    def match_region_mm(self) -> list[float]:
        """The region reference matching should score against.

        Variable-data labels score on their anchor only; everything else scores
        on the whole artwork.
        """
        if self.variable_data and len(self.anchor_region_mm) >= 4:
            return list(self.anchor_region_mm)
        w, h = (self.size_mm + [0.0, 0.0])[:2]
        return [0.0, 0.0, float(w), float(h)]


def validate_label_def(label: LabelDef) -> list[str]:
    """Human-readable problems with a label definition. Empty means usable.

    Advisory at entry time -- the wizard shows these so an operator can fix
    them now -- but the ones that would silently weaken inspection (no size, a
    must-decode code with no region) are worth being loud about.
    """
    issues: list[str] = []
    if not str(label.label_id).strip():
        issues.append("Label ID is required.")
    if label.family not in DEFAULT_FAMILIES:
        issues.append(f"Unknown detector family '{label.family}'.")
    if label.default_severity not in SEVERITIES:
        issues.append(f"Severity must be one of {', '.join(SEVERITIES)}.")
    w, h = (list(label.size_mm) + [0.0, 0.0])[:2]
    if float(w) <= 0 or float(h) <= 0:
        issues.append("Physical size in mm is required -- it is the scale sanity check.")
    if label.variable_data and len(label.anchor_region_mm) < 4:
        issues.append(
            "Variable-data label has no anchor region: matching would score against "
            "text that changes on every unit."
        )
    if not label.reference_images:
        issues.append("No reference image: this label cannot be identified by matching.")
    for i, code in enumerate(label.codes, start=1):
        if code.symbology not in SYMBOLOGIES:
            issues.append(f"Code {i}: unknown symbology '{code.symbology}'.")
        if code.policy not in CODE_POLICIES:
            issues.append(f"Code {i}: unknown policy '{code.policy}'.")
        if code.policy != "ignore" and len(code.region_mm) < 4:
            issues.append(
                f"Code {i}: no region on the artwork, so the runtime must search the "
                "whole label for it instead of cropping straight to it."
            )
        if code.policy == "must_match_pattern" and not code.pattern:
            issues.append(f"Code {i}: policy is must_match_pattern but no pattern was given.")
        if code.policy != "ignore" and not code.x_dim_mm:
            issues.append(
                f"Code {i}: no X-dimension, so decode-feasibility cannot be checked "
                "against the camera resolution."
            )
    for i, tf in enumerate(label.text_fields, start=1):
        if not tf.name:
            issues.append(f"Text field {i}: needs a name.")
        if tf.policy == "must_match_pattern" and not tf.pattern:
            issues.append(f"Text field {i}: policy is must_match_pattern but no pattern was given.")
    return issues


class LabelLibrary:
    """All known labels, keyed by ``label_id``."""

    def __init__(self, labels: list[LabelDef] | None = None):
        self._labels: dict[str, LabelDef] = {}
        for label in labels or []:
            self._labels[label.label_id] = label

    def __len__(self) -> int:
        return len(self._labels)

    def __contains__(self, label_id: object) -> bool:
        return str(label_id) in self._labels

    def add(self, label: LabelDef, *, replace: bool = False) -> None:
        if label.label_id in self._labels and not replace:
            raise ValueError(f"Label already exists: {label.label_id}")
        self._labels[label.label_id] = label

    def get(self, label_id: str) -> LabelDef | None:
        return self._labels.get(str(label_id))

    def remove(self, label_id: str) -> bool:
        return self._labels.pop(str(label_id), None) is not None

    def ids(self) -> list[str]:
        return sorted(self._labels)

    def all(self) -> list[LabelDef]:
        return [self._labels[k] for k in self.ids()]

    def by_family(self, family: str) -> list[LabelDef]:
        return [l for l in self.all() if l.family == family]

    def families_in_use(self) -> list[str]:
        """Detector classes actually needed by the library, in canonical order.

        Always includes ``battery_side``: without it there is nothing to
        rectify against and zone checks are impossible.
        """
        used = {l.family for l in self.all()}
        used.add("battery_side")
        return [f for f in DEFAULT_FAMILIES if f in used]

    def to_dict(self) -> dict[str, Any]:
        return {"labels": [l.to_dict() for l in self.all()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LabelLibrary":
        raw = (data or {}).get("labels", [])
        return cls([LabelDef.from_dict(d) for d in raw if isinstance(d, dict)])
