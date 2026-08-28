"""The label library: what a label *is*, independent of any recipe.

The central design decision of this tool lives here. A label's identity is
**data**, not a trained class. The detector learns a handful of coarse
families (``spec_plate``, ``warning_label``, ...) that stay stable for years;
which exact label a detection *is* -- artwork, revision, part number -- is
resolved afterwards by decoding its barcode and matching its reference
artwork. Adding a new label SKU is therefore a row in this library, not a
retraining cycle.

Regions -- the areas inside a label that have to be read on their own, a
barcode or a serial or a date code -- are stored in **label space**: ``[x, y,
w, h]`` as fractions of the label itself, ``0`` to ``1``. Once an operator
draws the label's four corners on a real image, every region follows by
homography, at whatever angle, distance or resolution the camera saw it.

Fractions rather than millimetres on purpose. The mapping is pure proportion,
so nothing has to be measured and nothing has to be calibrated for distance.
A region is a thing you drag on the artwork, not a number you look up.

Physical sizes appear in exactly one place -- ``CodeSpec.code_width_mm`` and
``x_dim_mm`` -- and only to answer "can the camera resolve this code at all",
which is a question about the printed symbol, not about the camera's distance.
Both are optional and both come off the label's print spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# The detector is trained on **label ids**. The recipe the front end authors is
# a list of label ids and quantities, so anything coarser has to be resolved
# back into an id before it is worth anything -- and that resolution step is
# pure cost: a decoder, artwork matching and a tie-breaker to maintain, each
# with its own failure modes, all to recover an answer the model could have
# given directly.
#
# The price is honest and worth stating: a new label is not detectable until it
# has images and a training run. That is a smaller price than it looks, because
# a new label needs its images collected either way -- train_target is already
# per label, and the dataset is already per label. What it adds is one batch
# training run over data that was being extended anyway.
#
# BATTERY_SIDE is the exception, and the only one: it is not a label, it is the
# whole face. It drives rectification and lets placement be judged against the
# battery instead of the frame, so it is a class in its own right and no label
# ever owns it.
STRUCTURAL_CLASSES = ["battery_side"]

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
    # Where the code sits on the label: [x, y, w, h] as fractions of the label,
    # 0 to 1. Drawn on the reference artwork, never typed. Present means the
    # runtime crops straight to the code instead of having to search for it.
    region: list[float] = field(default_factory=list)

    # Optional, and only for the decode-feasibility check below. Both come off
    # the label's print spec -- they describe the printed symbol, not the
    # camera or its distance.
    code_width_mm: float = 0.0     # physical width of the whole symbol
    x_dim_mm: float = 0.0          # narrow-bar width (1D) or cell size (2D)
    quiet_zone_mm: float = 0.0
    grade: bool = False            # run ISO 15415/15416 print grading

    def min_pixels_needed(self) -> float:
        """Pixels across the code needed before a decoder is reliable, or 0.

        A 1D symbol wants ~2 px per narrow bar and a 2D symbol ~3 px per cell.
        Multiplied out this is the concrete "the camera is not sharp enough"
        number, which is worth knowing before a trial rather than after -- no
        model recovers a code the optics never resolved.

        Returns 0 when the print spec was not entered, which is fine: it is an
        optional check, not a gate.
        """
        if not self.x_dim_mm or not self.code_width_mm:
            return 0.0
        per_module = 3.0 if self.symbology in ("qr", "datamatrix", "aztec") else 2.0
        return (float(self.code_width_mm) / float(self.x_dim_mm)) * per_module


@dataclass
class TextField:
    """A human-readable field on the label that inspection must read.

    This is the answer to text that changes per unit: the artwork around a
    serial never moves, so a region pinned to the artwork finds the serial on
    every battery without anyone knowing in advance what it will say.
    """
    name: str = ""
    # [x, y, w, h] as fractions of the label. Drawn, not measured.
    region: list[float] = field(default_factory=list)
    pattern: str = ""
    policy: str = "must_be_present"     # ignore | must_be_present | must_match_pattern
    # For date codes: reject when the read date is older than this many days.
    max_age_days: int = 0


@dataclass
class LabelDef:
    """Everything the add-label wizard asks about one label."""
    label_id: str
    name: str = ""
    revision: str = ""
    effective_date: str = ""            # ISO date this revision became valid
    supersedes: str = ""                # label_id this replaces, for changeover
    part_number: str = ""               # part number of the label stock itself
    vendor: str = ""

    # Optional documentation, off the label's drawing. Nothing computes with
    # it: region placement is proportional, so no size is needed to put a
    # barcode box where it belongs.
    size_mm: list[float] = field(default_factory=lambda: [0.0, 0.0])
    shape: str = "rectangle"
    surface: str = "matte"
    color_significant: bool = False     # two labels differing only by colour
    rotation_policy: str = "fixed"
    rotation_tol_deg: float = 8.0

    # Variable-data labels (serial, lot, date printed per unit) must be matched
    # on their unchanging artwork only, or every unit looks like a mismatch.
    variable_data: bool = False
    # The part of the artwork that never changes, as fractions of the label.
    # Matching scores against this alone, or a per-unit serial makes every
    # battery look like a mismatch.
    anchor_region: list[float] = field(default_factory=list)

    # Artwork, and the dataset image it was flattened out of. Recorded so the
    # image list can mark that capture: "which shot did I define regions from?"
    # is otherwise unanswerable once a dataset has a few hundred frames.
    reference_images: list[str] = field(default_factory=list)
    reference_source: str = ""
    # The artwork's width over its height. Regions are fractions of the label,
    # which is proportion without orientation -- so placing one back onto a
    # detected quad needs to know which way round the label's own proportions
    # run, or a label photographed standing up gets them ninety degrees out.
    reference_aspect: float = 0.0
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
        known_code = set(CodeSpec.__dataclass_fields__)
        known_text = set(TextField.__dataclass_fields__)
        codes = [CodeSpec(**{k: v for k, v in c.items() if k in known_code})
                 for c in data.pop("codes", []) if isinstance(c, dict)]
        texts = [TextField(**{k: v for k, v in t.items() if k in known_text})
                 for t in data.pop("text_fields", []) if isinstance(t, dict)]
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

    def match_region(self) -> list[float]:
        """The region matching should score against, as fractions of the label.

        Variable-data labels score on their anchor alone; everything else scores
        on the whole artwork.
        """
        if self.variable_data and len(self.anchor_region) >= 4:
            return list(self.anchor_region)
        return [0.0, 0.0, 1.0, 1.0]

    def regions(self) -> list[tuple[str, str, list[float]]]:
        """Every drawn region as ``(role, name, rect)``, for display and export."""
        out = [("code", c.role, list(c.region)) for c in self.codes if c.region]
        out += [("text", t.name, list(t.region)) for t in self.text_fields if t.region]
        if len(self.anchor_region) >= 4:
            out.append(("anchor", "anchor", list(self.anchor_region)))
        return out


def _valid_region(region: list[float]) -> bool:
    """A drawn region: four fractions inside the label, with real area."""
    if not region or len(region) < 4:
        return False
    try:
        x, y, w, h = (float(v) for v in region[:4])
    except (TypeError, ValueError):
        return False
    return w > 0 and h > 0 and x >= 0 and y >= 0 and x + w <= 1.001 and y + h <= 1.001


def validate_label_def(label: LabelDef) -> list[str]:
    """Human-readable problems with a label definition. Empty means usable.

    Advisory at entry time -- the wizard shows these so an operator can fix
    them now -- but the ones that would silently weaken inspection (no size, a
    must-decode code with no region) are worth being loud about.
    """
    issues: list[str] = []
    if not str(label.label_id).strip():
        issues.append("Label ID is required.")
    if str(label.label_id).strip() in STRUCTURAL_CLASSES:
        issues.append(
            f"'{label.label_id}' is a structural class, not a label. Pick another id.")
    if label.default_severity not in SEVERITIES:
        issues.append(f"Severity must be one of {', '.join(SEVERITIES)}.")
    if label.variable_data and len(label.anchor_region) < 4:
        issues.append(
            "Variable-data label has no anchor region: matching would score against "
            "text that changes on every unit."
        )
    # Artwork is not a prerequisite. It comes from a capture: draw the label's
    # box on a collected image and Define Regions flattens that box into
    # straight-on artwork. Only flag it when something needs positioning on it.
    if not label.reference_images and (label.codes or label.text_fields):
        issues.append(
            "This label reads something but has no artwork to position it on. Draw "
            "its box on a captured image and use Define Regions."
        )
    for i, code in enumerate(label.codes, start=1):
        if code.symbology not in SYMBOLOGIES:
            issues.append(f"Code {i}: unknown symbology '{code.symbology}'.")
        if code.policy not in CODE_POLICIES:
            issues.append(f"Code {i}: unknown policy '{code.policy}'.")
        if code.policy != "ignore" and not _valid_region(code.region):
            issues.append(
                f"Code {i}: no region drawn on the artwork, so the runtime must search "
                "the whole label for it instead of cropping straight to it."
            )
        if code.policy == "must_match_pattern" and not code.pattern:
            issues.append(f"Code {i}: policy is must_match_pattern but no pattern was given.")
    for i, tf in enumerate(label.text_fields, start=1):
        if not tf.name:
            issues.append(f"Text field {i}: needs a name.")
        if tf.policy != "ignore" and not _valid_region(tf.region):
            issues.append(
                f"Text field {i} ('{tf.name}'): no region drawn, so there is nowhere "
                "to read it from."
            )
        if tf.policy == "must_match_pattern" and not tf.pattern:
            issues.append(f"Text field {i}: policy is must_match_pattern but no pattern was given.")
    return issues


# Fields a typed query is matched against. Everything an operator might
# reasonably remember about a label: what it is called, its revision, and the
# part number on the purchase order.
SEARCH_FIELDS = ("label_id", "name", "revision", "part_number", "vendor")


def match_label(label: "LabelDef", query: str) -> bool:
    """Does a label match a typed query?

    Every whitespace-separated term must appear somewhere, in any field and in
    any order -- so "g31 warn" finds the G31 warning label without anyone having
    to remember whether it was named warning_g31 or g31_warning. Matching one
    field with one substring is not enough once there are hundreds: the operator
    remembers something about the label, not its exact id.
    """
    terms = str(query or "").lower().split()
    if not terms:
        return True
    haystack = " ".join(
        str(getattr(label, field, "") or "").lower() for field in SEARCH_FIELDS)
    return all(term in haystack for term in terms)


def search_labels(labels: list["LabelDef"], query: str) -> list["LabelDef"]:
    """The labels matching a query, in the order given."""
    return [label for label in labels if match_label(label, query)]


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

    def search(self, query: str) -> list[LabelDef]:
        """Labels matching a typed query."""
        return search_labels(self.all(), query)

    def detector_classes(self) -> list[str]:
        """Every class the detector is trained on: the structural ones, then
        one per label id.

        Structural classes lead so ``battery_side`` keeps index 0 as labels
        come and go. Class indices are written into every exported label file,
        so a list that reshuffles on an unrelated edit would silently
        re-point every annotation in the dataset at the wrong class.
        """
        return list(STRUCTURAL_CLASSES) + sorted(
            l.label_id for l in self.all() if l.label_id not in STRUCTURAL_CLASSES)

    def to_dict(self) -> dict[str, Any]:
        return {"labels": [l.to_dict() for l in self.all()]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "LabelLibrary":
        raw = (data or {}).get("labels", [])
        return cls([LabelDef.from_dict(d) for d in raw if isinstance(d, dict)])
