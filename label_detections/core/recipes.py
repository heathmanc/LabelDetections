"""Recipes: the vision program's bill of labels, with an ROI for each.

This is the runtime side's contract, authored here and consumed there. It
answers "what must be on this battery, and where do I look for it?" -- the
label library answers "what does that label look like?", and the labeling tool
trains one label at a time without ever reading a recipe. Keeping the three
apart is what lets a label be shared by forty recipes, trained once, and
re-aimed at a different ROI without retraining anything.

An **ROI** is a normalised ``[x, y, w, h]`` rectangle in its camera's frame,
each value 0..1. Normalised rather than pixels on purpose: a camera swap or a
resolution change re-scales every ROI for free, where a pixel rect silently
starts pointing at the wrong part of the battery. The ROI does two jobs at
once -- it scopes the search, so the runtime looks for the spec plate only
where the spec plate belongs, and it locates the result, so a label found in
the wrong place is a placement failure rather than a pass.

One recipe holds one view per camera, because the cross-checks that matter
most (the serial on the plate equalling the serial on the tag) span cameras
and cannot live inside any single one.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .labels import SEVERITIES
from .storage import safe_token

DEFAULT_CATEGORY = "General"

CROSS_CHECK_TYPES = ["equal", "not_equal", "pattern"]


def parse_count(spec: Any) -> tuple[int, int | None]:
    """Turn a count spec into ``(minimum, maximum)``; ``None`` means unbounded.

    Accepts ``2``, ``"2"``, ``"0..1"``, ``"1..*"``. Anything unparseable reads
    as "exactly one" -- the overwhelmingly common case -- rather than raising
    into the middle of an inspection.
    """
    if isinstance(spec, bool):
        return (1, 1)
    if isinstance(spec, int):
        n = max(0, spec)
        return (n, n)
    text = str(spec or "").strip()
    if not text:
        return (1, 1)
    if ".." in text:
        lo_text, _, hi_text = text.partition("..")
        try:
            lo = max(0, int(lo_text.strip() or 0))
        except ValueError:
            lo = 0
        hi_text = hi_text.strip()
        if hi_text in ("*", "n", "N", ""):
            return (lo, None)
        try:
            return (lo, max(lo, int(hi_text)))
        except ValueError:
            return (lo, None)
    try:
        n = max(0, int(text))
        return (n, n)
    except ValueError:
        return (1, 1)


def count_spec_error(spec: Any) -> str:
    """Why a count spec is wrong, or "" when it is fine.

    Separate from ``parse_count`` on purpose: parsing is forgiving because it
    runs inside an inspection, where raising is worse than guessing. Authoring
    is strict, because a typo caught in the wizard costs seconds and one caught
    in production costs a shift.
    """
    if isinstance(spec, bool):
        return "Count must be a number or a range, not a yes/no."
    if isinstance(spec, int):
        return "" if spec >= 0 else "Count cannot be negative."
    text = str(spec or "").strip()
    if not text:
        return ""
    if ".." in text:
        lo_text, _, hi_text = text.partition("..")
        try:
            lo = int(lo_text.strip() or 0)
        except ValueError:
            return f"'{lo_text.strip()}' is not a number."
        hi_text = hi_text.strip()
        if hi_text in ("*", "n", "N", ""):
            return ""
        try:
            hi = int(hi_text)
        except ValueError:
            return f"'{hi_text}' is not a number. Use a number or * for no limit."
        if hi < lo:
            return f"Count '{text}' has a maximum below its minimum."
        return ""
    try:
        int(text)
    except ValueError:
        return f"'{text}' is not a count. Use a number, or a range like 0..1 or 1..*."
    return ""


def format_count(spec: Any) -> str:
    lo, hi = parse_count(spec)
    if hi is None:
        return f"at least {lo}"
    if lo == hi:
        return f"exactly {lo}"
    return f"{lo} to {hi}"


def normalise_roi(roi: Any) -> list[float]:
    """Clean an ROI to four floats, or return an empty list.

    An empty ROI is legal and means "anywhere in this frame": presence is
    checked, placement is not. That is the right default for a label whose
    position genuinely varies.
    """
    try:
        values = [float(v) for v in list(roi)[:4]]
    except Exception:
        return []
    if len(values) < 4 or values[2] <= 0 or values[3] <= 0:
        return []
    return values


def roi_contains(roi: list[float], x: float, y: float, tol: float = 0.0) -> bool:
    """Is a normalised point inside an ROI, allowing ``tol`` slack on every side?"""
    if not roi:
        return True
    rx, ry, rw, rh = roi
    return (rx - tol) <= x <= (rx + rw + tol) and (ry - tol) <= y <= (ry + rh + tol)


def roi_pixels(roi: list[float], frame_w: int, frame_h: int) -> list[int]:
    """An ROI in pixels for a given frame, for drawing and for cropping."""
    if not roi:
        return [0, 0, int(frame_w), int(frame_h)]
    rx, ry, rw, rh = roi
    return [int(round(rx * frame_w)), int(round(ry * frame_h)),
            int(round(rw * frame_w)), int(round(rh * frame_h))]


@dataclass
class LabelRequirement:
    """One line of the bill: a label, where to look for it, and what it costs."""
    label_id: str
    roi: list[float] = field(default_factory=list)   # normalised [x, y, w, h]
    count: Any = 1
    severity: str = "fail"
    # Slack around the ROI, normalised. Absorbs fixture play and the few
    # percent a battery shifts on a conveyor without widening the search.
    roi_tol: float = 0.02
    rotation_tol_deg: float = 0.0        # 0 inherits the label's own tolerance
    # Per-recipe override of a code policy, keyed by the code's role. Lets one
    # recipe demand a decode where another only needs the code present.
    code_policy: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LabelRequirement":
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in dict(data or {}).items() if k in known}
        clean.setdefault("label_id", "")
        clean["roi"] = normalise_roi(clean.get("roi"))
        return cls(**clean)


@dataclass
class ViewSpec:
    """One camera: which labels it is responsible for, and where each sits."""
    view: str
    camera: str = ""
    frame_size: list[int] = field(default_factory=list)   # [w, h] px, for display
    labels: list[LabelRequirement] = field(default_factory=list)
    # Labels that must NOT appear. Usually the neighbouring model's: a wrong
    # label is present and correct-looking, so nothing in the required bill
    # notices it.
    forbidden: list[str] = field(default_factory=list)
    # A correctly identified label the bill never asked for. Usually a warning:
    # promo stickers come and go.
    unexpected_severity: str = "warn"
    notes: str = ""

    def requirement(self, label_id: str) -> LabelRequirement | None:
        for r in self.labels:
            if r.label_id == label_id:
                return r
        return None

    def expected_ids(self) -> set[str]:
        return {r.label_id for r in self.labels}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["labels"] = [l.to_dict() for l in self.labels]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ViewSpec":
        data = dict(data or {})
        labels = [LabelRequirement.from_dict(l) for l in data.pop("labels", []) if isinstance(l, dict)]
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in data.items() if k in known}
        clean.setdefault("view", "view")
        obj = cls(**clean)
        obj.labels = labels
        return obj


@dataclass
class CrossCheck:
    """A consistency rule between two read values, usually across cameras.

    References are ``view.label_id.role``, e.g.
    ``side_a.spec_plate_31agm.serial``. ``pattern`` checks use ``left`` only.
    """
    type: str = "equal"
    left: str = ""
    right: str = ""
    pattern: str = ""
    severity: str = "fail"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossCheck":
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in dict(data or {}).items() if k in known}
        return cls(**clean)


@dataclass
class Recipe:
    """A battery model's complete inspection definition."""
    group: str
    model: str
    category: str = DEFAULT_CATEGORY
    revision: str = ""
    # Off means the recipe imposes no bill at all -- free-form, everything
    # passes. Mirrors BungVision's ``constrained`` escape hatch.
    constrained: bool = True
    views: list[ViewSpec] = field(default_factory=list)
    cross_checks: list[CrossCheck] = field(default_factory=list)
    notes: str = ""

    @property
    def safe_name(self) -> str:
        base = f"{safe_token(self.group)}__{safe_token(self.model)}"
        if str(self.category).strip() in ("", DEFAULT_CATEGORY):
            return base
        return f"{safe_token(self.category)}__{base}"

    def view(self, name: str) -> ViewSpec | None:
        for v in self.views:
            if v.view == name:
                return v
        return None

    def view_names(self) -> list[str]:
        return [v.view for v in self.views]

    def label_ids(self) -> set[str]:
        """Every label the recipe mentions -- required or forbidden.

        This is the list the labeling tool cares about: these are the labels
        that need a trained model behind them before the recipe can run.
        """
        ids: set[str] = set()
        for v in self.views:
            ids |= v.expected_ids()
            ids |= set(v.forbidden)
        return ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "model": self.model,
            "category": self.category,
            "revision": self.revision,
            "constrained": self.constrained,
            "views": [v.to_dict() for v in self.views],
            "cross_checks": [c.to_dict() for c in self.cross_checks],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        data = dict(data or {})
        views = [ViewSpec.from_dict(v) for v in data.pop("views", []) if isinstance(v, dict)]
        checks = [CrossCheck.from_dict(c) for c in data.pop("cross_checks", []) if isinstance(c, dict)]
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in data.items() if k in known}
        clean.setdefault("group", "Default")
        clean.setdefault("model", "Model")
        obj = cls(**clean)
        obj.views = views
        obj.cross_checks = checks
        return obj


def parse_ref(ref: str) -> tuple[str, str, str] | None:
    """Split ``view.label_id.role`` into its parts. None when malformed."""
    parts = [p for p in str(ref or "").split(".") if p]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def rois_overlap(a: list[float], b: list[float]) -> bool:
    """Do two normalised ROIs intersect at all?"""
    if not a or not b:
        return False
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def validate_recipe(recipe: Recipe, library=None) -> list[str]:
    """Problems that would make a recipe misbehave in production.

    The expensive mistakes are the quiet ones: a bill naming a label nobody
    trained, an ROI written outside the frame so it can never match, or a
    cross-check pointing at a code role that label does not carry -- which
    does not fail loudly, it just never fires.
    """
    issues: list[str] = []
    if not str(recipe.group).strip() or not str(recipe.model).strip():
        issues.append("Group and model are required.")
    if not recipe.views:
        issues.append("A recipe needs at least one camera view.")

    seen_views: set[str] = set()
    for view in recipe.views:
        if view.view in seen_views:
            issues.append(f"Duplicate view '{view.view}'.")
        seen_views.add(view.view)
        if view.unexpected_severity not in SEVERITIES:
            issues.append(
                f"{view.view}: unexpected-label severity must be one of {', '.join(SEVERITIES)}.")

        seen_labels: set[str] = set()
        for req in view.labels:
            if req.label_id in seen_labels:
                issues.append(f"{view.view}: label '{req.label_id}' is listed twice.")
            seen_labels.add(req.label_id)
            if req.severity not in SEVERITIES:
                issues.append(
                    f"{view.view}/{req.label_id}: severity must be one of {', '.join(SEVERITIES)}.")
            if library is not None and req.label_id not in library:
                issues.append(
                    f"{view.view}: '{req.label_id}' is not in the label library, so nothing "
                    "has been trained to find it.")
            if req.roi:
                x, y, w, h = req.roi
                if x < 0 or y < 0 or x + w > 1.0001 or y + h > 1.0001:
                    issues.append(
                        f"{view.view}/{req.label_id}: ROI runs outside the frame "
                        f"({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f}); values are fractions of "
                        "the image, 0 to 1.")
            count_error = count_spec_error(req.count)
            if count_error:
                issues.append(f"{view.view}/{req.label_id}: {count_error}")
            lo, hi = parse_count(req.count)
            if lo == 0 and hi == 0:
                issues.append(
                    f"{view.view}/{req.label_id}: a required count of zero belongs on the "
                    "forbidden list instead.")
        for bad in view.forbidden:
            if bad in seen_labels:
                issues.append(f"{view.view}: '{bad}' is both required and forbidden.")
            if library is not None and bad not in library:
                issues.append(
                    f"{view.view}: forbidden label '{bad}' is not in the label library.")

    for i, check in enumerate(recipe.cross_checks, start=1):
        if check.type not in CROSS_CHECK_TYPES:
            issues.append(f"Cross-check {i}: unknown type '{check.type}'.")
        refs = [check.left] if check.type == "pattern" else [check.left, check.right]
        for ref in refs:
            parsed = parse_ref(ref)
            if parsed is None:
                issues.append(f"Cross-check {i}: '{ref}' is not view.label_id.role.")
                continue
            view_name, label_id, role = parsed
            view = recipe.view(view_name)
            if view is None:
                issues.append(f"Cross-check {i}: no view named '{view_name}'.")
            elif label_id not in view.expected_ids():
                issues.append(f"Cross-check {i}: '{label_id}' is not required on {view_name}.")
            if library is not None:
                label = library.get(label_id)
                if label is not None and label.code_by_role(role) is None:
                    if not any(t.name == role for t in label.text_fields):
                        issues.append(
                            f"Cross-check {i}: '{label_id}' has no code or text field with "
                            f"role '{role}', so this check can never fire.")
        if check.type == "pattern" and not check.pattern:
            issues.append(f"Cross-check {i}: pattern check with no pattern.")
    return issues
