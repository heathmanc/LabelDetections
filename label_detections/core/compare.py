"""The comparison engine: what a camera found, against the recipe's bill.

This is the executable definition of what a recipe *means*. The vision program
runs it per frame and per battery; the authoring tool runs it to show what a
recipe will and will not catch before anyone ships it. Pure -- dicts in,
findings out, no Qt, no OpenCV, no filesystem -- so the decision that fails a
battery can be tested exhaustively instead of observed on a conveyor.

The central asymmetry it exists to handle: a detector can only report what
*is* in the frame. "Missing spec plate" is not a detection, it is the absence
of one, and the only thing that can name that absence is the recipe. So the
loop is over requirements, not over detections, and detections no requirement
claimed are swept up afterwards.

Placement is judged against each requirement's **ROI** -- a normalised
rectangle in that camera's frame. Both the detection and the ROI are
fractions of the frame, so the check is resolution-independent.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any

from . import annotations as ann
from . import geometry as geo
from .recipes import (
    LabelRequirement, Recipe, ViewSpec, format_count, parse_count, parse_ref,
    roi_contains,
)

# Reason codes. Stable strings: they end up in production logs, get counted in
# Pareto charts, and are what someone greps for six months later.
MISSING = "missing"
WRONG_COUNT = "wrong_count"
FORBIDDEN = "forbidden"
UNEXPECTED = "unexpected"
UNIDENTIFIED = "unidentified"
OUT_OF_ROI = "out_of_roi"
ROTATED = "rotated"
WRONG_SHAPE = "wrong_shape"
CODE_MISSING = "code_missing"
CODE_UNREADABLE = "code_unreadable"
CODE_PATTERN = "code_pattern"
CROSS_CHECK = "cross_check"
NO_FRAME_SIZE = "no_frame_size"
NOT_IN_LIBRARY = "not_in_library"
LOW_CONFIDENCE = "low_confidence"

PASS, WARN, FAIL = "pass", "warn", "fail"
_RANK = {PASS: 0, "info": 0, WARN: 1, FAIL: 2}


def _worst(*verdicts: str) -> str:
    best = PASS
    for v in verdicts:
        if _RANK.get(v, 0) > _RANK.get(best, 0):
            best = v if v != "info" else best
    return best


@dataclass
class Finding:
    """One thing wrong, in terms an operator can act on."""
    code: str
    severity: str
    message: str
    view: str = ""
    label_id: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RequirementRow:
    """One line of the inspection report, mirroring one line of the bill."""
    label_id: str
    expected: str
    found: int
    verdict: str
    values: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ViewResult:
    view: str
    verdict: str = PASS
    rows: list[RequirementRow] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "verdict": self.verdict,
            "rows": [r.to_dict() for r in self.rows],
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class UnitResult:
    unit_id: str
    verdict: str = PASS
    views: list[ViewResult] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    values: dict[str, str] = field(default_factory=dict)

    def all_findings(self) -> list[Finding]:
        out: list[Finding] = []
        for view in self.views:
            out.extend(view.findings)
        out.extend(self.findings)
        return out

    def failures(self) -> list[Finding]:
        return [f for f in self.all_findings() if f.severity == FAIL]

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "verdict": self.verdict,
            "views": [v.to_dict() for v in self.views],
            "findings": [f.to_dict() for f in self.findings],
            "values": dict(self.values),
        }

    def summary_text(self) -> str:
        """Operator-facing summary, in the spirit of BungVision's count summary."""
        lines = [f"Unit {self.unit_id}: {self.verdict.upper()}"]
        for view in self.views:
            lines.append("")
            lines.append(f"[{view.view}] {view.verdict.upper()}")
            for row in view.rows:
                mark = {PASS: "OK", WARN: "!!", FAIL: "XX"}.get(row.verdict, "??")
                note = f"  ({'; '.join(row.notes)})" if row.notes else ""
                lines.append(f"  {mark} {row.label_id}: expected {row.expected}, found {row.found}{note}")
            for finding in view.findings:
                if finding.label_id and any(r.label_id == finding.label_id for r in view.rows):
                    continue
                lines.append(f"  -- {finding.message}")
        if self.findings:
            lines.append("")
            lines.append("Cross-view checks:")
            for finding in self.findings:
                lines.append(f"  -- {finding.message}")
        return "\n".join(lines)


# --- per-label checks ------------------------------------------------------

def _check_codes(box: dict, label_def, requirement: LabelRequirement,
                 view_name: str) -> tuple[list[Finding], dict[str, str]]:
    """Verify every code the library says this label carries."""
    findings: list[Finding] = []
    values: dict[str, str] = {}
    severity = requirement.severity

    for code in getattr(label_def, "codes", []) or []:
        policy = requirement.code_policy.get(code.role, code.policy)
        if policy == "ignore":
            continue
        region = ann.code_region(box, code.role)
        if region is None:
            findings.append(Finding(
                CODE_MISSING, severity,
                f"{view_name}: {label_def.label_id} has no {code.role} "
                f"{code.symbology} region",
                view_name, label_def.label_id, {"role": code.role},
            ))
            continue
        if policy == "must_be_present":
            continue

        decoded = str(region.get("decoded", "") or "")
        if not region.get("decode_ok") or not decoded:
            findings.append(Finding(
                CODE_UNREADABLE, severity,
                f"{view_name}: {label_def.label_id} {code.role} code did not decode",
                view_name, label_def.label_id,
                {"role": code.role, "symbology": code.symbology,
                 "px_per_module": region.get("px_per_module")},
            ))
            continue

        values[code.role] = decoded
        pattern = code.pattern
        if policy == "must_match_pattern" and pattern:
            try:
                ok = re.search(pattern, decoded) is not None
            except re.error:
                ok = False
                findings.append(Finding(
                    CODE_PATTERN, WARN,
                    f"{view_name}: {label_def.label_id} {code.role} pattern is not "
                    f"valid regex: {pattern}",
                    view_name, label_def.label_id, {"role": code.role},
                ))
            if not ok:
                findings.append(Finding(
                    CODE_PATTERN, severity,
                    f"{view_name}: {label_def.label_id} {code.role} read "
                    f"'{decoded}', which does not match {pattern}",
                    view_name, label_def.label_id,
                    {"role": code.role, "decoded": decoded, "pattern": pattern},
                ))
    return findings, values


def _check_placement(box: dict, label_def, requirement: LabelRequirement,
                     frame: tuple[int, int], view_name: str) -> list[Finding]:
    """ROI, rotation and shape checks for one detection."""
    findings: list[Finding] = []
    width, height = frame

    if requirement.roi:
        centre = ann.box_center_norm(box, width, height)
        if centre is None:
            findings.append(Finding(
                NO_FRAME_SIZE, WARN,
                f"{view_name}: {label_def.label_id} has an ROI but the frame size is "
                "unknown, so placement could not be checked",
                view_name, label_def.label_id,
            ))
        elif not roi_contains(requirement.roi, centre[0], centre[1], requirement.roi_tol):
            rx, ry, rw, rh = requirement.roi
            findings.append(Finding(
                OUT_OF_ROI, requirement.severity,
                f"{view_name}: {label_def.label_id} found at "
                f"({centre[0]:.2f}, {centre[1]:.2f}) of frame, outside its ROI "
                f"({rx:.2f}, {ry:.2f}, {rw:.2f}, {rh:.2f})",
                view_name, label_def.label_id,
                {"centre": [round(centre[0], 3), round(centre[1], 3)], "roi": requirement.roi},
            ))

    policy = getattr(label_def, "rotation_policy", "fixed")
    if policy != "any":
        tol_deg = requirement.rotation_tol_deg or getattr(label_def, "rotation_tol_deg", 0.0)
        if tol_deg:
            angle = geo.quad_angle_deg(ann.box_polygon(box))
            targets = [0.0, 180.0] if policy == "flip_ok" else [0.0]
            if min(geo.angle_delta_deg(angle, t) for t in targets) > float(tol_deg):
                findings.append(Finding(
                    ROTATED, requirement.severity,
                    f"{view_name}: {label_def.label_id} is rotated {angle:.0f} deg "
                    f"(tolerance {float(tol_deg):.0f} deg)",
                    view_name, label_def.label_id, {"angle_deg": round(angle, 1)},
                ))

    # Aspect ratio, not absolute size: it needs no mm calibration and still
    # catches the failure that matters -- a box that swallowed two labels, or
    # clipped half of one, has a badly wrong shape whatever the scale.
    size = list(getattr(label_def, "size_mm", []) or [])
    if len(size) >= 2 and float(size[0]) > 0 and float(size[1]) > 0:
        got_w, got_h = geo.quad_size(ann.box_polygon(box))
        if got_h > 0:
            expected = float(size[0]) / float(size[1])
            got = got_w / got_h
            if expected > 0 and (got / expected > 1.4 or expected / got > 1.4):
                findings.append(Finding(
                    WRONG_SHAPE, WARN,
                    f"{view_name}: {label_def.label_id} is {got:.2f}:1 but the library "
                    f"says {expected:.2f}:1 -- the box may have merged or clipped labels",
                    view_name, label_def.label_id,
                    {"aspect": round(got, 2), "expected_aspect": round(expected, 2)},
                ))
    return findings


# --- per view --------------------------------------------------------------

def compare_view(data: dict | None, view_spec: ViewSpec, library) -> ViewResult:
    """Compare one camera's frame against that view's bill of labels."""
    result = ViewResult(view=view_spec.view)
    frame = (int((data or {}).get("width", 0) or 0),
             int((data or {}).get("height", 0) or 0))
    if not all(frame) and view_spec.frame_size and len(view_spec.frame_size) >= 2:
        frame = (int(view_spec.frame_size[0]), int(view_spec.frame_size[1]))

    claimed: set[int] = set()

    for requirement in view_spec.labels:
        label_def = library.get(requirement.label_id) if library is not None else None
        found = ann.boxes_for(data, requirement.label_id)
        for box in found:
            claimed.add(id(box))

        row = RequirementRow(
            label_id=requirement.label_id,
            expected=format_count(requirement.count),
            found=len(found),
            verdict=PASS,
        )
        row_findings: list[Finding] = []

        if label_def is None:
            row_findings.append(Finding(
                NOT_IN_LIBRARY, WARN,
                f"{view_spec.view}: '{requirement.label_id}' is required but is not "
                "in the label library, so only its presence could be checked",
                view_spec.view, requirement.label_id,
            ))

        lo, hi = parse_count(requirement.count)
        if len(found) < lo:
            code = MISSING if not found else WRONG_COUNT
            row_findings.append(Finding(
                code, requirement.severity,
                f"{view_spec.view}: {requirement.label_id} -- expected "
                f"{format_count(requirement.count)}, found {len(found)}",
                view_spec.view, requirement.label_id,
                {"expected": format_count(requirement.count), "found": len(found)},
            ))
        elif hi is not None and len(found) > hi:
            row_findings.append(Finding(
                WRONG_COUNT, requirement.severity,
                f"{view_spec.view}: {requirement.label_id} -- expected "
                f"{format_count(requirement.count)}, found {len(found)}",
                view_spec.view, requirement.label_id,
                {"expected": format_count(requirement.count), "found": len(found)},
            ))

        if label_def is not None:
            min_conf = float(getattr(label_def, "min_confidence", 0.0) or 0.0)
            for box in found:
                conf = box.get("confidence")
                if conf is not None and float(conf) < min_conf:
                    row_findings.append(Finding(
                        LOW_CONFIDENCE, WARN,
                        f"{view_spec.view}: {requirement.label_id} detected at "
                        f"{float(conf):.2f}, below its {min_conf:.2f} threshold",
                        view_spec.view, requirement.label_id,
                    ))
                row_findings.extend(
                    _check_placement(box, label_def, requirement, frame, view_spec.view)
                )
                code_findings, values = _check_codes(box, label_def, requirement, view_spec.view)
                row_findings.extend(code_findings)
                row.values.update(values)

        row.verdict = _worst(*[f.severity for f in row_findings]) if row_findings else PASS
        row.notes = [f.message.split(": ", 1)[-1] for f in row_findings]
        result.rows.append(row)
        result.findings.extend(row_findings)

    for forbidden_id in view_spec.forbidden:
        found = ann.boxes_for(data, forbidden_id)
        for box in found:
            claimed.add(id(box))
        if found:
            result.findings.append(Finding(
                FORBIDDEN, FAIL,
                f"{view_spec.view}: {forbidden_id} must not be present, but "
                f"{len(found)} was found",
                view_spec.view, forbidden_id, {"found": len(found)},
            ))

    for box in ann.identified_boxes(data):
        if id(box) in claimed:
            continue
        label_id = str(box.get("label_id"))
        result.findings.append(Finding(
            UNEXPECTED, view_spec.unexpected_severity,
            f"{view_spec.view}: {label_id} is on the battery but not in the recipe",
            view_spec.view, label_id,
        ))

    unidentified = ann.unidentified_boxes(data)
    if unidentified:
        result.findings.append(Finding(
            UNIDENTIFIED, WARN,
            f"{view_spec.view}: {len(unidentified)} label(s) detected but not "
            "identified -- a new SKU, or a wrong label",
            view_spec.view, detail={"count": len(unidentified)},
        ))

    result.verdict = _worst(*[f.severity for f in result.findings]) if result.findings else PASS
    return result


# --- per unit --------------------------------------------------------------

def _collect_values(views: dict[str, dict | None]) -> dict[str, str]:
    """Every read value in the unit, keyed ``view.label_id.role``.

    Flattened up front because cross-checks reach across cameras, and a
    per-view lookup would make that the caller's problem.
    """
    values: dict[str, str] = {}
    for view_name, data in views.items():
        for box in ann.identified_boxes(data):
            label_id = str(box.get("label_id"))
            for region in ann.regions(box, "code"):
                role = str(region.get("code_role", ""))
                if role and region.get("decode_ok"):
                    text = str(region.get("decoded", "") or "")
                    if text:
                        values[f"{view_name}.{label_id}.{role}"] = text
            for region in ann.regions(box, "text"):
                field_name = str(region.get("field", ""))
                text = str(region.get("ocr", "") or "")
                if field_name and text:
                    values.setdefault(f"{view_name}.{label_id}.{field_name}", text)
    return values


def _check_cross(check, values: dict[str, str]) -> Finding | None:
    """Evaluate one cross-check against the unit's read values."""
    left = values.get(check.left)
    if check.type == "pattern":
        if left is None:
            return Finding(CROSS_CHECK, check.severity,
                           f"Cross-check: {check.left} was never read", detail={"ref": check.left})
        try:
            ok = re.search(check.pattern, left) is not None
        except re.error:
            return Finding(CROSS_CHECK, WARN,
                           f"Cross-check: '{check.pattern}' is not valid regex")
        if not ok:
            return Finding(CROSS_CHECK, check.severity,
                           f"Cross-check: {check.left} read '{left}', which does not "
                           f"match {check.pattern}",
                           detail={"left": left, "pattern": check.pattern})
        return None

    right = values.get(check.right)
    if left is None or right is None:
        missing = check.left if left is None else check.right
        return Finding(CROSS_CHECK, check.severity,
                       f"Cross-check: {missing} was never read, so "
                       f"{check.left} and {check.right} could not be compared",
                       detail={"missing": missing})
    if check.type == "equal" and left != right:
        return Finding(CROSS_CHECK, check.severity,
                       f"Cross-check: {check.left} read '{left}' but "
                       f"{check.right} read '{right}'",
                       detail={"left": left, "right": right})
    if check.type == "not_equal" and left == right:
        return Finding(CROSS_CHECK, check.severity,
                       f"Cross-check: {check.left} and {check.right} both read "
                       f"'{left}', but they must differ",
                       detail={"left": left, "right": right})
    return None


def compare_unit(views: dict[str, dict | None], recipe: Recipe, library,
                 unit_id: str = "") -> UnitResult:
    """Compare one battery -- every camera at once -- against its recipe.

    ``views`` maps view name to that view's annotation sidecar (or None when
    the camera did not capture). An unconstrained recipe passes everything,
    which is how free-form labeling and background capture stay out of the
    gate's way.
    """
    result = UnitResult(unit_id=unit_id or _infer_unit_id(views))
    if not recipe.constrained:
        return result

    for view_spec in recipe.views:
        data = views.get(view_spec.view)
        if data is None:
            result.findings.append(Finding(
                MISSING, FAIL,
                f"{view_spec.view}: no image was captured for this view",
                view_spec.view, detail={"view": view_spec.view},
            ))
            result.views.append(ViewResult(view=view_spec.view, verdict=FAIL))
            continue
        result.views.append(compare_view(data, view_spec, library))

    result.values = _collect_values({k: v for k, v in views.items() if v is not None})
    for check in recipe.cross_checks:
        finding = _check_cross(check, result.values)
        if finding is not None:
            result.findings.append(finding)

    result.verdict = _worst(
        *[v.verdict for v in result.views],
        *[f.severity for f in result.findings],
    )
    return result


def _infer_unit_id(views: dict[str, dict | None]) -> str:
    for data in views.values():
        if data and str(data.get("unit_id", "")):
            return str(data["unit_id"])
    return "unknown"
