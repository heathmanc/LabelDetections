"""The annotation sidecar: one label, its box, and its sub-regions.

One JSON file per training image. The labeling tool trains a label at a time,
so a sidecar describes instances of **one** label -- typically one, sometimes
several when a battery carries a pair -- plus whatever barcode and text
regions sit inside each.

Two fields carry the two-stage design:

``label`` / ``class_id``
    the coarse **detector family** the model is trained to find.
``label_id``
    the **library identity**: which exact label this is.

Conflating them is what forces a retrain every time a label SKU changes, so
they stay separate fields that never learn about each other.

``regions`` nest inside a box (barcodes, text fields). They are not top-level
boxes, so a YOLO exporter walks straight past them without needing to know
they exist.

The same structure describes a runtime frame result, which is what lets the
comparison engine be tested against hand-written annotations and then run
unchanged against live detections.
"""
from __future__ import annotations

from typing import Any

from . import geometry as geo
from .ids import ensure_id

# The optional whole-face box. A fixed camera on a fixed fixture does not need
# it -- ROIs in frame coordinates are enough -- but when the battery can shift,
# annotating the face lets placement be judged relative to the battery instead
# of relative to the frame.
BATTERY_SIDE = "battery_side"

REGION_ROLES = ["code", "text", "anchor"]


def new_annotation(image: str, label_id: str = "", width: int = 0, height: int = 0,
                   **meta: Any) -> dict[str, Any]:
    """An empty sidecar for one training image.

    ``label_id`` records which label's dataset this image belongs to, so a
    file that gets moved or re-imported still knows what it was collected for.
    ``meta`` carries optional provenance -- ``source``, ``session``, ``view`` --
    which the group-aware split uses to keep near-duplicates together.
    """
    data: dict[str, Any] = {
        "image": str(image),
        "label_id": str(label_id),
        "width": int(width),
        "height": int(height),
        "boxes": [],
    }
    data.update({k: v for k, v in meta.items() if v not in (None, "")})
    return data


def make_box(family: str, points: list[list[float]], *, box_id: str = "",
             label_id: str = "", parent_id: str = "",
             confidence: float | None = None, **extra: Any) -> dict[str, Any]:
    """One outer detection: a battery side or an identified label."""
    box: dict[str, Any] = {
        "id": box_id or ensure_id(),
        "label": str(family),
        "kind": "obb",
        "points": [[float(p[0]), float(p[1])] for p in points[:4]],
    }
    if label_id:
        box["label_id"] = str(label_id)
    if parent_id:
        box["parent_id"] = str(parent_id)
    if confidence is not None:
        box["confidence"] = float(confidence)
    box.update(extra)
    return box


def make_region(role: str, points: list[list[float]], **extra: Any) -> dict[str, Any]:
    """A sub-area of a label: a code, a text field, or the matching anchor."""
    region: dict[str, Any] = {
        "role": str(role),
        "kind": "obb",
        "points": [[float(p[0]), float(p[1])] for p in points[:4]],
    }
    region.update(extra)
    return region


def box_polygon(box: dict) -> list[list[float]]:
    """Four image-space corners for either an OBB or a legacy axis-aligned box."""
    pts = box.get("points") or box.get("obb") or []
    if len(pts) >= 4:
        return [[float(p[0]), float(p[1])] for p in pts[:4]]
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", 0.0))
    h = float(box.get("h", 0.0))
    return geo.rect_corners(x, y, w, h)


def box_center(box: dict) -> tuple[float, float]:
    return geo.quad_centroid(box_polygon(box))


def box_center_norm(box: dict, width: int, height: int) -> tuple[float, float] | None:
    """A box's centre as a fraction of the frame, for ROI comparison.

    Normalised rather than pixels so a recipe survives a camera or resolution
    change: the ROI and the detection are both fractions of whatever frame
    they came from.
    """
    if not width or not height:
        return None
    cx, cy = box_center(box)
    return cx / float(width), cy / float(height)


def boxes(data: dict | None) -> list[dict]:
    raw = (data or {}).get("boxes")
    return [b for b in raw if isinstance(b, dict)] if isinstance(raw, list) else []


def battery_side_box(data: dict | None) -> dict | None:
    """The optional whole-face box, or None when it was not annotated.

    Only the first is returned: two battery faces in one frame means two
    batteries in view, which is a fixture problem to fix rather than something
    inspection should quietly average over.
    """
    for box in boxes(data):
        if str(box.get("label", "")) == BATTERY_SIDE:
            return box
    return None


def label_boxes(data: dict | None) -> list[dict]:
    """Every box that is a label rather than the battery side."""
    return [b for b in boxes(data) if str(b.get("label", "")) != BATTERY_SIDE]


def identified_boxes(data: dict | None) -> list[dict]:
    return [b for b in label_boxes(data) if str(b.get("label_id", "")).strip()]


def unidentified_boxes(data: dict | None) -> list[dict]:
    """Labels the detector found but nothing could name.

    Never silently dropped: an unidentified label is either a new SKU nobody
    added to the library or a genuine wrong-label defect, and both need eyes.
    """
    return [b for b in label_boxes(data) if not str(b.get("label_id", "")).strip()]


def boxes_for(data: dict | None, label_id: str) -> list[dict]:
    return [b for b in label_boxes(data) if str(b.get("label_id", "")) == str(label_id)]


def label_inventory(data: dict | None) -> dict[str, int]:
    """``{label_id: count}`` for one view. The left-hand side of the comparison."""
    counts: dict[str, int] = {}
    for box in identified_boxes(data):
        key = str(box["label_id"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def family_inventory(data: dict | None) -> dict[str, int]:
    """``{detector_family: count}``. Used by dataset health, not by inspection."""
    counts: dict[str, int] = {}
    for box in boxes(data):
        key = str(box.get("label", "") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def regions(box: dict, role: str = "") -> list[dict]:
    raw = box.get("regions")
    items = [r for r in raw if isinstance(r, dict)] if isinstance(raw, list) else []
    return [r for r in items if not role or str(r.get("role", "")) == role]


def code_region(box: dict, code_role: str) -> dict | None:
    """The code region on this box carrying a given semantic role."""
    for region in regions(box, "code"):
        if str(region.get("code_role", "")) == str(code_role):
            return region
    return None


def read_value(box: dict, role: str) -> str | None:
    """The value read for ``role``, from a decoded code or an OCR'd text field.

    Codes win over text: a decoded barcode is ground truth in a way OCR of the
    human-readable line underneath it is not.
    """
    region = code_region(box, role)
    if region is not None and region.get("decode_ok"):
        text = str(region.get("decoded", "") or "")
        if text:
            return text
    for region in regions(box, "text"):
        if str(region.get("field", "")) == str(role):
            text = str(region.get("ocr", "") or "")
            if text:
                return text
    return None


# --- reference-anchored placement ------------------------------------------

def place_label_regions(box: dict, label_def) -> list[dict]:
    """Sub-regions for a label, positioned from its library artwork.

    This is the shortcut that removes most hand drawing, at labeling time and at
    runtime alike: the operator draws the label's four corners, the library
    already knows the barcode occupies a given fraction of the label, and the
    code's image quad is one homography away.

    It needs no measurement and no calibration -- the mapping is proportional,
    so it holds at any distance and any angle. A label with no regions drawn on
    its artwork simply gets none placed.
    """
    quad = box_polygon(box)

    out: list[dict] = []
    for code in getattr(label_def, "codes", []) or []:
        placed = geo.place_unit_rect(quad, list(getattr(code, "region", []) or []))
        if placed is None:
            continue
        out.append(make_region(
            "code", placed,
            code_role=code.role,
            symbology=code.symbology,
            placed_from="reference",
        ))
    for field in getattr(label_def, "text_fields", []) or []:
        placed = geo.place_unit_rect(quad, list(getattr(field, "region", []) or []))
        if placed is None:
            continue
        out.append(make_region(
            "text", placed, field=field.name, placed_from="reference",
        ))
    anchor = list(getattr(label_def, "anchor_region", []) or [])
    placed = geo.place_unit_rect(quad, anchor)
    if placed is not None:
        out.append(make_region("anchor", placed, placed_from="reference"))
    return out


def apply_reference_regions(box: dict, label_def, *, overwrite: bool = False) -> dict:
    """Attach reference-placed regions to a box, keeping hand-drawn ones.

    An operator who nudged a region because the artwork drifted has produced
    better data than the library has; only ``overwrite`` throws that away.
    """
    placed = place_label_regions(box, label_def)
    if not placed:
        return box
    existing = regions(box)
    if overwrite:
        kept = [r for r in existing if r.get("placed_from") != "reference"]
    else:
        kept = existing
    have = {(str(r.get("role")), str(r.get("code_role", "")), str(r.get("field", ""))) for r in kept}
    for region in placed:
        key = (str(region.get("role")), str(region.get("code_role", "")), str(region.get("field", "")))
        if key not in have:
            kept.append(region)
            have.add(key)
    box["regions"] = kept
    return box
