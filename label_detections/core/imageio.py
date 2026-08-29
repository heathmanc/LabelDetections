"""Image IO: writing captures and importing images into a label's dataset.

Split out from ``storage`` so that module stays stdlib-only and the schema,
gate and split logic remain testable with nothing installed. Everything here
needs OpenCV, because it decodes and re-encodes pixels.

Ported from BungVision Label Studio with one change: a dataset belongs to a
label rather than to a recipe. The folder shape is identical -- one directory
under ``captures/`` -- so the sidecar lookup and the import path carried over
untouched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry import landscape_quad, order_quad

from .storage import (
    IMAGE_SUFFIXES, dataset_folder, image_label_json_path, safe_token,
    save_annotations,
)

IMPORT_IMAGE_EXTS = IMAGE_SUFFIXES + (".webp",)


def _settled(quad, orient: bool, landscape: bool):
    """The corner order to flatten from: as given, ordered, or turned flat."""
    if landscape:
        return landscape_quad(quad)
    return order_quad(quad) if orient else quad


def rectify_quad(image_bgr: np.ndarray, quad: list[list[float]],
                 out_width: int = 0, max_side: int = 0,
                 orient: bool = True, landscape: bool = False) -> np.ndarray | None:
    """Flatten a drawn label out of a captured frame.

    The operator has already drawn the label's four corners on a real capture.
    Warping that quad to a rectangle gives a deskewed, straight-on picture of
    the label -- which is both the natural surface to draw read-regions on and
    the natural reference image for the library.

    It removes the need to go and find vendor artwork: the tool is already
    collecting pictures of the label, and one of them, flattened, IS the
    artwork. Returns None for a degenerate quad.

    ``max_side`` caps the longest output side. warpPerspective costs what its
    destination costs, so flattening a 2000 px label at native size and then
    shrinking it to a 224 px crop pays for 2000 px of warp to keep 224 -- about
    25x the work, measured on a 5496x3672 frame. It only ever shrinks: a label
    smaller than the cap is left alone rather than blown up.

    The corners are put in a known order first. This function maps slot 0 to
    the output's top-left and slot 1 to its top-right, so a quad that happens
    to wind anticlockwise flattens to a MIRROR IMAGE -- and a detector's
    oriented box winds whichever way the predicted angle implies. Drawn boxes
    are already in this order, so ordering them changes nothing; detector boxes
    are the ones that were coming out mirrored, or with the short edge across
    the top. Pass ``orient=False`` only when the caller's slot order is itself
    the thing being tested.

    ``landscape`` goes one step further and turns the quad so its long edge is
    the width. For a crop a classifier will see: the same label lying flat and
    standing up otherwise flattens to a 300x90 and a 90x300, which is one
    picture turned a quarter and two pictures to anything learning from them.
    See ``geometry.landscape_quad``.
    """
    if image_bgr is None or len(quad) < 4:
        return None
    points = [[float(x), float(y)] for x, y in
              (_settled(quad[:4], orient, landscape))]

    def edge(a, b):
        return float(np.hypot(points[b][0] - points[a][0], points[b][1] - points[a][1]))

    # Average opposite edges: a label on a curved battery face keystones by a
    # few percent, and picking one edge would bake that skew into the output.
    width = (edge(0, 1) + edge(3, 2)) / 2.0
    height = (edge(1, 2) + edge(0, 3)) / 2.0
    if width < 4 or height < 4:
        return None

    if out_width > 0:
        scale = out_width / width
        width, height = out_width, max(4.0, height * scale)
    if max_side > 0 and max(width, height) > max_side:
        scale = float(max_side) / max(width, height)
        width, height = max(4.0, width * scale), max(4.0, height * scale)

    src = np.array(points, dtype=np.float32)
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    try:
        matrix = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(image_bgr, matrix, (int(round(width)), int(round(height))))
    except cv2.error:
        return None


def save_reference(label_id: str, image_bgr: np.ndarray, root: Path | None = None) -> Path:
    """Store a flattened label crop as that label's reference artwork."""
    folder = (root or dataset_folder(label_id).parent.parent / "library") / "references"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{safe_token(label_id)}.png"
    cv2.imwrite(str(path), image_bgr)
    return path


def save_capture(
    label_id: str,
    frame_bgr: np.ndarray,
    adjusted_bgr: np.ndarray | None = None,
    save_raw: bool = True,
) -> tuple[Path | None, Path | None]:
    """Write a capture into a label's dataset folder.

    ``save_raw=False`` with an adjusted frame writes only the adjusted image.
    Capturing adjusted used to emit both, which doubled the dataset and left an
    unadjusted twin of every frame to label or delete by hand.

    Returns (raw_path, adjusted_path); either may be None.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    folder = dataset_folder(label_id)
    folder.mkdir(parents=True, exist_ok=True)
    stem = safe_token(label_id)
    # Two captures inside the same millisecond produced the same name, and the
    # second silently overwrote the first -- image, and the sidecar keyed to
    # it. Rare by hand, but reachable by holding the capture shortcut down.
    base = f"{stem}_{ts}_{ms:03d}"
    suffix = 1
    while (folder / f"{base}.jpg").exists() or (folder / f"{base}_adjusted.jpg").exists():
        base = f"{stem}_{ts}_{ms:03d}_{suffix}"
        suffix += 1

    raw_path = None
    if save_raw or adjusted_bgr is None:
        raw_path = folder / f"{base}.jpg"
        cv2.imwrite(str(raw_path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    adjusted_path = None
    if adjusted_bgr is not None:
        adjusted_path = folder / f"{base}_adjusted.jpg"
        cv2.imwrite(str(adjusted_path), adjusted_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    return raw_path, adjusted_path


def find_sidecar_json(image_src: Path, json_dir: Path | None = None) -> Path | None:
    """Locate a runtime-style sidecar label JSON for a source image.

    If ``json_dir`` is given the JSON is looked up there (parallel-directory
    layout: images and labels live in separate sibling folders).  Otherwise
    the JSON is looked up next to the image file (co-located layout).

    Supports both ``foo.json`` (stem) and ``foo.jpg.json`` (full-name) naming.
    """
    if json_dir is not None:
        candidates = [
            json_dir / f"{image_src.stem}.json",
            json_dir / f"{image_src.name}.json",
        ]
    else:
        candidates = [
            image_src.with_suffix(".json"),
            Path(str(image_src) + ".json"),
        ]
    for c in candidates:
        if c.exists():
            return c
    return None


def import_images(
    label_id: str,
    paths: list[Path | str],
    json_dir: Path | None = None,
    as_background: bool = False,
) -> tuple[list[Path], list[str], int]:
    """Copy external images (and any sidecar label JSON) into a label's dataset.

    Each source image is decoded and re-encoded to JPEG under the dataset's
    normal naming convention so it shows up in the captured-image list.

    ``as_background=True`` marks every imported image as a deliberate negative
    (an empty conveyor, a bare fixture) instead of looking for sidecar labels.

    If ``json_dir`` is supplied, the matching ``.json`` label file is looked up
    there (parallel-directory layout).  Otherwise the JSON is expected to sit
    next to the image (co-located layout).  When a sidecar is found, its boxes
    and review/source metadata are written into the label's sidecar folder under
    the new image name so imported labels appear immediately.

    Returns (imported_paths, errors, label_count).
    """
    folder = dataset_folder(label_id)
    folder.mkdir(parents=True, exist_ok=True)
    imported: list[Path] = []
    errors: list[str] = []
    label_count = 0
    ts = time.strftime("%Y%m%d_%H%M%S")
    for i, src in enumerate(paths):
        src = Path(src)
        try:
            img = cv2.imread(str(src))
            if img is None:
                errors.append(f"Could not read image: {src.name}")
                continue
            base = f"{safe_token(label_id)}_import_{ts}_{i:04d}"
            dest = folder / f"{base}.jpg"
            cv2.imwrite(str(dest), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            imported.append(dest)

            if as_background:
                # Bulk negatives: an empty conveyor has nothing to label, so the
                # annotation is written on import rather than making the operator
                # open and mark hundreds of images by hand.
                _write_background_label(dest, img)
                label_count += 1
                continue

            sidecar = find_sidecar_json(src, json_dir=json_dir)
            if sidecar is not None:
                try:
                    data = json.loads(sidecar.read_text(encoding="utf-8"))
                    _write_imported_label(dest, img, data)
                    label_count += 1
                except Exception as exc:
                    errors.append(f"{src.name} label JSON: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{src.name}: {exc}")
    return imported, errors, label_count


def _write_background_label(image_path: Path, img_bgr: "np.ndarray") -> Path:
    """Write a reviewed, zero-box annotation marking an image as a negative."""
    from .review import make_background_record

    h, w = img_bgr.shape[:2]
    return save_annotations(
        image_path, int(w), int(h), [], [],
        review=make_background_record(),
        background=True,
    )


def _write_imported_label(image_path: Path, img_bgr: "np.ndarray", data: dict[str, Any]) -> Path:
    """Write a sidecar label JSON for an imported image, preserving its content.

    The full source payload is kept (boxes plus any review/source metadata) but
    the image path and dimensions are corrected to the newly imported file.
    """
    h, w = img_bgr.shape[:2]
    payload = dict(data) if isinstance(data, dict) else {}
    payload["image"] = str(image_path)
    try:
        payload["width"] = int(payload.get("width") or w)
        payload["height"] = int(payload.get("height") or h)
    except (TypeError, ValueError):
        payload["width"], payload["height"] = w, h
    payload["boxes"] = payload.get("boxes") or []
    # Record provenance so review tooling treats these as imported.
    payload.setdefault("imported_from", "image_import")
    path = image_label_json_path(image_path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


