"""Importing images into a label's dataset.

Bulk negatives are the case worth pinning: nothing about them is visible in
the UI beyond a count, so a failure here reads as "it imported" right up until
training finds no background images at all.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("LABELVISION_DATA_DIR",
                      tempfile.mkdtemp(prefix="labelvision-import-"))

import pytest

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except Exception:  # pragma: no cover - depends on the environment
    HAVE_CV2 = False

pytestmark = pytest.mark.skipif(not HAVE_CV2, reason="cv2 not available")


def _source_image(tmp_path: Path, name="src.png") -> Path:
    path = tmp_path / name
    cv2.imwrite(str(path), np.zeros((80, 120, 3), dtype=np.uint8))
    return path


def test_importing_as_background_writes_the_negative_annotation(tmp_path):
    """It called a function the module never imported, so every background
    import came back as a NameError in the per-file error list -- counted as a
    failed file rather than surfacing as the bug it was."""
    from label_detections.core import imageio, storage

    src = _source_image(tmp_path)
    imported, errors, labelled = imageio.import_images(
        "bg_import", [src], as_background=True)

    assert errors == []
    assert (len(imported), labelled) == (1, 1)

    sidecar = storage.image_label_json_path(imported[0])
    assert sidecar.exists()

    import json
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data.get("background") is True
    assert data.get("boxes") == []


def test_a_plain_import_leaves_the_image_unlabelled(tmp_path):
    """No sidecar beside the source means no sidecar written -- an imported
    image with an empty annotation would read as a reviewed negative."""
    from label_detections.core import imageio, storage

    src = _source_image(tmp_path, "plain.png")
    imported, errors, labelled = imageio.import_images("bg_plain", [src])

    assert errors == []
    assert (len(imported), labelled) == (1, 0)
    assert not storage.image_label_json_path(imported[0]).exists()
