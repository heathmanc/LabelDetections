"""The model half of one-click outlining.

Kept apart from ``core.segment_assist`` so the geometry stays testable without
a checkpoint, and apart from the window so the window does not grow another
model-loading routine.

MobileSAM by default. The job is a printed rectangle of high-contrast paper on
a battery casing, which is an easy segmentation target -- the difference
between MobileSAM and full SAM here is a corner or two, against a 40 MB
download instead of 360 MB and roughly an order of magnitude in time. Any
checkpoint Ultralytics' SAM class accepts can be named instead
(``sam2.1_t.pt``, ``sam_b.pt``); nothing below assumes which.

This never runs on the line. It runs while somebody is labeling at a desk, so
it is allowed to be slow in a way the live path never is.
"""
from __future__ import annotations

from pathlib import Path

from ..core import segment_assist as geometry

DEFAULT_MODEL = "mobile_sam.pt"


class AssistUnavailable(RuntimeError):
    """The assistant could not be loaded, with a sentence for the operator."""


class SegmentAssistant:
    """Loads a promptable segmentation model once and outlines from a point.

    Loaded lazily rather than at startup: most sessions never press the button,
    and a first launch that stalls on a download nobody asked for is worse than
    a first click that does.
    """

    def __init__(self, model_path: str = "", device: str = "") -> None:
        self.model_path = str(model_path or DEFAULT_MODEL)
        self.device = str(device or "")
        self._model = None

    # -- loading ---------------------------------------------------------

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import SAM
        except Exception as exc:  # pragma: no cover - depends on the install
            raise AssistUnavailable(
                "Ultralytics is not installed in this environment, so the "
                f"outline assistant cannot run.\n\n{exc}") from exc
        try:
            self._model = SAM(self.model_path)
        except Exception as exc:
            # A named checkpoint downloads on first use; a path that does not
            # exist never will, and the two failures need different answers.
            hint = ("" if not Path(self.model_path).suffix or "/" not in self.model_path
                    else f"\n\nNo file at {self.model_path}.")
            raise AssistUnavailable(
                f"Could not load the outline model '{self.model_path}'.{hint}\n\n{exc}") from exc

    # -- outlining -------------------------------------------------------

    def outline(self, frame_bgr, x: float, y: float,
                max_px: int = geometry.DEFAULT_ASSIST_PX) -> tuple[list[list[float]], str]:
        """The quad around whatever the model finds at (x, y) in full-frame pixels.

        Returns ``(quad, "")`` or ``([], reason)``. A reason rather than an
        exception because every way this fails is something the operator can
        act on by clicking somewhere else.
        """
        import cv2
        import numpy as np

        if frame_bgr is None:
            return [], "No image is open."
        self.load()

        height, width = frame_bgr.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return [], "That click was outside the image."

        factor = geometry.assist_scale(width, height, max_px)
        if factor < 1.0:
            small = cv2.resize(frame_bgr, (max(1, int(width * factor)),
                                           max(1, int(height * factor))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr

        point = [[float(x) * factor, float(y) * factor]]
        try:
            kwargs = {"points": point, "labels": [1], "verbose": False}
            if self.device:
                kwargs["device"] = self.device
            results = self._model(small, **kwargs)
        except Exception as exc:
            return [], f"The outline model failed on this image.\n\n{exc}"

        mask = _first_mask(results, np)
        if mask is None:
            return [], "Nothing was outlined there. Click on the label itself."
        return geometry.outline_from_mask(mask, width, height, factor)


def _first_mask(results, np):
    """The mask out of an Ultralytics result, as a plain array.

    Defensive about shape because a point-prompted SAM result carries one mask
    per prompt and the array has arrived both as (1, H, W) and (H, W) across
    versions -- and ``.data`` is a torch tensor that must not travel further
    than this function.
    """
    for r in results or []:
        masks = getattr(r, "masks", None)
        if masks is None:
            continue
        data = getattr(masks, "data", None)
        if data is None:
            continue
        try:
            array = data.cpu().numpy() if hasattr(data, "cpu") else np.asarray(data)
        except Exception:
            continue
        if array.size == 0:
            continue
        if array.ndim == 3:
            # More than one mask means the model offered alternatives; the
            # largest is the one that covers the label rather than a detail
            # printed on it.
            return array[int(np.argmax(array.reshape(array.shape[0], -1).sum(axis=1)))]
        return array
    return None
