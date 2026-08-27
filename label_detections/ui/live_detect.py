"""Inference off the GUI thread, for the live detection view.

The preview must never wait on the model. A camera renders at its own rate and
overlays refresh whenever inference finishes, which means the two run at
different speeds and the slow one cannot hold up the fast one -- so the model
lives in its own thread and frames are handed to it only when it is idle.

Loading is in here too: a .pt takes seconds to come off disk and through torch,
and doing that on the GUI thread freezes the window at exactly the moment an
operator is watching for something to happen.
"""
from __future__ import annotations

import time

from PySide6.QtCore import QObject, Signal, Slot


class InferenceWorker(QObject):
    """Owns the model and runs it. Lives on its own thread."""

    loaded = Signal(str)          # human description of what came up
    failed = Signal(str)
    # ultralytics results, latency in seconds, per-detection (name, conf) from
    # stage 2 (empty when no classifier is loaded)
    result = Signal(object, float, object)

    def __init__(self, model_path: str, imgsz: int, conf: float, device,
                 track: bool = True, classifier_path: str = "",
                 crop_px: int = 224, margin: float = 0.06,
                 identity_floor: float = 0.55):
        super().__init__()
        self._path = str(model_path)
        self._imgsz = int(imgsz)
        self._conf = float(conf)
        self._device = device
        self._track = bool(track)
        self._classifier_path = str(classifier_path or "")
        self._crop_px = int(crop_px)
        self._margin = float(margin)
        self._identity_floor = float(identity_floor)
        self._model = None
        self._classifier = None
        self._stopping = False

    @Slot()
    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.failed.emit(
                "Ultralytics is not installed, so live detection cannot run.\n"
                f"{exc}")
            return
        try:
            self._model = YOLO(self._path)
        except Exception as exc:
            self.failed.emit(f"Could not load the model:\n{self._path}\n\n{exc}")
            return

        # A classification model loads perfectly happily as a "detector" and
        # then returns probabilities and no boxes, so the view simply goes
        # quiet with nothing anywhere saying why. Refuse it by name instead.
        task = str(getattr(self._model, "task", "") or "")
        if task == "classify":
            self._model = None
            self.failed.emit(
                f"That is a CLASSIFIER, not a detector:\n{self._path}\n\n"
                f"It has no boxes to give, which is why nothing would appear.\n\n"
                f"Put it in the 'Stage 2 classifier' field on this tab, and put "
                f"the detector run's best.pt in the Test Models field.")
            return
        if self._classifier_path:
            try:
                self._classifier = YOLO(self._classifier_path)
                cls_task = str(getattr(self._classifier, "task", "") or "")
                if cls_task and cls_task != "classify":
                    self._classifier = None
                    self.failed.emit(
                        f"The stage 2 model is a '{cls_task}' model, not a "
                        f"classifier:\n{self._classifier_path}\n\n"
                        f"Running stage 1 only -- boxes will have no identity.")
            except Exception as exc:
                self.failed.emit(
                    f"Detector loaded, but the classifier did not:\n"
                    f"{self._classifier_path}\n\n{exc}\n\n"
                    f"Running stage 1 only -- boxes will have no identity.")
                self._classifier = None
        which = ("detector + classifier" if self._classifier is not None
                 else f"{task or 'detector'} only, no stage 2")
        self.loaded.emit(f"Loaded {which}: {self._path}\n{self._device_report()}")

    def _device_report(self) -> str:
        """What hardware this is actually running on.

        Asked and answered rather than inferred from the frame rate. "Is it
        using the GPU" is otherwise guessed at from throughput, which says
        nothing useful when something else is the bottleneck -- and a torch
        built without CUDA looks, from the outside, exactly like a slow model.
        """
        try:
            import torch
        except Exception:
            return "Device: torch not importable."

        if not torch.cuda.is_available():
            return ("Device: CPU -- torch reports NO CUDA. Either the driver is "
                    "missing or this is a CPU-only torch build; reinstall torch "
                    "with the CUDA wheel for your card.")
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "unknown CUDA device"
        where = "?"
        try:
            where = str(next(self._model.model.parameters()).device)
        except Exception:
            pass
        asked = self._device if self._device is not None else "(unset)"
        line = f"Device: CUDA available ({name}); model on {where}; asked for {asked}."
        if where.startswith("cpu"):
            line += (" The model is on the CPU despite CUDA being available -- "
                     "set Device to 0 on the Test Models tab.")
        return line

    @Slot(object)
    def infer(self, frame) -> None:
        """Run one frame. Silently drops it if the model never loaded."""
        if self._model is None or self._stopping or frame is None:
            return
        args = {"imgsz": self._imgsz, "conf": self._conf, "verbose": False}
        if self._device is not None:
            args["device"] = self._device
        started = time.perf_counter()
        try:
            if self._track:
                # persist=True is what carries the tracker's state between
                # calls; without it every frame starts a fresh tracker and each
                # object is "new" forever, which is the same as not tracking.
                results = self._model.track(frame, persist=True, **args)
            else:
                results = self._model.predict(frame, **args)
        except Exception as exc:
            # One bad frame must not take the view down; the readout says so and
            # the next frame gets its own try.
            self.failed.emit(str(exc))
            return
        identities = self._identify(frame, results)
        self.result.emit(results, time.perf_counter() - started, identities)

    def _detection_quads(self, results):
        """Four corners per detection, in the SAME order the overlay builds them.

        Order is the coupling that makes identities line up with boxes, so it
        mirrors _detection_overlay_items exactly: oriented results first when
        present, otherwise axis-aligned, each in index order. A mismatch is
        caught downstream by length and degrades to no identity rather than to
        a wrong one.
        """
        import numpy as np

        quads = []
        for r in results or []:
            obb = getattr(r, "obb", None)
            if obb is not None:
                try:
                    polys = obb.xyxyxyxy.cpu().numpy()
                except Exception:
                    polys = []
                if len(polys):
                    for poly in polys:
                        pts = np.array(poly, dtype=float).reshape(-1, 2)[:4]
                        if len(pts) >= 4:
                            quads.append(pts.tolist())
                    continue
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            try:
                xyxy = boxes.xyxy.cpu().numpy()
            except AttributeError:
                xyxy = np.asarray(getattr(boxes, "xyxy", []))
            except Exception:
                continue
            for box in xyxy:
                x1, y1, x2, y2 = (float(v) for v in box[:4])
                quads.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return quads

    def _identify(self, frame, results):
        """Stage 2: crop each detection out of the FULL-RESOLUTION frame and name it.

        From ``frame``, deliberately -- not from anything the detector resized.
        The entire reason this stage exists is that the detector's input threw
        detail away; re-using its view would reproduce the problem it solves.
        """
        if self._classifier is None:
            return []
        from label_detections.core import live_detect as logic
        from label_detections.core.classify_export import expand_quad, letterbox
        from label_detections.core.imageio import rectify_quad

        crops = []
        for quad in self._detection_quads(results):
            patch = rectify_quad(frame, expand_quad(quad, self._margin))
            if patch is None or patch.size == 0:
                crops.append(None)
                continue
            crops.append(letterbox(patch, self._crop_px))

        usable = [c for c in crops if c is not None]
        if not usable:
            return []
        try:
            preds = self._classifier.predict(
                usable, imgsz=self._crop_px, verbose=False,
                **({"device": self._device} if self._device is not None else {}))
        except Exception as exc:
            self.failed.emit(f"Classifier failed on a frame: {exc}")
            return []

        named = []
        for pred in preds:
            probs = getattr(pred, "probs", None)
            names = getattr(pred, "names", {}) or {}
            if probs is None:
                named.append((logic.UNKNOWN, 0.0))
                continue
            try:
                top, conf = int(probs.top1), float(probs.top1conf)
            except Exception:
                named.append((logic.UNKNOWN, 0.0))
                continue
            named.append(logic.identify(str(names.get(top, "")), conf,
                                        self._identity_floor))

        # Put them back against the original detections, including the ones
        # whose crop failed -- position is what ties identity to box.
        out, i = [], 0
        for crop in crops:
            if crop is None:
                out.append((logic.UNKNOWN, 0.0))
            else:
                out.append(named[i] if i < len(named) else (logic.UNKNOWN, 0.0))
                i += 1
        return out

    @Slot()
    def stop(self) -> None:
        self._stopping = True
        self._model = None
        self._classifier = None
