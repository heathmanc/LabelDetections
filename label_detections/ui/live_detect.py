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
    result = Signal(object, float)   # ultralytics results, latency in seconds

    def __init__(self, model_path: str, imgsz: int, conf: float, device,
                 track: bool = True):
        super().__init__()
        self._path = str(model_path)
        self._imgsz = int(imgsz)
        self._conf = float(conf)
        self._device = device
        self._track = bool(track)
        self._model = None
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
        self.loaded.emit(f"Loaded {self._path}")

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
        self.result.emit(results, time.perf_counter() - started)

    @Slot()
    def stop(self) -> None:
        self._stopping = True
        self._model = None
