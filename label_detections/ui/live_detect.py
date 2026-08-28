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


def _as_array(obj, attr):
    """An Ultralytics tensor attribute as numpy, or [] when it is not there."""
    import numpy as np

    value = getattr(obj, attr, None)
    if value is None:
        return []
    try:
        return value.cpu().numpy()
    except AttributeError:
        pass
    except Exception:
        return []
    try:
        return np.asarray(value)
    except Exception:
        return []


def _class_name(names, cls_id: int) -> str:
    """Ultralytics exposes names as a dict or a list depending on version."""
    try:
        if isinstance(names, dict):
            return str(names.get(cls_id, f"class_{cls_id}"))
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return str(names[cls_id])
    except Exception:
        pass
    return f"class_{cls_id}"


def extract_items(results) -> list[dict]:
    """Plain dicts from an Ultralytics result. No tensors survive this.

    Runs on the worker thread, deliberately. Emitting the Results object itself
    put CUDA tensors on a Qt queued signal and left the GUI thread to call
    .cpu() on them -- GPU work on the thread that paints, and torch objects
    crossing a thread boundary they were never promised to cross. Everything
    the UI needs is a handful of floats and a name, so only those cross now.
    """
    import numpy as np

    # The drawn plate is the label id and its confidence, nothing else. The
    # track id is still carried on the item -- the readout groups by it -- but
    # on the box it only competes with the two things worth reading while parts
    # are moving past.
    items: list[dict] = []
    for r in results or []:
        names = getattr(r, "names", {}) or {}

        obb = getattr(r, "obb", None)
        if obb is not None:
            polys = _as_array(obb, "xyxyxyxy")
            confs = _as_array(obb, "conf")
            clss = _as_array(obb, "cls")
            ids = _as_array(obb, "id")
            if len(polys):
                for i, poly in enumerate(polys):
                    pts = np.array(poly, dtype=float).reshape(-1, 2)[:4]
                    if len(pts) < 4:
                        continue
                    cls_id = int(clss[i]) if i < len(clss) else 0
                    name = _class_name(names, cls_id)
                    conf = float(confs[i]) if i < len(confs) else 0.0
                    track_id = int(ids[i]) if i < len(ids) else None
                    items.append({
                        "type": "other_obb", "track_id": track_id,
                        "points": [[float(x), float(y)] for x, y in pts],
                        "cx": float(np.mean(pts[:, 0])),
                        "cy": float(np.mean(pts[:, 1])),
                        "conf": conf, "cls_id": cls_id, "name": name,
                        "label": f"{name} {conf:.2f}",
                    })
                continue

        boxes = getattr(r, "boxes", None)
        if boxes is None:
            continue
        xyxy = _as_array(boxes, "xyxy")
        confs = _as_array(boxes, "conf")
        clss = _as_array(boxes, "cls")
        ids = _as_array(boxes, "id")
        for i, box in enumerate(xyxy):
            cls_id = int(clss[i]) if i < len(clss) else 0
            name = _class_name(names, cls_id)
            x1, y1, x2, y2 = (float(v) for v in box[:4])
            conf = float(confs[i]) if i < len(confs) else 0.0
            track_id = int(ids[i]) if i < len(ids) else None
            items.append({
                "type": "other_box", "track_id": track_id,
                "xyxy": [x1, y1, x2, y2],
                "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                "conf": conf, "cls_id": cls_id, "name": name,
                "label": f"{name} {conf:.2f}",
            })
    return items


class InferenceWorker(QObject):
    """Owns the model and runs it. Lives on its own thread."""

    loaded = Signal(str)          # human description of what came up
    failed = Signal(str)
    # Plain dicts (never tensors), latency in seconds, and Ultralytics' own
    # per-phase timings. Identities from stage 2 are already merged in.
    result = Signal(object, float, object)

    def __init__(self, model_path: str, imgsz: int, conf: float, device,
                 track: bool = True, tracker: str = "bytetrack.yaml",
                 classifier_path: str = "",
                 crop_px: int = 224, margin: float = 0.06,
                 identity_floor: float = 0.55, warm_shape=None):
        super().__init__()
        self._path = str(model_path)
        self._imgsz = int(imgsz)
        self._conf = float(conf)
        self._device = device
        self._track = bool(track)
        self._tracker = str(tracker or "bytetrack.yaml")
        self._classifier_path = str(classifier_path or "")
        self._crop_px = int(crop_px)
        self._margin = float(margin)
        self._identity_floor = float(identity_floor)
        # The shape of a real camera frame, so the warm-up exercises the size
        # the live path will hand over rather than a convenient small square.
        self._warm_shape = tuple(warm_shape) if warm_shape else None
        self._model = None
        self._classifier = None
        self._stopping = False
        # One frame deep, newest wins. The busy flag upstream already prevents
        # a backlog; this makes it structural rather than a convention.
        import queue as _queue
        self._queue = _queue.Queue(maxsize=1)
        # Reported once, not once per frame: a failure that repeats at the
        # camera's rate buries every other message in the log.
        self._stage2_failed = False
        # Novelty rejection: where each enrolled class sits in feature space,
        # so a label that was never enrolled can be refused instead of being
        # named as whichever one it least differs from. Absent is a supported
        # state -- it is what every model built before this is in -- and the
        # readout says so rather than the check silently not running.
        self._novelty = None
        self._novelty_note = ""
        self._embedder = None
        self._novelty_failed = False
        # Where Ultralytics actually ran, asked once after the first inference.
        self._device_checked = False

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
        # Move the model onto the device NOW rather than leaving it to the
        # first predict call. Ultralytics takes device= per call, so a model
        # loaded here sits on the CPU until inference happens -- which made the
        # device report true at a useless moment ("model on cpu; asked for 0")
        # and, worse, left it genuinely on the CPU whenever the per-call move
        # did not take. Doing it here means the report describes what will
        # actually run, and a failure surfaces at load with a name on it
        # instead of as a mysteriously slow model.
        self._device_error = ""
        target = self._torch_device()
        if target:
            try:
                self._model.to(target)
            except Exception as exc:
                self._device_error = f"Could not move the detector to {target}: {exc}"

        if self._classifier_path:
            try:
                self._classifier = YOLO(self._classifier_path)
                # Same device as the detector, deliberately: two stages of one
                # pipeline on two devices would pay a host round-trip per crop.
                if target:
                    try:
                        self._classifier.to(target)
                    except Exception as exc:
                        self._device_error += (
                            f" Could not move the classifier to {target}: {exc}")
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
        if self._classifier is not None:
            self._load_novelty()
        # Build Ultralytics' predictor HERE, on this thread, before any frame
        # arrives. It is created lazily on the first predict/track call, and
        # setting it up runs select_device -> torch.cuda.set_device.
        #
        # That is not a detail. With inference called directly it happened on
        # the GUI thread and was merely a slow first frame. Moved to this
        # thread it deadlocked: a py-spy dump of a frozen session showed this
        # worker stopped inside torch.cuda.set_device, reached through
        # setup_model on the first track() call, and never coming back --
        # detections never started and Stop then blocked on a thread that could
        # not finish.
        #
        # Doing it during load costs nothing that was not already going to be
        # paid, removes the first-frame latency spike, and moves the risky call
        # into the one place already expected to take time, where the status
        # reads "Loading the model..." and the window stays live.
        # Before the warm-up, so the very first inference on this thread runs
        # the same way every later one will.
        self._timing_note = self.unsynchronised_timing()
        self._warm_up()

        which = ("detector + classifier" if self._classifier is not None
                 else f"{task or 'detector'} only, no stage 2")
        # The novelty state goes in the load message rather than only in a
        # settings dialog. "Off" and "on" look identical while every part in
        # front of the camera happens to be enrolled, and the run where they
        # stop being is the run where nobody remembers which it was.
        note = f"\n{self._novelty_note}" if self._novelty_note else ""
        self.loaded.emit(
            f"Loaded {which}: {self._path}\n{self._device_report()}{note}")

    @staticmethod
    def unsynchronised_timing() -> str:
        """Stop Ultralytics' profiler synchronising the accelerator.

        Aimed at the exact frame three py-spy dumps agree on:

            torch/cuda/__init__.py:604   _exchange_device      <- stopped here
            torch/cuda/__init__.py:1161  synchronize
            ultralytics/utils/ops.py:73  Profile.time
            ultralytics/utils/ops.py:58  Profile.__enter__

        Profile calls synchronize so that the milliseconds it reports are the
        GPU's rather than the queue's. That is a stopwatch, not the work: the
        model's forward pass does not switch device, and a synchronize for
        timing is the only thing in the live path that does. On this machine --
        a 5090 on WDDM also driving the desktop -- that call stops on the
        worker thread and does not return, while the same call on the same
        thread during warm-up completes.

        The cost is honest and worth stating: the per-phase numbers become
        wall-clock around asynchronous CUDA work, so they measure when calls
        were issued rather than when the GPU finished. Total latency is
        measured here and is unaffected. The readout says which it is showing.

        Returns "" when applied, or a reason when the library no longer looks
        the way this expects -- a patch that silently stops applying is worse
        than one that never did.
        """
        try:
            from ultralytics.utils import ops
        except Exception as exc:
            return f"Ultralytics' ops module could not be imported: {exc}"
        profile = getattr(ops, "Profile", None)
        if profile is None or not hasattr(profile, "time"):
            return "Ultralytics' Profile no longer looks the way this expects."
        if getattr(profile, "_lv_unsynchronised", False):
            return ""
        import time as _time

        def _time_without_sync(self):
            return _time.perf_counter()

        profile.time = _time_without_sync
        profile._lv_unsynchronised = True
        return ""

    def _warm_up(self) -> None:
        """One inference here, made as close to the live call as it can be.

        Identical on purpose, and it was not before. The first version warmed
        with a blank 640x640 and persist=False, and that succeeded on this
        thread -- then the first real frame, 5496x3672 with persist=True,
        stopped inside a CUDA device switch and never came back. Two
        differences between a call that works and a call that hangs is one too
        many to reason about.

        So: the camera's real frame size, and the same persist the live path
        uses. If the hang follows the frame size it now happens HERE, during
        load, where the window is still live and Stop still works. If it does
        not, the difference is something about the frame that arrives rather
        than the frame itself, which is a far more specific thing to chase.

        Failure is reported, not raised: the first real frame may still work.
        """
        if self._model is None:
            return
        import numpy as np

        shape = self._warm_shape or (max(32, int(self._imgsz)),
                                     max(32, int(self._imgsz)), 3)
        blank = np.zeros(shape, dtype=np.uint8)
        args = {"imgsz": self._imgsz, "conf": self._conf, "verbose": False}
        if self._device is not None:
            args["device"] = self._device
        try:
            if self._track:
                self._model.track(blank, persist=True, tracker=self._tracker,
                                  **args)
            else:
                self._model.predict(blank, **args)
        except Exception as exc:
            self.failed.emit(
                f"The model loaded but its first run failed: "
                f"{type(exc).__name__}: {exc}")
            return
        if self._classifier is not None:
            try:
                crop = np.zeros((self._crop_px, self._crop_px, 3), dtype=np.uint8)
                self._classifier.predict(
                    [crop], imgsz=self._crop_px, verbose=False,
                    **({"device": self._device} if self._device is not None else {}))
            except Exception as exc:
                self.failed.emit(
                    f"The classifier loaded but its first run failed: "
                    f"{type(exc).__name__}: {exc}")

    def _torch_device(self) -> str:
        """The device string torch wants, from what the UI collected.

        The UI hands over 0 / "cpu" / "cuda:0". torch.Module.to needs a string
        or a torch.device, and a bare 0 means "CPU 0" to it, not "GPU 0" -- the
        exact inversion that puts a model on the CPU while the field says 0.
        """
        dev = self._device
        if dev is None or dev == "":
            return ""
        if isinstance(dev, int):
            return f"cuda:{dev}"
        text = str(dev).strip()
        if text.isdigit():
            return f"cuda:{text}"
        return text

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
        asked = self._torch_device() or "(unset)"
        both = " (detector and classifier)" if self._classifier is not None else ""
        line = (f"Device: CUDA available ({name}); model on {where}; "
                f"asked for {asked}{both}.")
        if getattr(self, "_device_error", ""):
            line += "\n" + self._device_error
        elif where.startswith("cpu"):
            line += (" On the CPU anyway, which is why inference is slow. If "
                     "Device is already 0, this torch build cannot run this "
                     "card -- a 5090 is Blackwell and needs a CUDA 12.8+ wheel.")
        return line

    @Slot(object)
    def submit(self, frame) -> None:
        """Hand a frame to the worker thread. Never blocks the caller.

        A queue rather than a queued signal, and the thread is a plain
        threading.Thread rather than a QThread, because of where four py-spy
        dumps put the hang.

        QThread emits started() at the top of run(), BEFORE exec(). So load()
        and its warm-up ran outside the event loop, and every CUDA call in them
        completed. infer() arrived as a queued slot INSIDE exec() -- and
        QThread::exec on Windows runs a QEventDispatcherWin32, which owns a
        hidden window and pumps Windows messages. A blocking CUDA call made
        from inside a message-pump dispatch, on a WDDM GPU that is also driving
        the desktop, is a known deadlock shape: the driver can need that thread
        to pump while the thread is sitting inside the driver.

        Every observation fits that and nothing else did. It is also why the
        camera reader -- a plain thread, no pump, no window -- has never had
        this problem, and why each fix only moved the hang to the next CUDA
        call rather than removing it.

        So this thread has no Qt event loop at all. Results still travel back
        as Qt signals, which is safe in that direction: the GUI thread has the
        event loop, and delivering to it is what event loops are for.
        """
        if self._stopping:
            return
        try:
            self._queue.put_nowait(frame)
        except Exception:
            # Full: a frame is already waiting. Dropping the newer one would
            # show older boxes, so replace it.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(frame)
            except Exception:
                pass

    def run_forever(self) -> None:
        """Load, then take frames off the queue until asked to stop.

        The whole life of this thread, with no event loop anywhere in it.
        """
        self.load()
        while not self._stopping:
            try:
                frame = self._queue.get(timeout=0.2)
            except Exception:
                continue
            if frame is None or self._stopping:
                break
            self.infer(frame)

    def infer(self, frame) -> None:
        """Run one frame. Silently drops it if the model never loaded."""
        if self._model is None or self._stopping or frame is None:
            return
        args = {"imgsz": self._imgsz, "conf": self._conf, "verbose": False}
        if self._device is not None:
            args["device"] = self._device
        started = time.perf_counter()
        try:  # noqa: SIM105 -- the except below reports rather than passes
            if self._track:
                # persist=True is what carries the tracker's state between
                # calls; without it every frame starts a fresh tracker and each
                # object is "new" forever, which is the same as not tracking.
                #
                # bytetrack, not the default botsort. BoT-SORT runs global
                # motion compensation -- sparse optical flow over the WHOLE
                # frame, every frame -- to cancel out camera movement. On a
                # 20 MP frame that is the entire cost of tracking, and it buys
                # nothing here: the camera is bolted down. Turning tracking on
                # took inference from single-digit milliseconds to 120, and all
                # of it was this.
                results = self._model.track(frame, persist=True,
                                            tracker=self._tracker, **args)
            else:
                results = self._model.predict(frame, **args)
        except Exception as exc:
            # One bad frame must not take the view down; the readout says so and
            # the next frame gets its own try. Typed, because a bare str() of an
            # exception is frequently empty and reads as a failure with no cause.
            self.failed.emit(f"Inference failed: {type(exc).__name__}: {exc}")
            return
        # Stage 2 must never take stage 1 down with it. It was called bare, so
        # one exception anywhere in cropping or classifying propagated out of
        # the slot, Qt swallowed it, and no result was emitted at all -- the
        # view sat on its placeholder with a model loaded and nothing to say.
        # A failed classifier now costs identities, not detections.
        identities = []
        # Timed separately, because the readout was reporting a total that
        # covered stage 2 next to a phase breakdown that did not -- so a
        # detector doing 12 ms inside a 71 ms call looked like a slow detector.
        stage2_started = time.perf_counter()
        try:
            identities = self._identify(frame, results)
        except Exception as exc:
            if not self._stage2_failed:
                self._stage2_failed = True
                self.failed.emit(
                    f"Stage 2 failed, so boxes will have no identity: "
                    f"{type(exc).__name__}: {exc}\n\n"
                    f"Stage 1 detection continues.")
        stage2_ms = (time.perf_counter() - stage2_started) * 1000.0

        # Everything torch-shaped is converted here, on this thread, before
        # anything crosses back. The GUI never sees a tensor. Timed too: the
        # .cpu() calls in here force a CUDA sync, so any GPU work the detector
        # deferred is billed at this line and nowhere else.
        readout_started = time.perf_counter()
        try:
            from label_detections.core import live_detect as logic
            items = logic.apply_identities(extract_items(results), identities)
        except Exception as exc:
            self.failed.emit(f"Could not read the results: {type(exc).__name__}: {exc}")
            return
        readout_ms = (time.perf_counter() - readout_started) * 1000.0
        # Ultralytics already times its own three phases and we were throwing
        # the numbers away. "120 ms" is not actionable; "preprocess 95,
        # inference 8, postprocess 3" says immediately that the GPU is idle and
        # the cost is resizing a 20 MP frame on the CPU.
        # Ultralytics builds its own predictor on the first call, with its own
        # device. The model object reporting cuda:0 says nothing about where
        # the predictor ended up -- and a YOLO11 OBB at 640 costing 120 ms on a
        # current card is CPU-speed, not GPU-speed. Checked once, reported once.
        if not self._device_checked:
            self._device_checked = True
            try:
                where = str(getattr(getattr(self._model, "predictor", None),
                                    "device", "") or "")
                wanted = self._torch_device()
                if where and wanted and where.startswith("cpu") \
                        and not wanted.startswith("cpu"):
                    self.failed.emit(
                        f"The model object is on {wanted}, but Ultralytics is "
                        f"running inference on {where}.\n\n"
                        f"That is why it is slow: this is CPU inference. The "
                        f"usual cause is a torch build without kernels for this "
                        f"card -- a 5090 is Blackwell (sm_120) and needs a "
                        f"CUDA 12.8+ wheel.")
                elif where:
                    self.loaded.emit(f"Inference is running on {where}.")
            except Exception:
                pass

        speed = {}
        try:
            raw = getattr(results[0], "speed", None) if results else None
            if isinstance(raw, dict):
                speed = {k: float(v) for k, v in raw.items()
                         if isinstance(v, (int, float))}
        except Exception:
            speed = {}
        # Ours, alongside Ultralytics'. Named differently so nothing mistakes
        # them for the library's own three.
        speed["stage2"] = stage2_ms
        speed["readout"] = readout_ms
        # So the readout can say the phase numbers are unsynchronised rather
        # than let them be read as GPU time.
        speed["_unsynced"] = 1.0 if not getattr(self, "_timing_note", "x") else 0.0
        self.result.emit(items, time.perf_counter() - started, speed)

    def _detection_quads(self, results):
        """Four corners per detection, in the SAME order the overlay builds them.

        Order is the coupling that makes identities line up with boxes, so it
        mirrors _detection_overlay_items exactly: oriented results first when
        present, otherwise axis-aligned, each in index order. A mismatch is
        caught downstream by length and degrades to no identity rather than to
        a wrong one.
        """
        import numpy as np

        def as_array(obj, attr):
            """An Ultralytics tensor attribute as numpy, or [] if absent.

            The same tolerance _safe_np already had on the overlay path, and
            for the same reason: demanding .cpu() means anything already
            array-like returns [] instead, and an empty list here does not
            error -- it silently identifies nothing, forever, on an OBB model.
            This function was written after that fix and repeated the bug.
            """
            value = getattr(obj, attr, None)
            if value is None:
                return []
            try:
                return value.cpu().numpy()
            except AttributeError:
                pass
            except Exception:
                return []
            try:
                return np.asarray(value)
            except Exception:
                return []

        quads = []
        for r in results or []:
            obb = getattr(r, "obb", None)
            if obb is not None:
                polys = as_array(obb, "xyxyxyxy")
                if len(polys):
                    for poly in polys:
                        pts = np.array(poly, dtype=float).reshape(-1, 2)[:4]
                        if len(pts) >= 4:
                            quads.append(pts.tolist())
                    continue
            boxes = getattr(r, "boxes", None)
            if boxes is None:
                continue
            xyxy = as_array(boxes, "xyxy")
            for box in xyxy:
                x1, y1, x2, y2 = (float(v) for v in box[:4])
                quads.append([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return quads

    def _load_novelty(self) -> None:
        """Bring up the novelty profile for this classifier, if it has one.

        Never fatal. Running without one is exactly what every model built
        before this feature does, and the readout names the state so it is a
        visible gap rather than a check that looks on and is not.
        """
        from ..core import novelty as nv

        self._novelty = nv.Profile.load(nv.profile_path(self._classifier_path))
        if self._novelty is None or not len(self._novelty):
            self._novelty = None
            self._novelty_note = "no novelty profile"
            return
        from .novelty import Embedder

        embedder = Embedder()
        if not embedder.attach(self._classifier):
            self._novelty = None
            self._novelty_note = f"novelty off: {embedder.reason}"
            return
        self._embedder = embedder
        enforced = self._novelty.enforced_classes
        if not enforced:
            self._novelty_note = ("novelty profile covers no class with enough "
                                  "crops to enforce")
        else:
            self._novelty_note = f"novelty: {len(enforced)} class(es) enforced"

    def _reject_unknown(self, named, crops_used):
        """Replace any identity whose crop does not sit where its class sits.

        Fails open, loudly. A mismatch between vectors and predictions means
        the pairing is not trustworthy, and rejecting on an untrustworthy
        pairing would mark good parts unknown for a reason nobody can see. The
        message says the check is not running, which is the visible half of the
        same failure.
        """
        if self._novelty is None or self._embedder is None:
            return named
        vectors = self._embedder.take()
        if len(vectors) != crops_used:
            if not self._novelty_failed:
                self._novelty_failed = True
                self.failed.emit(
                    f"Novelty checking is NOT running: read {len(vectors)} "
                    f"feature vector(s) for {crops_used} crop(s).\n\n"
                    f"Identities are the classifier's own answers, so a label "
                    f"that was never enrolled will be named as the closest one "
                    f"that was.")
            return named
        from ..core import live_detect as logic

        out = []
        for (name, conf), vec in zip(named, vectors):
            if name == logic.UNKNOWN:
                out.append((name, conf))
                continue
            verdict = self._novelty.verdict(name, vec)
            out.append((name, conf) if verdict.known else (logic.UNKNOWN, conf))
        return out

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

        # Capped at twice the crop, never at native size. warpPerspective costs
        # what its destination costs, so flattening a 2000 px label in full and
        # then letterboxing it down to 224 paid for 2000 px of warp to keep 224
        # -- measured at 17.8 ms for four crops on a 5496x3672 frame, against
        # 2.4 ms this way. Twice rather than exactly the crop size because
        # letterbox finishes with INTER_AREA, and that averaging is what keeps
        # printed text legible; warping straight to 224 would alias it.
        crops = []
        cap = max(1, int(self._crop_px) * 2)
        for quad in self._detection_quads(results):
            patch = rectify_quad(frame, expand_quad(quad, self._margin),
                                 max_side=cap)
            if patch is None or patch.size == 0:
                crops.append(None)
                continue
            crops.append(letterbox(patch, self._crop_px))

        usable = [c for c in crops if c is not None]
        if not usable:
            return []
        # Anything left from a previous frame would pair with this frame's
        # crops and put one label's identity on another's box.
        if self._embedder is not None:
            self._embedder.clear()
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

        # Second opinion, from the layer under the softmax. The head can only
        # elect one of the classes it was given, so a label that was never
        # enrolled comes back as the nearest enrolled one at ~1.00; this asks
        # whether the crop sits where that class actually sits.
        named = self._reject_unknown(named, len(usable))

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
        """Ask the worker to stop. Sets a flag and nothing else.

        It used to drop the models here, and it is called from the GUI thread.
        While inference ran on the GUI thread too that was harmless; once it
        genuinely moved to this worker's thread, it became freeing a torch
        model out from under a running forward pass -- which exits the process
        with no Python traceback at all.

        The models are released when this object is dropped, which the owner
        only does after the thread has actually finished.
        """
        self._stopping = True
        try:
            self._queue.put_nowait(None)      # wake the loop out of its wait
        except Exception:
            pass
