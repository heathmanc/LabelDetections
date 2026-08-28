"""Policy for the live detection view: pacing, and when a frame is worth keeping.

This view exists to answer one question a saved-image test cannot: does the
model work *through this camera, at this standoff, under this light*. It is
deliberately not an inspection runtime -- no verdicts, no latching, no reject
output. Which labels a battery must carry is the front end's business, and a
second half-built HMI living in the labeling tool would be the worst of both.

What it adds instead is the loop back into labeling. When the model does badly
on a live frame, that frame is the most valuable training image available, and
one press -- or, if armed, no press at all -- puts it in the dataset.

Stdlib only: the decisions are here and testable, the pixels are in the UI.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

# Inference is skipped while a previous frame is still in flight, so this is a
# floor on how often it starts rather than a target rate. Well under a camera's
# frame interval: the preview must never wait on the model.
# A floor on how often inference may START.
#
# Back to 0.15, the value this branch inherited and ran stably on. Lowering it
# to 0.01 was defensible in isolation -- 6.7/s is a hard ceiling on any
# hardware -- but it did not happen in isolation: with inference also moved off
# the GUI thread, the display tick stopped being blocked by it and went from
# ~7 to ~60 iterations a second, so everything downstream, the camera included,
# was driven nine times harder. That is what destabilised a camera path that
# had been fine.
#
# It is a setting now rather than a constant, so the rate can be raised
# deliberately and watched, instead of being raised for everyone by someone
# who could not test it against the hardware.
MIN_INTERVAL_S = 0.15

# How long to wait for a result before assuming it is never coming. The busy
# flag is cleared by the result, so a result that never arrives -- a worker
# exception Qt swallowed, a signal dropped on shutdown -- freezes the view
# permanently with no error and no way back but a restart.
BUSY_TIMEOUT_S = 5.0

# How badly the model has to do before an armed capture keeps the frame.
# Scored by active_learning.disagreement_score, where a clean single detection
# is 0 and a complete miss is 10.
CAPTURE_THRESHOLD = 4.0

# Nothing captures twice inside this. One battery sitting in front of a camera
# that struggles with it would otherwise produce hundreds of near-identical
# frames, which is worse than no frames: they all say the same thing and they
# each cost a review.
CAPTURE_COOLDOWN_S = 3.0

# A session limit, so walking away from an armed view does not fill a disk.
CAPTURE_SESSION_LIMIT = 50


@dataclass
class Rolling:
    """Recent inference timings, for a readout that is not a single sample."""
    window: int = 30
    latencies: deque = field(default_factory=lambda: deque(maxlen=30))
    stamps: deque = field(default_factory=lambda: deque(maxlen=30))

    def record(self, latency_s: float, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.latencies.append(float(latency_s))
        self.stamps.append(float(now))

    @property
    def mean_ms(self) -> float:
        return 1000.0 * sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def rate(self) -> float:
        """Inferences per second across the window, not one over the latency.

        They differ whenever frames are skipped, and the honest number is the
        one that counts the skips.
        """
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 0 else 0.0

    def reset(self) -> None:
        self.latencies.clear()
        self.stamps.clear()


@dataclass
class CaptureGate:
    """Decides whether an armed live view should keep the current frame."""
    threshold: float = CAPTURE_THRESHOLD
    cooldown_s: float = CAPTURE_COOLDOWN_S
    limit: int = CAPTURE_SESSION_LIMIT
    captured: int = 0
    _last: float = -1e9

    def consider(self, score: float, now: float | None = None) -> tuple[bool, str]:
        """``(capture?, why)``. The reason is shown, so it must be worth reading."""
        now = time.monotonic() if now is None else now
        if self.captured >= self.limit:
            return False, f"session limit reached ({self.limit})"
        if float(score) < self.threshold:
            return False, "model is handling this frame"
        if now - self._last < self.cooldown_s:
            remaining = self.cooldown_s - (now - self._last)
            return False, f"cooling down ({remaining:.1f}s)"
        return True, f"model struggled here (score {float(score):.1f})"

    def mark(self, now: float | None = None) -> None:
        self._last = time.monotonic() if now is None else now
        self.captured += 1

    def reset(self) -> None:
        self.captured = 0
        self._last = -1e9


def should_infer(busy: bool, since_last_s: float,
                 min_interval_s: float = MIN_INTERVAL_S) -> bool:
    """Start another inference, or let the preview have the frame?

    Skipping while busy is what keeps the live view live: the camera renders at
    its own rate and overlays refresh whenever the model finishes, rather than
    the preview stuttering along at the model's pace.
    """
    return not busy and since_last_s >= float(min_interval_s)


# The detector is trained on label ids, so what it returns *is* the identity
# the recipe is written in. The readout says it plainly.
# After this many consecutive empty frames, silence is a symptom rather than
# an absence, and the readout should say what usually causes it.
QUIET_FRAMES = 12


def quiet_hint(empty_frames: int, conf: float, imgsz: int,
               has_classifier: bool) -> str:
    """Why a running model might be showing nothing at all.

    A live view that finds nothing looks identical whether the camera is
    pointed at a wall, the threshold is too high, or a classifier was loaded
    into the detector slot. The operator cannot tell those apart from an empty
    screen, so the readout names the usual causes rather than leaving them to
    be guessed one at a time.
    """
    if empty_frames < QUIET_FRAMES:
        return ""
    lines = [f"", f"Nothing found in {empty_frames} frames. Usual causes:",
             f"  - Confidence is {conf:.2f}. A fresh model often needs 0.25 to "
             f"show anything.",
             f"  - Image size {imgsz} should match what the model trained at.",
             "  - Is the model in the Test Models field the DETECTOR run, not "
             "the classifier?"]
    if not has_classifier:
        lines.append("  - Under a two-stage export, boxes read 'label' until a "
                     "stage 2 classifier is set on this tab.")
    return "\n".join(lines)


def throughput_note(rolling: "Rolling", interval_s: float = MIN_INTERVAL_S,
                    camera_fps: float = 0.0) -> str:
    """Say when the rate is far below what the measured latency allows, and
    name the thing actually holding it back.

    It used to quote MIN_INTERVAL_S, the module default, rather than the
    interval in force -- so with the rate set to 30/s it still reported a
    "150 ms start floor" that had not applied for some time. And it offered
    the camera and the floor as alternatives without saying which, when the
    numbers to tell them apart are right there.
    """
    if not rolling.mean_ms or rolling.rate <= 0:
        return ""
    possible = 1000.0 / rolling.mean_ms
    if possible <= rolling.rate * 1.8:
        return ""

    floor_fps = 1.0 / interval_s if interval_s > 0 else float("inf")
    # Whichever ceiling is lowest is the one in force.
    if camera_fps and camera_fps <= floor_fps and camera_fps < possible:
        cause = f"the camera, at {camera_fps:.0f}/s"
    elif floor_fps < possible:
        cause = f"the {interval_s * 1000:.0f} ms start floor ({floor_fps:.0f}/s)"
    else:
        cause = "something upstream of the model"
    return f"   (model could run ~{possible:.0f}/s; the limit is {cause}, not the GPU)"


def frame_summary(counts: dict[str, int], label_id: str,
                  rolling: Rolling) -> str:
    """The readout under the live view."""
    total = sum(counts.values())
    lines = [f"{total} detection(s)   {rolling.mean_ms:.0f} ms   "
             f"{rolling.rate:.1f}/s"]
    # Only when it is actually there. The label open in the labeling tab is
    # not what the operator is presenting -- a line saying it is missing is a
    # complaint about the wrong thing, on every frame of every other label.
    if label_id and counts.get(label_id):
        lines.append(f"{label_id}: {counts[label_id]} found")
    for name in sorted(counts):
        if name != label_id:
            lines.append(f"  {name}: {counts[name]}")
    return "\n".join(lines)


def capture_note(captured: int, limit: int, last_reason: str) -> str:
    """One line of state for an armed view, so it never looks like it hung."""
    if not last_reason:
        return f"Armed — {captured}/{limit} kept this session."
    return f"Armed — {captured}/{limit} kept. Last: {last_reason}."


# --- tracking --------------------------------------------------------------

# A track that has not been seen for this long is dropped from the readout.
# Long enough to survive a few missed frames on a struggling detection, short
# enough that a battery taken away stops being listed.
TRACK_TTL_S = 2.0


@dataclass
class Track:
    """What one tracked object has done since it appeared.

    Per-frame confidence flickers -- the same label reads 0.91, 0.87, 0.94 on
    consecutive frames -- and reading a single number off a moving overlay tells
    you almost nothing. A track that has been held for sixty frames at a mean of
    0.91 tells you the model has it; one that keeps being lost and re-acquired
    under a new id tells you it does not, which is the failure the single number
    hides completely.
    """
    track_id: int
    name: str
    frames: int = 0
    last_conf: float = 0.0
    min_conf: float = 1.0
    max_conf: float = 0.0
    _sum: float = 0.0
    last_seen: float = 0.0

    def record(self, conf: float, now: float) -> None:
        conf = float(conf)
        self.frames += 1
        self._sum += conf
        self.last_conf = conf
        self.min_conf = min(self.min_conf, conf)
        self.max_conf = max(self.max_conf, conf)
        self.last_seen = now

    @property
    def mean_conf(self) -> float:
        return self._sum / self.frames if self.frames else 0.0


class TrackBook:
    """Per-track history across frames, pruned as objects leave."""

    def __init__(self, ttl_s: float = TRACK_TTL_S):
        self.ttl_s = float(ttl_s)
        self._tracks: dict[int, Track] = {}
        # Counted rather than inferred from the id: trackers reuse ids, and
        # "how many times did it lose and re-acquire" is the number that says
        # whether the model actually holds the object.
        self.reacquired = 0

    def update(self, detections, now: float | None = None) -> None:
        """``detections`` are ``(track_id, name, confidence)``; ids may be None."""
        now = time.monotonic() if now is None else now
        for track_id, name, conf in detections:
            if track_id is None:
                continue
            key = int(track_id)
            track = self._tracks.get(key)
            if track is None:
                track = Track(track_id=key, name=str(name))
                self._tracks[key] = track
            elif track.name != str(name):
                # The tracker kept the id but the classifier changed its mind.
                # That is a real thing to see, so the track restarts under the
                # new name rather than averaging two classes together.
                self.reacquired += 1
                track = Track(track_id=key, name=str(name))
                self._tracks[key] = track
            track.record(conf, now)
        self.prune(now)

    def prune(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        for key in [k for k, t in self._tracks.items() if now - t.last_seen > self.ttl_s]:
            del self._tracks[key]

    def rows(self) -> list[Track]:
        """Longest-held first: the stable objects are the ones worth reading."""
        return sorted(self._tracks.values(), key=lambda t: (-t.frames, t.track_id))

    def reset(self) -> None:
        self._tracks.clear()
        self.reacquired = 0

    def text(self) -> str:
        rows = self.rows()
        if not rows:
            return "No tracked objects."
        return "\n".join(track_line(t) for t in rows)


# One line per tracked object: which one it is, and how sure the model is.
# The frame counts and the min/max spread the readout used to carry were the
# evidence *for* the mean rather than the answer, and they pushed the two
# numbers an operator is actually reading off the end of the line. They are
# still recorded -- rows() orders by them, and a mean is only worth reading
# because it is held over frames -- just not printed.
def track_line(track: Track) -> str:
    # The id and the confidence, nothing else -- the same rule the drawn plate
    # follows. The track number is still the key the book groups by; it just is
    # not something anyone reads while parts are moving past.
    return f"{track.name} {track.mean_conf:.2f}"


def track_summary(book: TrackBook, label_id: str, rolling: Rolling) -> str:
    """The readout when tracking is on."""
    rows = book.rows()
    lines = [f"{len(rows)} tracked   {rolling.mean_ms:.0f} ms   "
             f"{rolling.rate:.1f}/s"]
    # Marked when present, silent when not: the open label is not necessarily
    # the one in front of the camera, and "NOT TRACKED" both named the wrong
    # mechanism -- nothing had lost a track -- and filled the readout with a
    # complaint about a label nobody was presenting.
    mine = [t for t in rows if t.name == label_id] if label_id else []
    for track in rows:
        # Marked rather than filtered: the other labels on the battery are
        # what the recipe counts too, so they stay visible.
        mark = "  <-- this label" if (mine and track is mine[0]) else ""
        lines.append(track_line(track) + mark)
    if not rows:
        lines.append("No tracked objects.")
    return "\n".join(lines)


# --- keeping a frame, with or without what the model proposed --------------
#
# Two ways to keep a live frame, and the difference matters more than it looks.
#
# Image only is right when the model got it *wrong*. Pre-filling boxes a human
# is about to correct anchors them: a labeler nudges a bad box far more often
# than they delete it and draw the right one, so wrong proposals do not just
# waste time, they leak into the dataset as slightly-wrong truth.
#
# Image plus proposals is right when the model is mostly there. Correcting
# four boxes is minutes of work; drawing them is not.
#
# Either way the sidecar written here is **never** review-marked. A machine
# proposal that could pass for an operator's approval is the one failure this
# whole marker discipline exists to prevent.

PROPOSED_BY = "live_detect"


def proposal_session(started_at: float) -> str:
    """The capture-session id for one live-detect run.

    Frames kept from a single run are near-duplicates of each other -- same
    lens, same light, seconds apart -- so they must not straddle the train/val
    split. Stamping them with one session is what makes the group-aware split
    keep them together.
    """
    return "live_" + time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))


def _item_points(item: dict) -> list[list[float]]:
    """Four corners for a detection, whether it came back oriented or not."""
    points = item.get("points")
    if points and len(points) >= 4:
        return [[float(p[0]), float(p[1])] for p in points[:4]]
    xyxy = item.get("xyxy")
    if xyxy and len(xyxy) >= 4:
        x1, y1, x2, y2 = (float(v) for v in xyxy[:4])
        from . import geometry as geo
        return geo.rect_corners(x1, y1, x2 - x1, y2 - y1)
    return []


def proposed_boxes(items, label_id: str = "", known_ids=None) -> list[dict]:
    """Sidecar boxes from live detections, as proposals.

    ``label_id`` is written only when the reported class is genuinely a label
    in the library. That distinction is what makes this work under both
    detectors:

    - A per-label detector reports "2220-9199", which IS the identity.
    - A generic detector reports "label", which is not an identity at all --
      and stamping it would put label_id="label" on every box, a value no
      recipe contains and no library row matches. Left blank, the box says
      what is true: something label-shaped is here, nobody has said what.
    - After stage 2 has run, the class is the classifier's answer, so a known
      one stamps and an "unknown" correctly does not.

    Without ``known_ids`` nothing is stamped, which is the safe direction: a
    missing identity is visible, an invented one is not.
    """
    from . import annotations as ann
    from .labels import STRUCTURAL_CLASSES

    known = set(known_ids or ())
    out: list[dict] = []
    for item in items or []:
        name = str(item.get("name", "") or "")
        points = _item_points(item)
        if not name or not points:
            continue
        extra: dict = {"proposed_by": PROPOSED_BY}
        if item.get("track_id") is not None:
            extra["track_id"] = int(item["track_id"])
        identity = name if (name in known and name not in STRUCTURAL_CLASSES) else ""
        out.append(ann.make_box(
            name, points, label_id=identity,
            confidence=float(item.get("conf", 0.0)),
            **extra))
    return out


def proposed_annotation(image: str, label_id: str, items,
                        width: int = 0, height: int = 0,
                        session: str = "", known_ids=None) -> dict:
    """A sidecar pre-filled with what the model just found. Never reviewed.

    ``label_id`` records whose dataset the image landed in, not what the boxes
    are -- each box carries its own identity from the model.
    """
    from . import annotations as ann

    meta: dict = {"proposed_by": PROPOSED_BY}
    if session:
        meta["session"] = session
    data = ann.new_annotation(image, label_id, width, height, **meta)
    data["boxes"] = proposed_boxes(items, label_id, known_ids)
    return data


# --- stage 2 at runtime: naming what the detector found --------------------

# Below this the classifier is guessing. A classifier always returns its best
# class, so without a floor a label that was never trained on comes back as
# whichever known label it least resembles -- confidently, and wrongly. On a
# line that counts label ids against a recipe, a confident wrong id is worse
# than an honest blank, because nothing downstream can tell it was a guess.
UNKNOWN = "unknown"
DEFAULT_IDENTITY_FLOOR = 0.55


def identify(name: str, conf: float, floor: float = DEFAULT_IDENTITY_FLOOR) -> tuple[str, float]:
    """The classifier's answer, or UNKNOWN when it is not sure enough."""
    conf = float(conf or 0.0)
    if not name or conf < float(floor):
        return UNKNOWN, conf
    return str(name), conf


def apply_identities(items: list[dict], identities) -> list[dict]:
    """Put stage 2's answers onto stage 1's boxes, by position.

    Degrades to leaving the detector's own class name in place when the two
    lists disagree in length. That direction matters: a misaligned identity
    would put a real label id on the wrong box, which is indistinguishable
    downstream from a correct read. Showing `label` is visibly incomplete;
    showing the wrong id is invisibly false.
    """
    if not identities or len(identities) != len(items):
        return items
    out = []
    for item, (name, conf) in zip(items, identities):
        merged = dict(item)
        merged["detector_name"] = item.get("name", "")
        merged["name"] = name
        merged["identity_conf"] = float(conf)
        # Stage 2's name and stage 2's confidence, and nothing else. The track
        # id stays on the item for the readout to group by; on the box it only
        # competes with the two things worth reading as parts move past.
        merged["label"] = f"{name} {conf:.2f}"
        out.append(merged)
    return out


# --- what is actually running, and how fast --------------------------------

# Past this, inference is slow enough to be worth explaining rather than just
# reporting. A YOLO11s at 640 runs in single-digit milliseconds on a current
# GPU; 100 ms+ is the CPU's signature.
SLOW_INFER_MS = 60.0


def rate_line(display_fps: float, camera_fps: float, rolling: "Rolling",
              gui_ms: float = 0.0) -> str:
    """Camera, display and inference rates side by side.

    Three different numbers that were being read as one. The camera delivers at
    its rate, the preview repaints at its own, and the model runs at a third --
    and "6 fps" meant the third while sounding like the first. Separating them
    is what makes it obvious which one is the problem.
    """
    parts = [f"camera {camera_fps:.0f}/s" if camera_fps else "camera -/s",
             f"display {display_fps:.0f}/s" if display_fps else "display -/s",
             f"inference {rolling.rate:.1f}/s"]
    if rolling.mean_ms:
        parts.append(f"{rolling.mean_ms:.0f} ms/frame")
    if gui_ms:
        # The GUI thread's own cost per frame. Inference runs on it, so this
        # includes the model -- and the window cannot repaint or respond while
        # it is inside here.
        parts.append(f"gui {gui_ms:.0f} ms")
    return "  |  ".join(parts)


def slow_hint(rolling: "Rolling", device_line: str, speed: dict | None = None) -> str:
    """Why inference might be taking as long as it is.

    Reads the phase breakdown before saying anything. The version this replaces
    did not, and so recited "check the image size, export to TensorRT" at a
    detector doing 12 ms of a 71 ms call -- advice about the one part that was
    already fast. It even claimed the total covered the detector only, when the
    total is the whole call, stage 2 included.
    """
    if not rolling.mean_ms or rolling.mean_ms < SLOW_INFER_MS:
        return ""
    total = rolling.mean_ms
    lines = ["", f"{total:.0f} ms per inference is slow."]

    if ("NO CUDA" in device_line) or ("model on cpu" in device_line.lower()):
        lines.append("  - This is running on the CPU. That alone explains it; "
                     "everything else is secondary.")
        return "\n".join(lines)

    speed = speed or {}
    detector = sum(float(speed.get(k, 0.0))
                   for k in ("preprocess", "inference", "postprocess"))
    stage2 = float(speed.get("stage2", 0.0))
    readout = float(speed.get("readout", 0.0))
    other = max(0.0, total - detector - stage2 - readout)

    # Whatever is actually largest, named first. Everything else is a guess.
    if detector and detector < total * 0.5:
        lines.append(f"  - The detector is only {detector:.0f} ms of that. "
                     f"Image size and TensorRT act on this part, so neither "
                     f"would help much here.")
    if stage2 > total * 0.3:
        lines.append(f"  - Stage 2 is {stage2:.0f} ms: cropping each detection "
                     f"out of the full-resolution frame and classifying it. "
                     f"More detections cost more. A smaller Stage 2 image size, "
                     f"or a classifier exported to TensorRT, acts here.")
    if readout > total * 0.3:
        lines.append(f"  - Reading the results is {readout:.0f} ms. That is "
                     f"where .cpu() forces a CUDA sync, so GPU work the "
                     f"detector deferred is billed at this line.")
    if other > total * 0.3:
        lines.append(f"  - {other:.0f} ms is outside every phase measured -- "
                     f"model call overhead, or the frame copy in front of it.")
    if float(speed.get("inference", 0.0)) > total * 0.5:
        lines += [
            "  - Most of it is the model itself. Cost rises with the square of "
            "the image size, so 1664 is ~7x the work of 640.",
            "  - A .pt runs through PyTorch every frame. Exporting to TensorRT "
            "(yolo export format=engine half=True) typically gives 2-4x on an "
            "RTX card, and Live Detect loads a .engine the same way.",
        ]
    if len(lines) == 2:
        lines.append("  - No single phase dominates; the breakdown above is "
                     "where to look.")
    return "\n".join(lines)


def phase_line(speed: dict, total_ms: float = 0.0) -> str:
    """Where the milliseconds actually go, from Ultralytics' own timings.

    A single latency figure cannot distinguish a slow model from a slow resize
    in front of it, and on a 20 MP source those look identical from outside.
    The three phases separate them: preprocess is CPU work on the full frame,
    inference is the GPU, postprocess is NMS and decoding. ``stage2`` and
    ``readout`` are ours -- the classifier pass and the tensor conversion --
    and ``other`` is whatever the measured total has left over, which is the
    number that says the breakdown is not the whole story.
    """
    if not speed:
        return ""
    order = ("preprocess", "inference", "postprocess", "stage2", "readout")
    parts = [f"{name} {float(speed[name]):.0f}" for name in order if name in speed]
    if not parts:
        return ""
    line = "   ms: " + "  ".join(parts)
    # What none of the phases account for. Printed rather than left implicit:
    # a breakdown that adds to 12 next to a total of 71 is the single most
    # useful thing on this readout, and it was invisible.
    if total_ms:
        rest = float(total_ms) - sum(float(speed.get(k, 0.0)) for k in order)
        if rest > 1.0:
            line += f"  other {rest:.0f}"

    pre = float(speed.get("preprocess", 0.0))
    inf = float(speed.get("inference", 0.0))
    if pre > inf * 1.5 and pre > 20:
        line += ("\n   Most of it is preprocess -- CPU work resizing the frame "
                 "before the model sees it, not the model. A smaller camera AOI "
                 "or a smaller frame would cut it; a faster model would not.")
    elif inf > 40:
        line += ("\n   Most of it is the model. Lower the image size, or export "
                 "to TensorRT (yolo export format=engine half=True).")
    return line
