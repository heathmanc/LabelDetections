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
MIN_INTERVAL_S = 0.15

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


def frame_summary(counts: dict[str, int], family: str, label_id: str,
                  rolling: Rolling) -> str:
    """The readout under the live view."""
    total = sum(counts.values())
    lines = [f"{total} detection(s)   {rolling.mean_ms:.0f} ms   {rolling.rate:.1f}/s"]
    if family:
        found = counts.get(family, 0)
        state = "found" if found else "NOT FOUND"
        lines.append(f"{label_id} ({family}): {found} {state}")
    for name in sorted(counts):
        if name != family:
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
        lines = []
        for track in rows:
            lines.append(
                f"#{track.track_id} {track.name}: {track.last_conf:.2f} now, "
                f"{track.mean_conf:.2f} mean over {track.frames} frames "
                f"({track.min_conf:.2f}-{track.max_conf:.2f})")
        return "\n".join(lines)


def track_summary(book: TrackBook, family: str, label_id: str,
                  rolling: Rolling) -> str:
    """The readout when tracking is on."""
    rows = book.rows()
    lines = [f"{len(rows)} tracked   {rolling.mean_ms:.0f} ms   {rolling.rate:.1f}/s"]
    if family:
        mine = [t for t in rows if t.name == family]
        if mine:
            best = mine[0]
            lines.append(f"{label_id}: held {best.frames} frames, "
                         f"mean {best.mean_conf:.2f}")
        else:
            lines.append(f"{label_id} ({family}): NOT TRACKED")
    lines.append("")
    lines.append(book.text())
    return "\n".join(lines)
