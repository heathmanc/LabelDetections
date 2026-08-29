"""Which captures are of the same battery, so the split never separates them.

The train/val split never separates a capture group, and that guarantee is what
makes a validation number mean anything: two frames of one battery, taken a
second apart in the same pose, are very nearly the same image. Put one in train
and the other in val and the model is being tested on something it memorised.

Grouping came from ``session`` or ``source`` in the sidecar, and only frames
kept from Live Detect ever carried a session. The Capture button writes an
image and no sidecar at all, so ``Entry.group_key()`` fell through to the
filename and every frame became its own group -- which is not a group, it is
the absence of one. Fire off twenty frames of one battery and the splitter was
free to put frame 7 in train and frame 8 in val.

Nothing in software knows when the battery was changed, so this does not guess.
A session starts when the camera opens and when the operator says a new one
has, and every capture until then belongs to it. That errs toward LARGER
groups, which is the safe direction: merging two batteries into one group costs
a little validation data, while splitting one battery across two costs the
meaning of the number.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .storage import label_folder

# Beside the sidecars rather than in them: an image is captured long before it
# is annotated, often in a different sitting, and the sidecar does not exist
# yet at the moment the only thing that knows the session is running.
FILENAME = "capture_sessions.json"


_last_id = ""


def new_id(now: float | None = None) -> str:
    """A session token. Sorts chronologically and reads as a timestamp.

    Milliseconds, and never the same token twice in a row. Two batteries
    swapped inside one second would otherwise share an id and be merged into
    one group -- the safe direction, but it silently defeats the button whose
    entire job is to say they are different.
    """
    global _last_id
    moment = time.time() if now is None else float(now)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(moment))
    token = f"cap_{stamp}_{int(moment * 1000) % 1000:03d}"
    if token == _last_id:
        token = f"{token}x"
        while token == _last_id:
            token += "x"
    _last_id = token
    return token


def path_for(label_id: str, root: Path | None = None) -> Path:
    return (root or label_folder(label_id)) / FILENAME


def load(label_id: str, root: Path | None = None) -> dict[str, str]:
    """``{image name: session}`` for one label. Empty when there is none."""
    try:
        data: Any = json.loads(path_for(label_id, root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if k and v}


def record(label_id: str, image: str | Path, session: str,
           root: Path | None = None) -> None:
    """Note which session a capture belongs to. Silent on failure.

    Silent because the alternative is refusing to capture over a bookkeeping
    file. A missing entry costs the grouping for that one image, which is what
    every image had before this existed.
    """
    name = Path(str(image)).name
    if not name or not session:
        return
    known = load(label_id, root)
    if known.get(name) == session:
        return
    known[name] = str(session)
    target = path_for(label_id, root)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(known, indent=1, sort_keys=True),
                          encoding="utf-8")
    except OSError:
        pass


def session_for(known: dict[str, str], image: str | Path) -> str:
    """The session a capture belongs to, or "" when it was never recorded."""
    return known.get(Path(str(image)).name, "")


def group_summary(sessions: dict[str, str], images: int = 0) -> str:
    """One line about how well a dataset is grouped, for the export report.

    A dataset in one group cannot be split at all, and a dataset in as many
    groups as it has images is not grouped. Both are worth saying out loud
    before anybody reads a validation number off it.
    """
    if not images:
        return ""
    grouped = len(sessions)
    groups = len(set(sessions.values()))
    ungrouped = max(0, images - grouped)
    if not groups:
        return (f"No capture grouping: all {images} image(s) are their own "
                f"group, so near-identical frames of one battery can land on "
                f"both sides of the split.")
    parts = [f"{groups} capture group(s) across {grouped} image(s)"]
    if ungrouped:
        parts.append(f"{ungrouped} ungrouped -- captured before grouping "
                     f"existed, so each is its own group")
    if groups == 1 and not ungrouped:
        parts.append("one group cannot be split: train and val will be the "
                     "same images")
    return "; ".join(parts) + "."
