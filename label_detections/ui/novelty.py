"""Reading feature vectors out of a classifier, and building a profile from them.

The reasoning for all of this is in ``core/novelty``. This half is the part
that has to touch torch: finding the layer whose input is the description of
what the network saw, capturing it, and running the enrolled crops through
once to measure where each class lives.

Capture is a forward pre-hook on the final linear layer rather than a second
pass with ``embed=``. Ultralytics' ``embed`` returns embeddings *instead of*
predictions, so using it would mean running every crop through the network
twice -- and stage 2 already costs 11 ms of a 33 ms budget on a frame that has
to keep up with a 30 fps camera. A pre-hook rides the pass that was happening
anyway and costs a tensor copy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..core import novelty as logic


def final_linear(module):
    """The last ``nn.Linear`` in a model, whose input is the feature vector.

    Found by walking rather than by path. ``model.model.model[-1].linear`` is
    where an Ultralytics ``Classify`` head keeps it today, and a path spelled
    out like that breaks silently on the next version -- silently being the
    problem, because the failure mode is novelty checking quietly not running.
    """
    import torch.nn as nn

    last = None
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            last = layer
    return last


class Embedder:
    """Captures the feature vectors of whatever passes through a model.

    Vectors accumulate across forward passes and are taken all at once, so a
    predictor that splits a list of crops into several batches still yields one
    vector per crop, in order.
    """

    def __init__(self):
        self._handle = None
        self._batches: list[np.ndarray] = []
        self.reason = ""

    @property
    def attached(self) -> bool:
        return self._handle is not None

    def attach(self, model) -> bool:
        """Hook the classifier. False, with a reason, when there is nothing to hook."""
        self.detach()
        inner = getattr(model, "model", model)
        try:
            linear = final_linear(inner)
        except Exception as exc:
            self.reason = f"could not inspect the model ({exc})"
            return False
        if linear is None:
            self.reason = ("this model has no linear classification head, so "
                           "there is no feature vector to read")
            return False

        def pre(_module, args):
            if not args:
                return
            try:
                # float() before numpy: half precision does not convert, and
                # the distances downstream are float64 anyway.
                self._batches.append(
                    args[0].detach().float().cpu().numpy().reshape(
                        args[0].shape[0], -1))
            except Exception:
                pass

        try:
            self._handle = linear.register_forward_pre_hook(pre)
        except Exception as exc:
            self.reason = f"could not attach to the model ({exc})"
            return False
        self.reason = ""
        return True

    def detach(self) -> None:
        if self._handle is not None:
            try:
                self._handle.remove()
            except Exception:
                pass
        self._handle = None
        self._batches = []

    def clear(self) -> None:
        self._batches = []

    def take(self) -> list[np.ndarray]:
        """Every vector captured since the last take, in order."""
        if not self._batches:
            return []
        rows = np.concatenate(self._batches, axis=0)
        self._batches = []
        return [np.asarray(row, dtype=np.float64) for row in rows]


# --- building a profile from the enrolled crops -----------------------------

# One forward pass per batch, and the crops are already on disk at the size the
# model wants. Big enough to keep a GPU busy, small enough not to matter on a
# machine that has to hold a 20 MP frame at the same time.
BATCH = 32


def crop_folders(dataset: Path) -> dict[str, list[Path]]:
    """``{class name: [crop, ...]}`` from a YOLO classification dataset.

    Both splits. Train says where a class sits, and val says how far a crop of
    it can honestly fall from there -- the model was fitted to train, so train
    distances alone read tighter than the line will ever be.
    """
    dataset = Path(dataset)
    out: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        base = dataset / split
        if not base.is_dir():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            files = sorted(f for f in folder.iterdir()
                           if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
            if files:
                out.setdefault(folder.name, []).extend(files)
    return out


def build_profile(weights: str | Path, dataset: str | Path, *,
                  crop_px: int = 224, device=None, progress=None) -> logic.Profile:
    """Measure where every enrolled class lives, and save it beside the weights.

    Raises rather than returning something empty: a profile that silently
    covers nothing is worse than no profile, because the readout would report
    novelty checking as on.

    Measured through the crops on disk, which is what the classifier was
    trained on. The live path reaches the same size by a slightly different
    road -- it caps the warp at twice the crop rather than warping at full size
    and then reducing -- so a runtime vector sits a little further from its
    centre than a dataset one does. That is what the margin in ``core/novelty``
    is for, and if honest parts start reading unknown it is the first place to
    look.
    """
    import time

    from ultralytics import YOLO

    folders = crop_folders(dataset)
    if not folders:
        raise FileNotFoundError(
            f"No classification crops under:\n{dataset}\n\n"
            "This wants the folder the two-stage export writes "
            "(train/<label id>/*.jpg), not the weights folder.")

    model = YOLO(str(weights))
    embedder = Embedder()
    if not embedder.attach(model):
        raise RuntimeError(f"Cannot read feature vectors: {embedder.reason}")

    total = sum(len(v) for v in folders.values())
    done = 0
    samples: dict[str, list[np.ndarray]] = {}
    try:
        for name, files in folders.items():
            vectors: list[np.ndarray] = []
            for start in range(0, len(files), BATCH):
                chunk = files[start:start + BATCH]
                embedder.clear()
                model.predict([str(f) for f in chunk], imgsz=int(crop_px),
                              verbose=False,
                              **({"device": device} if device is not None else {}))
                got = embedder.take()
                # Only when they line up. A partial batch silently paired with
                # the wrong crops would put one class's centre inside another's.
                if len(got) == len(chunk):
                    vectors.extend(got)
                done += len(chunk)
                if progress is not None:
                    progress(done, total, f"{name}: {len(vectors)} crop(s)")
            if vectors:
                samples[name] = vectors
    finally:
        embedder.detach()

    if not samples:
        raise RuntimeError(
            "Read no feature vectors from any crop. The model loaded and the "
            "crops were found, so this is the hook not firing -- the profile "
            "would cover nothing.")

    profile = logic.build(
        samples, crop_px=int(crop_px), weights=str(weights),
        built=time.strftime("%Y-%m-%d %H:%M:%S"))
    profile.save(logic.profile_path(weights))
    return profile
