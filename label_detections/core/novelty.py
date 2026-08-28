"""Refusing to name a label the classifier was never taught.

The failure this exists for: a battery goes past carrying a label that has
never been enrolled, its die cut happens to match one that has, and the
classifier reports it as that label at 1.00. On a line that counts label ids
against a recipe, that is the worst possible answer -- a confident wrong id is
indistinguishable downstream from a correct read, where a blank is not.

**No confidence threshold can catch it.** A classifier head is a softmax over
the classes it was given. Softmax sums to one and always elects a winner, so
"none of these" is not an answer it is able to return; a novel input is
assigned to whichever enrolled class it most resembles. And because these are
trained to convergence with cross-entropy on a handful of well-separated
classes, the winner comes back at ~1.00 whether it is right or wrong. Raising
the floor from 0.55 to 0.90, or to 0.99, moves nothing: correct reads and this
failure produce the same number. The threshold is not set too low. It is
measuring a quantity that does not contain the answer.

What does contain the answer is the layer underneath. Before the final linear
layer the network has a feature vector -- a description of what it actually
saw. Crops of one enrolled label land close together there; the novel label
lands somewhere else entirely, and the linear layer's job is only to pick the
nearest *enrolled* direction to it, which it does confidently because that is
all it can do. So the test is not "how sure is the classifier" but "does this
crop sit where crops of that class sit". A die cut in common moves the two
closer; printed content, colour and layout keep them far apart.

That test needs no new labelling. Every crop already in the classification
dataset is a sample of where its class lives, so the acceptable radius around
each class is measured from data that has been on disk since the export.

Distances are cosine, on L2-normalised vectors. Deep features vary in
magnitude with contrast and exposure in ways that say nothing about identity,
and normalising drops exactly that. The textbook alternative, Mahalanobis,
wants a covariance matrix: 1280 dimensions needs far more crops per class than
this ever has, and the estimate is singular long before it is useful.

The decisions are here and tested; reading vectors out of a model is in
``ui/novelty``, which is where torch is allowed to be.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Where the acceptable radius is drawn, as a percentile of the class's own
# crops. Not the maximum: one mislabelled or badly-cropped training image sets
# a radius that admits everything, and there is always one.
DEFAULT_PERCENTILE = 99.0

# Slack on top of that percentile. The crops were measured under whatever
# lighting and pose the dataset happens to contain, and a line runs wider than
# its dataset -- so the radius is deliberately generous. Erring this way costs
# an occasional unknown label that should have been named, which an operator
# sees and can correct. Erring the other way costs a wrong id nobody sees.
DEFAULT_MARGIN = 1.25

# Below this many crops, the spread of a class is not measured, it is guessed.
# A radius drawn from three images rejects honest parts all day, so a class
# this small is left unenforced and said so out loud.
MIN_SAMPLES = 8

# A radius no tighter than this, however tight the training crops cluster.
# Classes photographed in one session under one light can sit almost on top of
# each other, and a radius of 0.01 would reject the same label photographed
# tomorrow.
MIN_RADIUS = 0.05

# Beside the weights, named after them: a profile belongs to the exact model it
# was measured through, and pairing them by filename makes that hard to get
# wrong.
PROFILE_SUFFIX = ".novelty.json"

FORMAT = 1


def profile_path(weights: str | Path) -> Path:
    """Where the profile for these weights lives."""
    weights = Path(weights)
    return weights.with_suffix(weights.suffix + PROFILE_SUFFIX)


def unit(vec) -> np.ndarray:
    """L2-normalised, and safe on a zero vector."""
    arr = np.asarray(vec, dtype=np.float64).ravel()
    norm = float(np.linalg.norm(arr))
    return arr if norm == 0.0 else arr / norm


def centre_of(vectors) -> np.ndarray:
    """The direction a class sits in: the mean of its unit vectors, renormalised.

    Normalising before averaging and again after is what makes this a direction
    rather than a magnitude -- a handful of high-contrast crops would otherwise
    drag the centre toward themselves for a reason that is about the lighting.
    """
    stacked = np.stack([unit(v) for v in vectors])
    return unit(stacked.mean(axis=0))


def distance(vec, centre) -> float:
    """Cosine distance, in ``[0, 2]``. Zero is the same direction."""
    return float(1.0 - float(np.dot(unit(vec), unit(centre))))


def radius_for(distances, percentile: float = DEFAULT_PERCENTILE,
               margin: float = DEFAULT_MARGIN) -> float:
    """How far a crop may sit from its class centre and still be that class."""
    if len(distances) == 0:
        return 0.0
    at = float(np.percentile(np.asarray(distances, dtype=np.float64),
                             float(percentile)))
    return max(MIN_RADIUS, at * float(margin))


@dataclass
class ClassProfile:
    """Where one enrolled label lives in feature space, and how far it spreads."""
    name: str
    centre: np.ndarray
    radius: float
    samples: int
    typical: float = 0.0        # median distance, for the report

    @property
    def enforced(self) -> bool:
        """Is there enough evidence here to reject anything?"""
        return self.samples >= MIN_SAMPLES and self.radius > 0.0


@dataclass
class Verdict:
    """What the profile made of one crop."""
    known: bool
    distance: float = 0.0
    radius: float = 0.0
    reason: str = ""

    @property
    def ratio(self) -> float:
        """How far out, as a multiple of the radius. 1.0 is exactly on it."""
        return self.distance / self.radius if self.radius else 0.0


class Profile:
    """Per-class centres and radii for one classifier."""

    def __init__(self, classes: dict[str, ClassProfile] | None = None,
                 *, dim: int = 0, crop_px: int = 0, weights: str = "",
                 built: str = ""):
        self.classes = dict(classes or {})
        self.dim = int(dim)
        self.crop_px = int(crop_px)
        self.weights = str(weights)
        self.built = str(built)

    def __len__(self) -> int:
        return len(self.classes)

    @property
    def enforced_classes(self) -> list[str]:
        return sorted(n for n, c in self.classes.items() if c.enforced)

    def verdict(self, name: str, vec) -> Verdict:
        """Does this crop sit where crops of ``name`` sit?

        Silent about anything it cannot judge. A class that is not in the
        profile, or one measured from too few crops, returns known -- the
        profile's job is to reject what it has evidence against, and inventing
        a rejection from no evidence is the same error in the other direction.
        """
        entry = self.classes.get(str(name))
        if entry is None:
            return Verdict(True, reason="not in the profile")
        if not entry.enforced:
            return Verdict(True, reason=f"only {entry.samples} crop(s) enrolled")
        gap = distance(vec, entry.centre)
        if gap <= entry.radius:
            return Verdict(True, distance=gap, radius=entry.radius)
        return Verdict(False, distance=gap, radius=entry.radius,
                       reason=f"{gap:.3f} from {name}, past {entry.radius:.3f}")

    # --- on disk ----------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "dim": self.dim,
            "crop_px": self.crop_px,
            "weights": self.weights,
            "built": self.built,
            "percentile": DEFAULT_PERCENTILE,
            "margin": DEFAULT_MARGIN,
            "classes": {
                name: {
                    # Rounded: six decimals on a unit vector is far below the
                    # precision any of this is measured to, and it keeps the
                    # file readable by a human trying to work out what it says.
                    "centre": [round(float(x), 6) for x in entry.centre],
                    "radius": round(float(entry.radius), 6),
                    "samples": int(entry.samples),
                    "typical": round(float(entry.typical), 6),
                }
                for name, entry in sorted(self.classes.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        classes = {}
        for name, entry in (data.get("classes") or {}).items():
            classes[str(name)] = ClassProfile(
                name=str(name),
                centre=np.asarray(entry.get("centre") or [], dtype=np.float64),
                radius=float(entry.get("radius") or 0.0),
                samples=int(entry.get("samples") or 0),
                typical=float(entry.get("typical") or 0.0),
            )
        return cls(classes, dim=int(data.get("dim") or 0),
                   crop_px=int(data.get("crop_px") or 0),
                   weights=str(data.get("weights") or ""),
                   built=str(data.get("built") or ""))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Profile | None":
        """The profile for these weights, or None when there is not one.

        None rather than an exception: running without a profile is a supported
        state -- it is what every existing model is in -- and the caller says so
        in the readout instead of failing to start.
        """
        path = Path(path)
        if not path.is_file():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return None


def build(samples: dict[str, list], *, percentile: float = DEFAULT_PERCENTILE,
          margin: float = DEFAULT_MARGIN, crop_px: int = 0,
          weights: str = "", built: str = "") -> Profile:
    """Centres and radii from the embeddings of each class's own crops."""
    classes: dict[str, ClassProfile] = {}
    dim = 0
    for name, vectors in samples.items():
        vectors = [v for v in vectors if v is not None and len(np.asarray(v).ravel())]
        if not vectors:
            continue
        centre = centre_of(vectors)
        dim = dim or int(centre.size)
        gaps = np.array([distance(v, centre) for v in vectors], dtype=np.float64)
        classes[str(name)] = ClassProfile(
            name=str(name), centre=centre,
            radius=radius_for(gaps, percentile, margin),
            samples=len(vectors), typical=float(np.median(gaps)))
    return Profile(classes, dim=dim, crop_px=int(crop_px),
                   weights=str(weights), built=str(built))


def report(profile: Profile | None) -> str:
    """What the profile knows, for a human deciding whether to trust it."""
    if profile is None or not len(profile):
        return ("No novelty profile. Every crop will be named as the closest "
                "enrolled label, however far from it the crop actually sits.")
    lines = [f"{len(profile)} class(es) profiled, {profile.dim}-d features."]
    weak = []
    for name in sorted(profile.classes):
        entry = profile.classes[name]
        if entry.enforced:
            lines.append(f"  {name}: {entry.samples} crops, typical "
                         f"{entry.typical:.3f}, rejects past {entry.radius:.3f}")
        else:
            weak.append(f"{name} ({entry.samples})")
    if weak:
        lines.append(f"  Not enforced, under {MIN_SAMPLES} crops: "
                     + ", ".join(weak))
        lines.append("  Those classes will still accept anything. Capture and "
                     "label more of them, then rebuild.")
    return "\n".join(lines)
