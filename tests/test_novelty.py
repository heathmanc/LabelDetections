"""Refusing to name a label the classifier was never taught.

The failure being tested for is specific and was seen on a real line: a
battery carrying a label that has never been enrolled, whose die cut matches
one that has, reported as that label at 1.00. Everything here is arithmetic on
vectors, so none of it needs torch -- which is the point of keeping the
decisions out of the half that does.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from label_detections.core import novelty as nv


def _cluster(direction, n, spread=0.02, seed=0):
    """``n`` vectors pointing roughly along ``direction``."""
    rng = np.random.default_rng(seed)
    base = np.asarray(direction, dtype=np.float64)
    return [base + rng.normal(0.0, spread, size=base.shape) for _ in range(n)]


# --- the arithmetic ---------------------------------------------------------

def test_a_zero_vector_does_not_divide_by_zero():
    """A crop that produced nothing must not take the process with it."""
    assert np.allclose(nv.unit(np.zeros(4)), np.zeros(4))


def test_distance_is_zero_along_the_same_direction_and_ignores_length():
    """Cosine, deliberately: deep features change magnitude with exposure and
    contrast, and none of that is about which label it is."""
    assert nv.distance([1.0, 0.0], [3.0, 0.0]) == pytest.approx(0.0)
    assert nv.distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert nv.distance([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(2.0)


def test_the_centre_is_a_direction_not_an_average_of_lengths():
    """One high-contrast crop with a big vector would otherwise drag the centre
    toward itself for a reason that is about the lighting."""
    centre = nv.centre_of([[10.0, 0.0], [0.0, 1.0]])
    assert nv.distance(centre, [1.0, 1.0]) == pytest.approx(0.0, abs=1e-9)


def test_the_radius_ignores_one_bad_crop_but_not_the_spread():
    """There is always one mislabelled or half-cropped training image, and a
    radius set by the maximum admits everything."""
    good = [0.02] * 99 + [1.4]
    assert nv.radius_for(good, percentile=99.0, margin=1.0) < 0.3


def test_a_class_whose_crops_are_nearly_identical_still_gets_a_usable_radius():
    """Photographed in one session under one light, a class can cluster so
    tightly that its own label photographed tomorrow falls outside."""
    assert nv.radius_for([0.0] * 50) >= nv.MIN_RADIUS


# --- the failure this exists for -------------------------------------------

def _profile_of_two_labels():
    return nv.build({
        "PC680": _cluster([1.0, 0.0, 0.0], 40, seed=1),
        "2220-9199": _cluster([0.0, 1.0, 0.0], 40, seed=2),
    })


def test_an_enrolled_label_is_still_named():
    """The check is worth nothing if it rejects the parts that are fine."""
    profile = _profile_of_two_labels()
    for vec in _cluster([1.0, 0.0, 0.0], 10, seed=99):
        assert profile.verdict("PC680", vec).known


def test_a_label_that_was_never_enrolled_is_refused():
    """The screenshot: a Genesys NP16-12B whose die cut matches PC680's. The
    softmax has no 'none of these' to return, so it returns PC680 at 1.00 --
    and the feature vector says plainly that the crop is nowhere near PC680."""
    profile = _profile_of_two_labels()
    novel = np.array([0.0, 0.0, 1.0])
    verdict = profile.verdict("PC680", novel)
    assert not verdict.known
    assert verdict.ratio > 1.0
    assert "PC680" in verdict.reason


def test_a_shared_die_cut_moves_it_closer_and_still_does_not_get_it_in():
    """Shape is a real cue and a novel label may share it. What it cannot share
    is the printed content, which is most of what the features describe."""
    profile = _profile_of_two_labels()
    # 60% of the way toward PC680's direction: much closer than chance, and
    # still far outside a radius measured from PC680's own crops.
    lookalike = 0.6 * np.array([1.0, 0.0, 0.0]) + 0.4 * np.array([0.0, 0.0, 1.0])
    assert not profile.verdict("PC680", lookalike).known


# --- refusing to judge what it has no evidence about ------------------------

def test_a_class_with_too_few_crops_is_left_alone_and_says_so():
    """A radius drawn from three images rejects honest parts all day. Not
    enforcing is the honest answer, and the report has to name it."""
    profile = nv.build({"barely": _cluster([1.0, 0.0, 0.0], 3, seed=3)})
    verdict = profile.verdict("barely", [0.0, 0.0, 1.0])
    assert verdict.known
    assert "3 crop" in verdict.reason
    assert "barely (3)" in nv.report(profile)
    assert "still accept anything" in nv.report(profile)


def test_a_class_the_profile_never_saw_is_not_rejected_on_no_evidence():
    """Inventing a rejection from nothing is the same error in the other
    direction, and it would fire on every class added since the profile."""
    profile = _profile_of_two_labels()
    assert profile.verdict("added-last-week", [0.0, 0.0, 1.0]).known


def test_only_the_classes_with_evidence_are_reported_as_enforced():
    profile = nv.build({
        "solid": _cluster([1.0, 0.0, 0.0], 40, seed=4),
        "thin": _cluster([0.0, 1.0, 0.0], 2, seed=5),
    })
    assert profile.enforced_classes == ["solid"]


# --- on disk ----------------------------------------------------------------

def test_a_profile_survives_the_round_trip_and_still_refuses_the_same_crop():
    profile = _profile_of_two_labels()
    data = json.loads(json.dumps(profile.to_dict()))
    back = nv.Profile.from_dict(data)
    assert back.enforced_classes == profile.enforced_classes
    assert not back.verdict("PC680", [0.0, 0.0, 1.0]).known
    assert back.verdict("PC680", [1.0, 0.0, 0.0]).known


def test_the_profile_is_named_after_the_weights_it_was_measured_through():
    """It describes one model's feature space and is meaningless against
    another, so pairing them by filename makes that hard to get wrong."""
    path = nv.profile_path("runs/classify/train3/weights/best.pt")
    assert path.name == "best.pt.novelty.json"


def test_a_missing_or_corrupt_profile_reads_as_no_profile_not_a_crash(tmp_path):
    """Running without one is what every model built before this does."""
    assert nv.Profile.load(tmp_path / "nothing.json") is None
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert nv.Profile.load(broken) is None


def test_saving_and_loading_gives_back_a_working_profile(tmp_path):
    saved = _profile_of_two_labels().save(tmp_path / "w.pt.novelty.json")
    back = nv.Profile.load(saved)
    assert back is not None and len(back) == 2
    assert not back.verdict("PC680", [0.0, 0.0, 1.0]).known


def test_the_report_says_plainly_when_there_is_no_profile_at_all():
    """"Off" and "on" look identical while every part happens to be enrolled."""
    text = nv.report(None)
    assert "No novelty profile" in text
    assert "closest enrolled label" in text


# --- building ---------------------------------------------------------------

def test_building_records_how_many_crops_each_class_was_measured_from():
    profile = nv.build({"a": _cluster([1.0, 0.0], 12, seed=6)})
    assert profile.classes["a"].samples == 12
    assert profile.dim == 2
    assert profile.classes["a"].typical > 0.0


def test_a_class_with_no_usable_vectors_is_dropped_rather_than_faked():
    profile = nv.build({"a": _cluster([1.0, 0.0], 12, seed=7), "b": [], "c": [None]})
    assert sorted(profile.classes) == ["a"]


# --- the wiring into live detect --------------------------------------------
#
# The arithmetic above is worth nothing if the vectors never reach it, and the
# ways that can go wrong are all silent: a hook that does not fire, a profile
# that does not load, a batch left over from the previous frame pairing with
# this frame's crops. These check the seams, with a stub in place of torch.

class _StubEmbedder:
    """Stands in for the forward hook: hands back whatever it was given."""

    def __init__(self, vectors):
        self._vectors = list(vectors)
        self.cleared = 0

    def clear(self):
        self.cleared += 1

    def take(self):
        out, self._vectors = self._vectors, []
        return out


def _worker(profile=None, vectors=()):
    from label_detections.ui.live_detect import InferenceWorker

    worker = InferenceWorker("det.pt", 640, 0.5, None, classifier_path="cls.pt")
    worker._novelty = profile
    worker._embedder = _StubEmbedder(vectors) if profile is not None else None
    return worker


def test_the_worker_replaces_a_refused_identity_with_unknown():
    """End of the wire. The classifier said PC680 at 1.00 and meant it; the
    feature vector is what disagrees."""
    profile = _profile_of_two_labels()
    worker = _worker(profile, [np.array([0.0, 0.0, 1.0])])
    assert worker._reject_unknown([("PC680", 1.0)], 1) == [("unknown", 1.0)]


def test_an_enrolled_crop_passes_the_worker_untouched():
    profile = _profile_of_two_labels()
    worker = _worker(profile, [np.array([1.0, 0.0, 0.0])])
    assert worker._reject_unknown([("PC680", 0.97)], 1) == [("PC680", 0.97)]


def test_each_crop_is_judged_against_its_own_class_not_the_first_one():
    """Order is the only thing tying a vector to a prediction, and getting it
    wrong would reject good parts and admit bad ones at the same time."""
    profile = _profile_of_two_labels()
    worker = _worker(profile, [np.array([1.0, 0.0, 0.0]),
                               np.array([0.0, 1.0, 0.0])])
    out = worker._reject_unknown([("PC680", 1.0), ("2220-9199", 1.0)], 2)
    assert out == [("PC680", 1.0), ("2220-9199", 1.0)]


def test_a_vector_count_that_does_not_match_fails_open_and_says_so():
    """Rejecting on an untrustworthy pairing would mark good parts unknown for
    a reason nobody can see. The message is the visible half of that."""
    profile = _profile_of_two_labels()
    worker = _worker(profile, [np.array([0.0, 0.0, 1.0])])
    said = []
    worker.failed.connect(said.append)
    assert worker._reject_unknown([("PC680", 1.0), ("2220-9199", 1.0)], 2) \
        == [("PC680", 1.0), ("2220-9199", 1.0)]
    assert said and "NOT running" in said[0]


def test_that_warning_is_said_once_not_once_a_frame():
    """At the camera's rate it would bury every other message in the log."""
    worker = _worker(_profile_of_two_labels(), [])
    said = []
    worker.failed.connect(said.append)
    for _ in range(5):
        worker._embedder = _StubEmbedder([])
        worker._reject_unknown([("PC680", 1.0)], 1)
    assert len(said) == 1


def test_no_profile_means_the_identities_pass_through_unchanged():
    """Every model built before this feature is in that state, and it has to
    keep working exactly as it did."""
    worker = _worker(None)
    assert worker._reject_unknown([("PC680", 1.0)], 1) == [("PC680", 1.0)]


def test_an_already_unknown_identity_is_not_re_judged():
    """It failed the confidence floor; there is no class to measure against."""
    profile = _profile_of_two_labels()
    worker = _worker(profile, [np.array([0.0, 0.0, 1.0])])
    assert worker._reject_unknown([("unknown", 0.2)], 1) == [("unknown", 0.2)]


def test_a_missing_profile_is_named_in_the_load_message_not_left_silent():
    """"On" and "off" look identical while every part happens to be enrolled,
    and the run where they stop being is the run where nobody remembers."""
    worker = _worker(None)
    worker._classifier_path = "does-not-exist.pt"
    worker._load_novelty()
    assert worker._novelty is None
    assert worker._novelty_note == "no novelty profile"


def test_a_saved_profile_is_found_beside_the_weights(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_text("not really weights", encoding="utf-8")
    _profile_of_two_labels().save(nv.profile_path(weights))
    worker = _worker(None)
    worker._classifier_path = str(weights)
    # The classifier is None here, so attaching the hook is what fails -- and
    # that has to read as novelty being off, with a reason, not as it being on.
    worker._load_novelty()
    assert worker._novelty is None
    assert worker._novelty_note.startswith("novelty off:")


# --- finding the crops ------------------------------------------------------

def test_both_splits_are_measured(tmp_path):
    """Train says where a class sits; val says how far a crop of it can
    honestly fall from there. The model was fitted to train, so train distances
    alone read tighter than the line will ever be."""
    from label_detections.ui import novelty as novelty_ui

    for split, count in (("train", 3), ("val", 2)):
        folder = tmp_path / split / "PC680"
        folder.mkdir(parents=True)
        for i in range(count):
            (folder / f"{i}.jpg").write_bytes(b"")
    found = novelty_ui.crop_folders(tmp_path)
    assert len(found["PC680"]) == 5


def test_a_dataset_with_no_crops_reads_as_empty_rather_than_raising(tmp_path):
    from label_detections.ui import novelty as novelty_ui

    assert novelty_ui.crop_folders(tmp_path) == {}


def test_raising_the_confidence_threshold_is_not_what_fixes_this():
    """Stated as a test because it is the thing that keeps being reached for.
    The reported failure came back at 1.00 -- the top of the scale -- so there
    is no threshold below it to set. The profile is not handed a confidence at
    all, and refuses the crop at every one of them."""
    profile = _profile_of_two_labels()
    novel = np.array([0.0, 0.0, 1.0])
    for conf in (0.55, 0.90, 0.99, 1.00):
        worker = _worker(profile, [novel])
        assert worker._reject_unknown([("PC680", conf)], 1) == [("unknown", conf)]
