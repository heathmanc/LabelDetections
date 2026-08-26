"""Ranking which unlabeled image to work on next."""
from __future__ import annotations

from label_detections.core import active_learning as al


def test_a_miss_outranks_everything_else():
    """The model cannot see the label here, which is what more data fixes."""
    miss = al.disagreement_score(found=0, expected=1, total_detections=0, avg_conf=0.9)
    extra = al.disagreement_score(found=3, expected=1, total_detections=3, avg_conf=0.9)
    assert miss > extra


def test_a_confident_single_find_scores_near_zero():
    assert al.disagreement_score(found=1, expected=1, total_detections=1, avg_conf=1.0) == 0.0


def test_extra_finds_of_the_same_family_raise_the_score():
    one = al.disagreement_score(found=1, expected=1, total_detections=1)
    two = al.disagreement_score(found=2, expected=1, total_detections=2)
    assert two > one


def test_low_confidence_lifts_an_otherwise_clean_image():
    sure = al.disagreement_score(found=1, expected=1, total_detections=1, avg_conf=0.95)
    unsure = al.disagreement_score(found=1, expected=1, total_detections=1, avg_conf=0.30)
    assert unsure > sure


def test_a_frame_the_model_fires_on_everywhere_is_noisy_too():
    quiet = al.disagreement_score(found=1, expected=1, total_detections=1)
    noisy = al.disagreement_score(found=1, expected=1, total_detections=12)
    assert noisy > quiet


def test_confidence_is_optional():
    assert al.disagreement_score(found=1) == 0.0


def test_ranking_is_highest_first_and_stable_on_ties():
    items = [al.QueueItem("b", 1.0), al.QueueItem("a", 1.0), al.QueueItem("c", 5.0)]
    assert [i.key for i in al.rank_items(items)] == ["c", "a", "b"]
