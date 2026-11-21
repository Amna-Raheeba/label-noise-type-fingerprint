"""Tests for the fingerprint statistics."""

import math

from noise_fingerprint.fingerprint import (
    compute_fingerprint,
    detect_dup_pairs,
    loo_predictions,
    majority_label,
    margin_profile,
    neighbour_cache,
    vote_margins,
)
from noise_fingerprint.generate import build_dataset
from noise_fingerprint.models import FINGERPRINT_KEYS


def test_fingerprint_has_exactly_the_declared_keys():
    ds = build_dataset(5, "uniform", 0.20)
    fp = compute_fingerprint(ds.features, ds.labels)
    assert set(fp.keys()) == set(FINGERPRINT_KEYS)


def test_fingerprint_values_are_finite():
    ds = build_dataset(5, "boundary", 0.25)
    fp = compute_fingerprint(ds.features, ds.labels)
    for value in fp.values():
        assert math.isfinite(value)


def test_fingerprint_is_deterministic():
    ds = build_dataset(5, "uniform", 0.20)
    a = compute_fingerprint(ds.features, ds.labels)
    b = compute_fingerprint(ds.features, ds.labels)
    assert a == b


def test_noise_raises_disagreement():
    clean = build_dataset(5, None, 0.0)
    noisy = build_dataset(5, "uniform", 0.30)
    fp_clean = compute_fingerprint(clean.features, clean.labels)
    fp_noisy = compute_fingerprint(noisy.features, noisy.labels)
    assert fp_noisy["disagree_rate"] > fp_clean["disagree_rate"]


def test_neighbour_cache_shape_and_no_self():
    ds = build_dataset(5, None, 0.0, n=60)
    idx, dist = neighbour_cache(ds.features)
    assert len(idx) == ds.n
    for i in range(ds.n):
        assert len(idx[i]) == 25
        assert i not in idx[i]
        assert dist[i] == sorted(dist[i])


def test_majority_label_clear_winner():
    assert majority_label([1, 2, 2, 1, 2]) == 2


def test_majority_label_tie_goes_to_nearest_supporter():
    assert majority_label([3, 1, 1, 3]) == 3


def test_loo_predictions_on_separable_toy_data():
    features = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                [5.0, 5.0], [5.1, 5.0], [5.0, 5.1]]
    labels = [0, 0, 0, 1, 1, 1]
    cache = neighbour_cache(features)
    preds = loo_predictions(labels, cache[0], k=2)
    assert preds == labels


def test_vote_margins_in_unit_interval():
    ds = build_dataset(5, "uniform", 0.20)
    cache = neighbour_cache(ds.features)
    margins = vote_margins(ds.labels, cache[0])
    assert len(margins) == ds.n
    for m in margins:
        assert 0.0 <= m <= 1.0


def test_detect_dup_pairs_recovers_planted_pairs():
    ds = build_dataset(5, None, 0.0)
    cache = neighbour_cache(ds.features)
    detected = detect_dup_pairs(cache[0], cache[1])
    planted = set(ds.dup_pairs)
    found = sum(1 for pair in detected if pair in planted)
    assert found >= int(0.9 * len(planted))


def test_margin_profile_covers_all_points():
    ds = build_dataset(5, "uniform", 0.20)
    profile = margin_profile(ds.features, ds.labels)
    assert len(profile) == 10
    assert sum(p[2] for p in profile) == ds.n
    for _margin, rate, _count in profile:
        assert 0.0 <= rate <= 1.0


def test_boundary_noise_concentrates_in_low_margins():
    ds = build_dataset(5, "boundary", 0.30)
    fp = compute_fingerprint(ds.features, ds.labels)
    assert fp["low_margin_share"] > 0.45


def test_clean_data_has_low_dup_conflict():
    ds = build_dataset(5, None, 0.0)
    fp = compute_fingerprint(ds.features, ds.labels)
    assert fp["dup_conflict_rate"] == 0.0
