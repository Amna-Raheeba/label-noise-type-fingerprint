"""Tests for the seeded generator and the four noise processes."""

import random

from noise_fingerprint.generate import (
    CLASS_PAIR,
    apply_noise,
    build_dataset,
    clean_margins,
    flip_probabilities,
)
from noise_fingerprint.models import CLEAN, NOISE_TYPES


def test_deterministic_same_seed():
    a = build_dataset(5, "uniform", 0.20)
    b = build_dataset(5, "uniform", 0.20)
    assert a.features == b.features
    assert a.labels == b.labels
    assert a.flipped == b.flipped


def test_different_seeds_differ():
    a = build_dataset(5, "uniform", 0.20)
    b = build_dataset(6, "uniform", 0.20)
    assert a.features != b.features


def test_clean_dataset_has_no_flips():
    ds = build_dataset(5, None, 0.0)
    assert ds.noise_type == CLEAN
    assert ds.actual_rate == 0.0
    assert ds.labels == ds.clean_labels


def test_dataset_size_includes_duplicates():
    ds = build_dataset(5, "uniform", 0.20, n=200)
    assert ds.n == 200 + len(ds.dup_pairs)
    assert len(ds.dup_pairs) == 24


def test_duplicate_pairs_share_clean_label():
    ds = build_dataset(5, "uniform", 0.20)
    for (i, j) in ds.dup_pairs:
        assert ds.clean_labels[i] == ds.clean_labels[j]


def test_each_type_flips_at_planted_rate():
    for noise_type in NOISE_TYPES:
        ds = build_dataset(5, noise_type, 0.20)
        assert ds.actual_rate > 0.05
        assert abs(ds.actual_rate - 0.20) < 0.09


def test_flipped_mask_matches_labels():
    ds = build_dataset(5, "uniform", 0.25)
    for i in range(ds.n):
        assert ds.flipped[i] == (ds.labels[i] != ds.clean_labels[i])


def test_class_dependent_destinations_are_the_planted_pairs():
    ds = build_dataset(5, "class_dependent", 0.60)
    flipped = [i for i in range(ds.n) if ds.flipped[i]]
    assert len(flipped) > 50
    for i in flipped:
        assert ds.labels[i] == CLASS_PAIR[ds.clean_labels[i]]


def test_boundary_flips_sit_at_lower_margins():
    margins = None
    means = {}
    for noise_type in ("uniform", "boundary"):
        ds = build_dataset(5, noise_type, 0.25)
        if margins is None:
            margins = clean_margins(ds.features, ds.clean_labels, 4)
        flipped = [margins[i] for i in range(ds.n) if ds.flipped[i]]
        means[noise_type] = sum(flipped) / float(len(flipped))
    assert means["boundary"] < means["uniform"]


def test_uniform_probabilities_are_constant():
    ds = build_dataset(5, None, 0.0)
    probs = flip_probabilities(ds.features, ds.clean_labels,
                               "uniform", 0.20, random.Random(0))
    assert all(p == 0.20 for p in probs)


def test_class_probabilities_depend_only_on_class():
    ds = build_dataset(5, None, 0.0)
    probs = flip_probabilities(ds.features, ds.clean_labels,
                               "class_dependent", 0.20,
                               random.Random(0))
    by_class = {}
    for p, y in zip(probs, ds.clean_labels):
        by_class.setdefault(y, set()).add(p)
    for y, values in by_class.items():
        assert len(values) == 1


def test_instance_probabilities_vary_by_point():
    ds = build_dataset(5, None, 0.0)
    probs = flip_probabilities(ds.features, ds.clean_labels,
                               "instance_dependent", 0.20,
                               random.Random(0))
    assert max(probs) > 2.0 * min(probs)
    mean = sum(probs) / len(probs)
    assert abs(mean - 0.20) < 0.03


def test_boundary_probabilities_fall_with_margin():
    ds = build_dataset(5, None, 0.0)
    probs = flip_probabilities(ds.features, ds.clean_labels,
                               "boundary", 0.20, random.Random(0))
    margins = clean_margins(ds.features, ds.clean_labels, 4)
    pairs = sorted(zip(margins, probs))
    low = [p for _m, p in pairs[:40]]
    high = [p for _m, p in pairs[-40:]]
    assert sum(low) / len(low) > 3.0 * (sum(high) / len(high))


def test_apply_noise_reports_expected_rate():
    ds = build_dataset(5, None, 0.0)
    _noisy, _flipped, info = apply_noise(
        ds.features, ds.clean_labels, "uniform", 0.20,
        random.Random(0))
    assert abs(info["expected_rate"] - 0.20) < 1e-9
