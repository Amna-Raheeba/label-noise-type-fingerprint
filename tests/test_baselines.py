"""Tests for the cleanlab-style matrix-inspection baseline."""

from noise_fingerprint.baselines import (
    estimated_noise_matrix,
    matrix_heuristic,
)
from noise_fingerprint.generate import build_dataset
from noise_fingerprint.models import NOISE_TYPES


def test_matrix_rows_are_distributions():
    ds = build_dataset(5, "uniform", 0.20)
    rates = estimated_noise_matrix(ds.features, ds.labels)
    for a in rates:
        assert abs(sum(rates[a].values()) - 1.0) < 1e-9


def test_heuristic_only_ever_answers_two_types():
    for noise_type in NOISE_TYPES:
        ds = build_dataset(5, noise_type, 0.20)
        predicted, _rate = matrix_heuristic(ds.features, ds.labels)
        assert predicted in ("uniform", "class_dependent")


def test_heuristic_catches_strong_class_noise():
    ds = build_dataset(101, "class_dependent", 0.30)
    predicted, pair_rate = matrix_heuristic(ds.features, ds.labels)
    assert predicted == "class_dependent"
    assert pair_rate >= 0.05


def test_heuristic_pair_rate_in_range():
    ds = build_dataset(5, "boundary", 0.20)
    _predicted, pair_rate = matrix_heuristic(ds.features, ds.labels)
    assert 0.0 <= pair_rate <= 1.0


def test_matrix_diagonal_dominates_on_clean_data():
    ds = build_dataset(5, None, 0.0)
    rates = estimated_noise_matrix(ds.features, ds.labels)
    for a in rates:
        assert rates[a][a] > 0.5
