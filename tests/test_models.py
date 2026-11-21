"""Tests for shared constants and data structures."""

from noise_fingerprint.models import (
    BANK_RATES,
    CLEAN,
    DEFAULT_RATES,
    FINGERPRINT_KEYS,
    NOISE_TYPES,
    REMEDIES,
    Dataset,
)


def test_four_noise_types():
    assert len(NOISE_TYPES) == 4
    assert set(NOISE_TYPES) == {
        "uniform", "class_dependent", "instance_dependent", "boundary"}


def test_remedy_for_every_type():
    for noise_type in NOISE_TYPES:
        assert noise_type in REMEDIES
        assert len(REMEDIES[noise_type]) > 20
    assert CLEAN in REMEDIES


def test_fingerprint_keys_unique():
    assert len(FINGERPRINT_KEYS) == len(set(FINGERPRINT_KEYS))
    assert len(FINGERPRINT_KEYS) == 10


def test_rate_grids_are_sane():
    assert list(DEFAULT_RATES) == sorted(DEFAULT_RATES)
    assert list(BANK_RATES) == sorted(BANK_RATES)
    for rate in DEFAULT_RATES:
        assert 0.0 < rate <= 0.35


def test_dataset_actual_rate():
    ds = Dataset([[0.0], [1.0]], [0, 1], [0, 0], [False, True],
                 "uniform", 0.5, [], seed=1)
    assert ds.n == 2
    assert ds.actual_rate == 0.5


def test_dataset_actual_rate_empty_flips():
    ds = Dataset([[0.0]], [0], [0], [], CLEAN, 0.0, [], seed=1)
    assert ds.actual_rate == 0.0
