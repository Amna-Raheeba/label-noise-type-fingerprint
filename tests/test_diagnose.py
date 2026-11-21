"""Tests for the bank-matching diagnosis."""

from noise_fingerprint.diagnose import (
    diagnose,
    fit_class_model,
    proxy_clean_labels,
    sample_base_labels,
)
from noise_fingerprint.fingerprint import (
    detect_dup_pairs,
    neighbour_cache,
)
from noise_fingerprint.generate import build_dataset, demo_dataset
from noise_fingerprint.models import NOISE_TYPES, REMEDIES

import random


def test_demo_dataset_diagnosed_as_boundary():
    ds = demo_dataset()
    verdict = diagnose(ds.features, ds.labels)
    assert verdict.predicted_type == "boundary"


def test_strong_uniform_noise_diagnosed():
    ds = build_dataset(101, "uniform", 0.30)
    verdict = diagnose(ds.features, ds.labels)
    assert verdict.predicted_type == "uniform"


def test_strong_class_noise_diagnosed():
    ds = build_dataset(202, "class_dependent", 0.30)
    verdict = diagnose(ds.features, ds.labels)
    assert verdict.predicted_type == "class_dependent"


def test_strong_instance_noise_diagnosed():
    ds = build_dataset(101, "instance_dependent", 0.30)
    verdict = diagnose(ds.features, ds.labels)
    assert verdict.predicted_type == "instance_dependent"


def test_diagnosis_fields_are_sane():
    ds = build_dataset(101, "uniform", 0.20)
    verdict = diagnose(ds.features, ds.labels)
    assert verdict.predicted_type in NOISE_TYPES
    assert 0.0 <= verdict.confidence <= 1.0
    assert 0.0 <= verdict.estimated_rate <= 0.35
    assert set(verdict.distances.keys()) == set(NOISE_TYPES)
    assert verdict.remedy == REMEDIES[verdict.predicted_type]
    best = min(verdict.distances, key=lambda t: verdict.distances[t])
    assert best == verdict.predicted_type


def test_proxy_labels_agree_more_with_clean_than_noisy_do():
    ds = build_dataset(101, "uniform", 0.25)
    cache = neighbour_cache(ds.features)
    proxy = proxy_clean_labels(ds.labels, cache[0])
    agree_proxy = sum(1 for a, b in zip(proxy, ds.clean_labels)
                      if a == b) / float(ds.n)
    agree_noisy = sum(1 for a, b in zip(ds.labels, ds.clean_labels)
                      if a == b) / float(ds.n)
    assert agree_proxy > agree_noisy


def test_fit_class_model_priors_sum_to_one():
    ds = build_dataset(101, None, 0.0)
    model = fit_class_model(ds.features, ds.clean_labels)
    assert abs(sum(model["priors"].values()) - 1.0) < 1e-9
    assert set(model["classes"]) == {0, 1, 2, 3}
    assert model["variance"] > 0.0


def test_sampled_base_labels_share_duplicates():
    ds = build_dataset(101, None, 0.0)
    cache = neighbour_cache(ds.features)
    model = fit_class_model(ds.features, ds.clean_labels)
    dup_pairs = detect_dup_pairs(cache[0], cache[1])
    labels = sample_base_labels(ds.features, model, dup_pairs,
                                random.Random(0))
    assert len(labels) == ds.n
    for (i, j) in dup_pairs:
        assert labels[i] == labels[j]


def test_sampled_base_labels_are_valid_classes():
    ds = build_dataset(101, None, 0.0)
    model = fit_class_model(ds.features, ds.clean_labels)
    labels = sample_base_labels(ds.features, model, [],
                                random.Random(1))
    assert set(labels) <= {0, 1, 2, 3}
