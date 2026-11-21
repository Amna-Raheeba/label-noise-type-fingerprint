"""Tests for the evaluation harness (small, fast configurations)."""

from noise_fingerprint.evaluate import evaluate_datasets, run_evaluation
from noise_fingerprint.models import NOISE_TYPES


def test_run_evaluation_structure():
    results = run_evaluation(seeds=(11,), rates=(0.10, 0.20), n=300)
    assert results["aggregate"]["n_datasets"] == 8
    assert len(results["records"]) == 8
    assert 0.0 <= results["aggregate"]["accuracy"] <= 1.0
    assert 0.0 <= results["aggregate"]["baseline_accuracy"] <= 1.0
    assert results["aggregate"]["rate_mae"] >= 0.0
    for noise_type in NOISE_TYPES:
        assert noise_type in results["aggregate"]["per_type"]


def test_record_fields():
    records = evaluate_datasets([(101, "uniform", 0.20)], n=300)
    assert len(records) == 1
    record = records[0]
    assert record["noise_type"] == "uniform"
    assert record["planted_rate"] == 0.20
    assert record["predicted_type"] in NOISE_TYPES
    assert record["correct"] == (
        record["predicted_type"] == "uniform")
    assert abs(record["rate_error"]
               - abs(record["estimated_rate"] - 0.20)) < 1e-9


def test_confusion_counts_sum_to_per_type_n():
    results = run_evaluation(seeds=(11,), rates=(0.10,), n=300)
    agg = results["aggregate"]
    for noise_type in NOISE_TYPES:
        total = sum(agg["confusion"][noise_type].values())
        assert total == agg["per_type"][noise_type]["n"]
