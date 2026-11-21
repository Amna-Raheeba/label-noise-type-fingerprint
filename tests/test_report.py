"""Tests for the human-readable reports."""

from noise_fingerprint.diagnose import diagnose
from noise_fingerprint.evaluate import run_evaluation
from noise_fingerprint.generate import demo_dataset
from noise_fingerprint.report import (
    dataset_report,
    evaluation_summary,
    headline,
)


def test_dataset_report_contents():
    ds = demo_dataset()
    verdict = diagnose(ds.features, ds.labels)
    text = dataset_report(ds, verdict)
    assert "Label-noise fingerprint audit" in text
    assert "Diagnosis: boundary" in text
    assert "Estimated noise rate:" in text
    assert "Remedy:" in text


def test_evaluation_summary_contents():
    results = run_evaluation(seeds=(11,), rates=(0.10,), n=300)
    text = evaluation_summary(results)
    assert "## Noise-type classification accuracy" in text
    assert "## Rate estimation" in text
    assert "## Diagnosis confusion" in text
    for noise_type in ("uniform", "class_dependent",
                       "instance_dependent", "boundary"):
        assert noise_type in text


def test_headline_contents():
    results = run_evaluation(seeds=(11,), rates=(0.10,), n=300)
    text = headline(results)
    assert "# Headline results" in text
    assert "Noise-type classification accuracy" in text
    assert "Noise-rate estimation MAE" in text
