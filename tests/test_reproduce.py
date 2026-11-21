"""Tests for the headline-drift guard in scripts/reproduce_results.py."""

import importlib.util
import os

from noise_fingerprint.evaluate import run_evaluation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_guard():
    path = os.path.join(ROOT, "scripts", "reproduce_results.py")
    spec = importlib.util.spec_from_file_location("reproduce_results",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_headline_values_cover_all_types():
    guard = _load_guard()
    values = guard.headline_values(
        run_evaluation(seeds=(11,), rates=(0.10,), n=300))
    assert values["accuracy"] >= 0.0
    assert "baseline_accuracy" in values
    assert "rate_mae" in values
    assert "accuracy_boundary" in values
    assert "accuracy_instance_dependent" in values
    assert "rate_mae_uniform" in values


def test_guard_reproduces_shipped_results():
    guard = _load_guard()
    assert guard.main() == 0
