"""Headline-drift guard.

Re-runs the full seeded evaluation and compares the aggregate
headline numbers against the shipped results/evaluation.json.
Exits non-zero if any headline value moved by more than the
tolerance -- i.e. if the code no longer reproduces the published
results.

Run from the repository root:  python scripts/reproduce_results.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noise_fingerprint.evaluate import run_evaluation
from noise_fingerprint.models import NOISE_TYPES

TOLERANCE = 0.02


def headline_values(results):
    agg = results["aggregate"]
    values = {
        "accuracy": agg["accuracy"],
        "baseline_accuracy": agg["baseline_accuracy"],
        "rate_mae": agg["rate_mae"],
    }
    for noise_type in NOISE_TYPES:
        entry = agg["per_type"].get(noise_type)
        if entry is None:
            continue
        values["accuracy_%s" % noise_type] = entry["accuracy"]
        values["baseline_%s" % noise_type] = entry["baseline_accuracy"]
        values["rate_mae_%s" % noise_type] = entry["rate_mae"]
    return values


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    shipped_path = os.path.join(root, "results", "evaluation.json")
    with io.open(shipped_path, encoding="utf-8") as fh:
        shipped = json.load(fh)
    fresh = run_evaluation()
    expected = headline_values(shipped)
    actual = headline_values(fresh)
    failures = []
    for key in sorted(expected):
        if key not in actual:
            failures.append("%s missing from fresh run" % key)
        elif abs(expected[key] - actual[key]) > TOLERANCE:
            failures.append(
                "%s: shipped %.4f vs fresh %.4f"
                % (key, expected[key], actual[key])
            )
    if failures:
        print("HEADLINE DRIFT DETECTED:")
        for failure in failures:
            print("  " + failure)
        return 1
    print("All %d headline values reproduce within %.2f."
          % (len(expected), TOLERANCE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
