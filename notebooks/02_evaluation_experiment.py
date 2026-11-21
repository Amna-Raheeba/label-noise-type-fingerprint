"""Notebook 2 (runnable .py): reduced evaluation experiment.

Runs the headline evaluation on a reduced grid (2 seeds x 3 rates
instead of 3 seeds x 6 rates, so CI stays fast) and prints the
per-type accuracy of the fingerprinter and of the noise-matrix
baseline, plus the rate-estimation error.  The shipped results/
come from the full grid via `python -m noise_fingerprint evaluate`.
Run:  python notebooks/02_evaluation_experiment.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noise_fingerprint.evaluate import run_evaluation
from noise_fingerprint.models import NOISE_TYPES


def main():
    results = run_evaluation(seeds=(11, 23), rates=(0.10, 0.20, 0.30))
    agg = results["aggregate"]
    print("Reduced evaluation: %d datasets" % agg["n_datasets"])
    print("  %-20s %10s %10s" % ("noise type", "fingerprint", "baseline"))
    for noise_type in NOISE_TYPES:
        entry = agg["per_type"][noise_type]
        print("  %-20s %10.3f %10.3f"
              % (noise_type, entry["accuracy"],
                 entry["baseline_accuracy"]))
    print("  %-20s %10.3f %10.3f"
          % ("overall", agg["accuracy"], agg["baseline_accuracy"]))
    print("Rate MAE: %.3f" % agg["rate_mae"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
