"""Notebook 1 (runnable .py): the demo dataset, diagnosed live.

Builds the fixed demo dataset (boundary noise at 20%), fingerprints
it, matches it against the reference bank and prints the audit
report.  Run:  python notebooks/01_demo_run.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noise_fingerprint.diagnose import diagnose
from noise_fingerprint.generate import demo_dataset
from noise_fingerprint.report import dataset_report


def main():
    dataset = demo_dataset()
    verdict = diagnose(dataset.features, dataset.labels)
    sys.stdout.write(dataset_report(dataset, verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
