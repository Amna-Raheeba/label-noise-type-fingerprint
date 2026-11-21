"""Example: diagnose one seeded dataset end to end.

Run from the repository root:  python examples/run_audit.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noise_fingerprint.diagnose import diagnose
from noise_fingerprint.generate import build_dataset
from noise_fingerprint.report import dataset_report


def main():
    dataset = build_dataset(101, "instance_dependent", 0.20)
    verdict = diagnose(dataset.features, dataset.labels)
    sys.stdout.write(dataset_report(dataset, verdict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
