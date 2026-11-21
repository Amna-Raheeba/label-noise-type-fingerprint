"""Build the shipped data files: scenarios.json + demo_case.json.

Run from the repository root:  python scripts/build_cases.py
"""

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from noise_fingerprint.diagnose import diagnose
from noise_fingerprint.generate import demo_dataset
from noise_fingerprint.models import (
    BANK_RATES,
    DEFAULT_RATES,
    DEFAULT_SEEDS,
    FINGERPRINT_KEYS,
    NOISE_TYPES,
)


def main():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
    )
    os.makedirs(data_dir, exist_ok=True)
    scenarios = {
        "noise_types": list(NOISE_TYPES),
        "rates": list(DEFAULT_RATES),
        "seeds": list(DEFAULT_SEEDS),
        "bank_rates": list(BANK_RATES),
        "fingerprint_keys": list(FINGERPRINT_KEYS),
        "n_points": 420,
        "n_features": 6,
        "n_classes": 4,
    }
    with io.open(os.path.join(data_dir, "scenarios.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(scenarios, fh, indent=2, sort_keys=True)
    dataset = demo_dataset()
    verdict = diagnose(dataset.features, dataset.labels)
    demo = {
        "seed": dataset.seed,
        "noise_type": dataset.noise_type,
        "planted_rate": dataset.planted_rate,
        "actual_rate": dataset.actual_rate,
        "n_points": dataset.n,
        "n_dup_pairs": len(dataset.dup_pairs),
        "diagnosis": {
            "predicted_type": verdict.predicted_type,
            "confidence": verdict.confidence,
            "estimated_rate": verdict.estimated_rate,
            "distances": verdict.distances,
            "fingerprint": verdict.fingerprint,
        },
    }
    with io.open(os.path.join(data_dir, "demo_case.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(demo, fh, indent=2, sort_keys=True)
    print("Wrote data/scenarios.json and data/demo_case.json")
    print("Demo diagnosis: %s (planted %s at %.2f)"
          % (verdict.predicted_type, dataset.noise_type,
             dataset.planted_rate))
    return 0


if __name__ == "__main__":
    sys.exit(main())
