"""Headline evaluation: planted noise, diagnosed blind.

For every (seed, noise type, rate) combination a dataset is
generated with known ground truth, then diagnosed by the
fingerprinter and by the noise-matrix baseline -- neither sees the
planted type.  Aggregates: type-classification accuracy (overall
and per type), rate-estimation error, and the diagnosis confusion
counts.
"""

from .baselines import matrix_heuristic
from .diagnose import diagnose
from .generate import build_dataset
from .models import DEFAULT_N, DEFAULT_RATES, DEFAULT_SEEDS, NOISE_TYPES


def _dataset_seed(seed, type_index, rate):
    return seed * 100000 + type_index * 1000 + int(round(rate * 100))


def evaluate_datasets(specs, n=DEFAULT_N):
    """Run the diagnosis experiment over explicit dataset specs.

    ``specs`` is a list of (seed, noise_type, rate) triples.
    """
    records = []
    for seed, noise_type, rate in specs:
        dataset = build_dataset(seed, noise_type, rate, n=n)
        verdict = diagnose(dataset.features, dataset.labels)
        base_type, _pair = matrix_heuristic(dataset.features,
                                            dataset.labels)
        records.append({
            "seed": seed,
            "noise_type": noise_type,
            "planted_rate": rate,
            "actual_rate": dataset.actual_rate,
            "predicted_type": verdict.predicted_type,
            "correct": verdict.predicted_type == noise_type,
            "baseline_type": base_type,
            "baseline_correct": base_type == noise_type,
            "estimated_rate": verdict.estimated_rate,
            "rate_error": abs(verdict.estimated_rate - rate),
            "confidence": verdict.confidence,
        })
    return records


def _aggregate(records):
    total = len(records)
    agg = {
        "n_datasets": total,
        "accuracy": (sum(1 for r in records if r["correct"])
                     / float(total) if total else 0.0),
        "baseline_accuracy": (sum(1 for r in records
                                  if r["baseline_correct"])
                              / float(total) if total else 0.0),
        "rate_mae": (sum(r["rate_error"] for r in records)
                     / float(total) if total else 0.0),
        "per_type": {},
        "confusion": {},
    }
    for noise_type in NOISE_TYPES:
        sub = [r for r in records if r["noise_type"] == noise_type]
        if not sub:
            continue
        agg["per_type"][noise_type] = {
            "n": len(sub),
            "accuracy": (sum(1 for r in sub if r["correct"])
                         / float(len(sub))),
            "baseline_accuracy": (sum(1 for r in sub
                                      if r["baseline_correct"])
                                  / float(len(sub))),
            "rate_mae": (sum(r["rate_error"] for r in sub)
                         / float(len(sub))),
        }
        row = {}
        for r in sub:
            row[r["predicted_type"]] = row.get(r["predicted_type"], 0) + 1
        agg["confusion"][noise_type] = row
    return agg


def run_evaluation(seeds=DEFAULT_SEEDS, rates=DEFAULT_RATES,
                   n=DEFAULT_N, noise_types=NOISE_TYPES):
    """The full seeded headline evaluation."""
    specs = []
    for seed in seeds:
        for ti, noise_type in enumerate(noise_types):
            for rate in rates:
                specs.append((_dataset_seed(seed, ti, rate),
                              noise_type, rate))
    records = evaluate_datasets(specs, n=n)
    results = {
        "records": records,
        "aggregate": _aggregate(records),
        "config": {
            "seeds": list(seeds),
            "rates": list(rates),
            "n": n,
            "noise_types": list(noise_types),
        },
    }
    return results
