"""Command line interface: demo and evaluate."""

import argparse
import csv
import io
import json
import os
import sys

from .diagnose import diagnose
from .evaluate import run_evaluation
from .generate import demo_dataset
from .report import dataset_report, evaluation_summary, headline


def run_demo():
    dataset = demo_dataset()
    verdict = diagnose(dataset.features, dataset.labels)
    sys.stdout.write(dataset_report(dataset, verdict))
    return 0


def run_evaluate(out_dir):
    results = run_evaluation()
    os.makedirs(out_dir, exist_ok=True)
    with io.open(os.path.join(out_dir, "evaluation.json"), "w",
                 encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    with io.open(os.path.join(out_dir, "evaluation.csv"), "w",
                 encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "seed", "noise_type", "planted_rate", "actual_rate",
            "predicted_type", "correct", "baseline_type",
            "baseline_correct", "estimated_rate", "rate_error",
            "confidence",
        ])
        for r in results["records"]:
            writer.writerow([
                r["seed"], r["noise_type"], "%.2f" % r["planted_rate"],
                "%.4f" % r["actual_rate"], r["predicted_type"],
                int(r["correct"]), r["baseline_type"],
                int(r["baseline_correct"]),
                "%.4f" % r["estimated_rate"], "%.4f" % r["rate_error"],
                "%.4f" % r["confidence"],
            ])
    with io.open(os.path.join(out_dir, "evaluation_summary.md"), "w",
                 encoding="utf-8") as fh:
        fh.write(evaluation_summary(results))
    with io.open(os.path.join(out_dir, "HEADLINE.md"), "w",
                 encoding="utf-8") as fh:
        fh.write(headline(results))
    agg = results["aggregate"]
    print("Evaluated %d datasets." % agg["n_datasets"])
    print("Type accuracy: %.3f" % agg["accuracy"])
    print("Baseline accuracy: %.3f" % agg["baseline_accuracy"])
    print("Rate MAE: %.3f" % agg["rate_mae"])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="noise_fingerprint",
        description="Fingerprint the type of label noise in a "
                    "tabular dataset.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="diagnose the seeded demo dataset")
    ev = sub.add_parser("evaluate", help="run the headline evaluation")
    ev.add_argument("--out", default="results",
                    help="output directory (default: results)")
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return run_evaluate(args.out)
    return run_demo()
