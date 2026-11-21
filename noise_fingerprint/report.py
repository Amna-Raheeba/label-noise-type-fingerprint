"""Human-readable reports: demo audit, evaluation summary, headline."""

from .models import FINGERPRINT_KEYS, NOISE_TYPES


def dataset_report(dataset, verdict):
    """Audit report for one dataset (ground truth shown: demo only)."""
    lines = []
    lines.append("Label-noise fingerprint audit")
    lines.append("=============================")
    lines.append("")
    lines.append("Dataset: %d points, seed %d"
                 % (dataset.n, dataset.seed))
    lines.append("Planted ground truth: %s at rate %.2f "
                 "(actual flipped %.3f)"
                 % (dataset.noise_type, dataset.planted_rate,
                    dataset.actual_rate))
    lines.append("")
    lines.append("Fingerprint:")
    for key in FINGERPRINT_KEYS:
        lines.append("  %-22s %.4f" % (key, verdict.fingerprint[key]))
    lines.append("")
    lines.append("Bank distances (lower = closer match):")
    for noise_type in NOISE_TYPES:
        lines.append("  %-22s %.4f"
                     % (noise_type, verdict.distances[noise_type]))
    lines.append("")
    lines.append("Diagnosis: %s (confidence %.3f)"
                 % (verdict.predicted_type, verdict.confidence))
    lines.append("Estimated noise rate: %.3f" % verdict.estimated_rate)
    lines.append("Remedy: %s" % verdict.remedy)
    lines.append("")
    return "\n".join(lines)


def evaluation_summary(results):
    agg = results["aggregate"]
    cfg = results["config"]
    lines = []
    lines.append("# Evaluation summary")
    lines.append("")
    lines.append("%d seeded datasets (%d noise types x %d rates x "
                 "%d seeds), %d generated points plus planted "
                 "near-duplicates each, planted ground truth."
                 % (agg["n_datasets"], len(cfg["noise_types"]),
                    len(cfg["rates"]), len(cfg["seeds"]), cfg["n"]))
    lines.append("")
    lines.append("## Noise-type classification accuracy")
    lines.append("")
    lines.append("| noise type | n | fingerprinter | matrix baseline |")
    lines.append("|---|---|---|---|")
    for noise_type in NOISE_TYPES:
        entry = agg["per_type"].get(noise_type)
        if entry is None:
            continue
        lines.append("| %s | %d | %.3f | %.3f |"
                     % (noise_type, entry["n"], entry["accuracy"],
                        entry["baseline_accuracy"]))
    lines.append("| **overall** | %d | **%.3f** | **%.3f** |"
                 % (agg["n_datasets"], agg["accuracy"],
                    agg["baseline_accuracy"]))
    lines.append("")
    lines.append("## Rate estimation")
    lines.append("")
    lines.append("Mean absolute error of the estimated noise rate "
                 "vs the planted rate: **%.3f** (fingerprinter; the "
                 "matrix baseline does not estimate a type-specific "
                 "rate)." % agg["rate_mae"])
    lines.append("")
    lines.append("| noise type | rate MAE |")
    lines.append("|---|---|")
    for noise_type in NOISE_TYPES:
        entry = agg["per_type"].get(noise_type)
        if entry is None:
            continue
        lines.append("| %s | %.3f |" % (noise_type, entry["rate_mae"]))
    lines.append("")
    lines.append("## Diagnosis confusion (true -> predicted counts)")
    lines.append("")
    for noise_type in NOISE_TYPES:
        row = agg["confusion"].get(noise_type, {})
        parts = ["%s: %d" % (pred, row[pred]) for pred in sorted(row)]
        lines.append("- %s -> %s" % (noise_type, ", ".join(parts)))
    lines.append("")
    return "\n".join(lines)


def headline(results):
    agg = results["aggregate"]
    lines = []
    lines.append("# Headline results")
    lines.append("")
    lines.append("Across %d seeded datasets (4 noise types x rates "
                 "5-30%% x %d seeds):"
                 % (agg["n_datasets"],
                    len(results["config"]["seeds"])))
    lines.append("")
    lines.append("- Noise-type classification accuracy: **%.3f**"
                 % agg["accuracy"])
    lines.append("- Cleanlab-style matrix-inspection baseline: **%.3f**"
                 % agg["baseline_accuracy"])
    for noise_type in NOISE_TYPES:
        entry = agg["per_type"].get(noise_type)
        if entry is None:
            continue
        lines.append("- %s: %.3f (baseline %.3f)"
                     % (noise_type, entry["accuracy"],
                        entry["baseline_accuracy"]))
    lines.append("- Noise-rate estimation MAE: **%.3f**" % agg["rate_mae"])
    lines.append("")
    lines.append("All numbers are measured on synthetic seeded data; "
                 "see README section 7 for limits.")
    lines.append("")
    return "\n".join(lines)
