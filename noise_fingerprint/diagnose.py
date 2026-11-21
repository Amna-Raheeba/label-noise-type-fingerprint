"""Diagnosis: match a dataset's fingerprint against a reference bank.

The bank is a parametric bootstrap on the target itself:

1. A proxy-clean labelling is estimated by iterated kNN smoothing,
   and a Gaussian class model (per-class centroids, shared pooled
   variance, class priors) is fitted to the target's features under
   that labelling.
2. Bank base labels are *sampled from the fitted posterior at the
   target's own points* -- so the bank's clean labellings have
   genuine overlap and a realistic margin distribution on exactly
   the target's feature layout (near-duplicates share one sampled
   label), instead of the over-smooth geometry a smoothed proxy
   would give.
3. Each of the four noise processes is planted on the sampled bases
   at a grid of rates and fingerprinted.  The target is classified
   by the nearest rate-matched reference fingerprint (standardised
   Euclidean), with a confidence margin from the runner-up distance.
   The noise rate is estimated by inverting the bank's
   disagreement-vs-rate curve.
"""

import math
import random

from .fingerprint import (
    cluster_assignment,
    compute_fingerprint,
    detect_dup_pairs,
    majority_label,
    neighbour_cache,
)
from .generate import apply_noise, flip_probabilities
from .models import (
    BANK_RATES,
    BANK_SEEDS,
    FINGERPRINT_KEYS,
    NOISE_TYPES,
    REMEDIES,
    Diagnosis,
)

RATE_WINDOW = 0.075   # bank entries within this of the estimated rate
SMOOTH_K = 10
SMOOTH_ROUNDS = 2
POSTERIOR_TEMPERATURE = 0.75   # < 1 sharpens the sampled base labels


def proxy_clean_labels(labels, cache_idx, rounds=SMOOTH_ROUNDS,
                       k=SMOOTH_K):
    """Estimate the clean labels by iterated neighbourhood smoothing.

    Each round replaces every label with the majority label of its
    k nearest neighbours (from the previous round).  Planted noise is
    (mostly) locally outvoted; the result is used to *fit the class
    model*, not as the bank base itself.
    """
    current = list(labels)
    for _ in range(rounds):
        nxt = []
        for i in range(len(labels)):
            neigh = [current[j] for j in cache_idx[i][:k]]
            nxt.append(majority_label(neigh))
        current = nxt
    return current


def fit_class_model(features, labels):
    """Per-class centroids, shared pooled variance, class priors."""
    classes = sorted(set(labels))
    d = len(features[0])
    n = len(features)
    centroids = {}
    priors = {}
    for c in classes:
        members = [features[i] for i in range(n) if labels[i] == c]
        priors[c] = len(members) / float(n)
        centroids[c] = [sum(x[j] for x in members) / float(len(members))
                        for j in range(d)]
    var = 0.0
    count = 0
    for i in range(n):
        cen = centroids[labels[i]]
        for j in range(d):
            diff = features[i][j] - cen[j]
            var += diff * diff
            count += 1
    variance = var / float(count) if count else 1.0
    return {"classes": classes, "centroids": centroids,
            "priors": priors, "variance": max(variance, 1e-6)}


def _log_posteriors(features, model):
    """Per point: unnormalised log posterior per class."""
    classes = model["classes"]
    centroids = model["centroids"]
    priors = model["priors"]
    var = model["variance"]
    out = []
    for x in features:
        logps = []
        for c in classes:
            cen = centroids[c]
            d2 = 0.0
            for a, b in zip(x, cen):
                diff = a - b
                d2 += diff * diff
            logps.append(-d2 / (2.0 * var) + math.log(priors[c]))
        out.append(logps)
    return out


def sample_base_labels(features, model, dup_pairs, rng):
    """Sample one clean labelling from the fitted posterior.

    Points are drawn independently from their posterior, except
    near-duplicates, which share their source point's draw (a
    duplicate of a point carries that point's clean label).
    """
    classes = model["classes"]
    dup_of = {}
    for (i, j) in dup_pairs:
        dup_of[j] = i
    labels = [None] * len(features)
    all_logps = _log_posteriors(features, model)
    for i in range(len(features)):
        if i in dup_of and labels[dup_of[i]] is not None:
            labels[i] = labels[dup_of[i]]
            continue
        logps = all_logps[i]
        top = max(logps)
        weights = [math.exp((lp - top) / POSTERIOR_TEMPERATURE)
                   for lp in logps]
        total = sum(weights)
        draw = rng.random() * total
        acc = 0.0
        chosen = classes[-1]
        for c, w in zip(classes, weights):
            acc += w
            if draw <= acc:
                chosen = c
                break
        labels[i] = chosen
    return labels


def _argmax_labels(features, model):
    """Posterior-mode label per point (the fitted geometry's labelling)."""
    classes = model["classes"]
    out = []
    for logps in _log_posteriors(features, model):
        best = max(range(len(classes)), key=lambda idx: logps[idx])
        out.append(classes[best])
    return out


def _boundary_probs(features, model, rate, rng):
    """Boundary flip probabilities for the bank.

    Computed with the *same* margin formula the target planting
    uses (centroid margins, exponential fall-off), but on the fitted
    model's posterior-mode labels -- a smooth geometric labelling --
    rather than on the noisy sampled base labels.
    """
    mode_labels = _argmax_labels(features, model)
    return flip_probabilities(features, mode_labels, "boundary", rate,
                              rng)


def build_bank(features, model, dup_pairs, cache, clusters,
               rates=BANK_RATES, seeds=BANK_SEEDS):
    """Reference fingerprints: every noise type x rate x seed.

    Each entry draws a fresh posterior-sampled base labelling, so the
    references average over the base-labelling uncertainty.
    """
    entries = []
    for noise_type in NOISE_TYPES:
        for rate in rates:
            for seed in seeds:
                rng = random.Random("bank:%s:%.2f:%d"
                                    % (noise_type, rate, seed))
                base = sample_base_labels(features, model, dup_pairs,
                                          rng)
                override = None
                if noise_type == "boundary":
                    override = _boundary_probs(features, model, rate, rng)
                noisy, _flipped, _info = apply_noise(
                    features, base, noise_type, rate, rng,
                    probs_override=override)
                fp = compute_fingerprint(features, noisy, cache=cache,
                                         clusters=clusters)
                entries.append({
                    "noise_type": noise_type,
                    "rate": rate,
                    "fingerprint": fp,
                })
    return entries


def estimate_rate(target_fingerprint, bank):
    """Invert the bank's disagreement-vs-rate curve at the target."""
    points = sorted(
        (entry["fingerprint"]["disagree_rate"], entry["rate"])
        for entry in bank
    )
    target = target_fingerprint["disagree_rate"]
    if target <= points[0][0]:
        return max(0.0, points[0][1]
                   * (target / points[0][0] if points[0][0] > 0 else 0.0))
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x0 <= target <= x1:
            if x1 == x0:
                return y0
            frac = (target - x0) / (x1 - x0)
            return y0 + frac * (y1 - y0)
    return points[-1][1]


def _reference_at(bank, noise_type, rate):
    """Mean bank fingerprint for one (type, rate) cell."""
    cell = [e for e in bank
            if e["noise_type"] == noise_type and e["rate"] == rate]
    ref = {}
    for key in FINGERPRINT_KEYS:
        ref[key] = (sum(e["fingerprint"][key] for e in cell)
                    / float(len(cell)))
    return ref


def _component_scales(bank):
    scales = {}
    for key in FINGERPRINT_KEYS:
        values = [e["fingerprint"][key] for e in bank]
        mean = sum(values) / float(len(values))
        var = sum((v - mean) ** 2 for v in values) / float(len(values))
        scales[key] = max(math.sqrt(var), 1e-3)
    return scales


def diagnose(features, labels):
    """Full diagnosis of one labelled dataset.  Returns a Diagnosis."""
    cache = neighbour_cache(features)
    clusters = cluster_assignment(features)
    target_fp = compute_fingerprint(features, labels, cache=cache,
                                    clusters=clusters)
    proxy = proxy_clean_labels(labels, cache[0])
    model = fit_class_model(features, proxy)
    dup_pairs = detect_dup_pairs(cache[0], cache[1])
    bank = build_bank(features, model, dup_pairs, cache, clusters)
    scales = _component_scales(bank)

    def distance_to(ref):
        total = 0.0
        for key in FINGERPRINT_KEYS:
            z = (target_fp[key] - ref[key]) / scales[key]
            total += z * z
        return math.sqrt(total / len(FINGERPRINT_KEYS))

    # Joint (type, rate) match: each type converts flips into visible
    # disagreement at a different rate (boundary flips hide among
    # rival neighbours), so a single pooled rate estimate would match
    # every type at the wrong rate.  Instead, score every bank cell
    # and take each type's best cell as its distance.
    distances = {}
    best_cell = {}
    for noise_type in NOISE_TYPES:
        per_rate = {}
        for rate in BANK_RATES:
            per_rate[rate] = distance_to(
                _reference_at(bank, noise_type, rate))
        best_rate = min(BANK_RATES, key=lambda r: per_rate[r])
        distances[noise_type] = per_rate[best_rate]
        best_cell[noise_type] = best_rate
    ranked = sorted(NOISE_TYPES, key=lambda t: distances[t])
    winner = ranked[0]
    # The rate estimate refines the winning cell by inverting the
    # winning type's own disagreement curve.
    own_curve = [e for e in bank if e["noise_type"] == winner]
    estimated = estimate_rate(target_fp, own_curve)
    best = ranked[0]
    d_best = distances[best]
    d_second = distances[ranked[1]]
    confidence = (d_second - d_best) / (d_second + 1e-9)
    return Diagnosis(best, confidence, estimated, target_fp, distances,
                     REMEDIES[best])
