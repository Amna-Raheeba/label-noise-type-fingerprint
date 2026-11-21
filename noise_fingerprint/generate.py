"""Seeded dataset generation with planted label-noise processes.

Features are a Gaussian mixture with genuine class overlap (so a
real boundary zone exists).  A fraction of points are planted as
near-duplicates, which the duplicate-conflict fingerprint component
needs.  Noise is then applied by one of the four processes in
models.NOISE_TYPES; the flip mask is recorded as ground truth.

Everything is deterministic: all randomness flows from
``random.Random(seed)`` instances created here.
"""

import math
import random

from .models import (
    CLEAN,
    DEFAULT_CLASSES,
    DEFAULT_D,
    DEFAULT_N,
    Dataset,
)

CENTRE_SCALE = 1.30   # spread of the class centres
SIGMA = 1.10          # per-class isotropic spread (sets the overlap)
DUP_FRACTION = 0.12   # share of points planted as near-duplicates
DUP_JITTER = 0.02     # per-dimension jitter of a duplicate copy

# Class-dependent planting: fixed destination pairs and asymmetric
# per-source-class flip multipliers (mean 0.925, so the realised rate
# tracks the requested rate).
CLASS_PAIR = {0: 1, 1: 0, 2: 3, 3: 2}
CLASS_MULT = {0: 1.45, 1: 0.55, 2: 1.25, 3: 0.45}

IDN_BETA = 1.1        # strength of the instance-dependent direction
BOUNDARY_TAU = 0.10   # margin scale of the boundary process


def _centroids(features, labels, k_classes):
    d = len(features[0])
    sums = [[0.0] * d for _ in range(k_classes)]
    counts = [0] * k_classes
    for x, y in zip(features, labels):
        counts[y] += 1
        for j in range(d):
            sums[y][j] += x[j]
    out = []
    for c in range(k_classes):
        if counts[c] == 0:
            out.append([0.0] * d)
        else:
            out.append([s / counts[c] for s in sums[c]])
    return out


def _dist2(a, b):
    total = 0.0
    for x, y in zip(a, b):
        diff = x - y
        total += diff * diff
    return total


def clean_margins(features, labels, k_classes):
    """Geometric margin in [0, 1]: 0 at the boundary, 1 deep inside.

    margin = (d2 - d1) / (d2 + d1) with d1 the distance to the point's
    own class centroid and d2 the distance to the nearest rival
    centroid, both computed from the given (clean) labels.
    """
    cents = _centroids(features, labels, k_classes)
    out = []
    for x, y in zip(features, labels):
        d1 = math.sqrt(_dist2(x, cents[y]))
        d2 = None
        for c in range(k_classes):
            if c == y:
                continue
            dc = math.sqrt(_dist2(x, cents[c]))
            if d2 is None or dc < d2:
                d2 = dc
        out.append((d2 - d1) / (d2 + d1 + 1e-12))
    return out


def nearest_rival(features, labels, k_classes):
    """Index of the nearest rival class (by centroid) per point."""
    cents = _centroids(features, labels, k_classes)
    out = []
    for x, y in zip(features, labels):
        best = None
        best_d = None
        for c in range(k_classes):
            if c == y:
                continue
            dc = _dist2(x, cents[c])
            if best_d is None or dc < best_d:
                best_d = dc
                best = c
        out.append(best)
    return out


def flip_probabilities(features, labels, noise_type, rate, rng,
                       k_classes=DEFAULT_CLASSES):
    """Per-point flip probability for one noise process.

    Each process is normalised so the *expected* overall flip rate
    equals ``rate`` (before clipping at 0.95).
    """
    n = len(labels)
    if noise_type == "uniform":
        return [rate] * n
    if noise_type == "class_dependent":
        return [min(0.95, rate * CLASS_MULT[y]) for y in labels]
    if noise_type == "instance_dependent":
        d = len(features[0])
        w = [rng.gauss(0.0, 1.0) for _ in range(d)]
        norm = math.sqrt(sum(v * v for v in w)) or 1.0
        w = [v / norm for v in w]
        scores = [sum(wj * xj for wj, xj in zip(w, x)) for x in features]
        mean = sum(scores) / n
        var = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(var) or 1.0
        raw = [math.exp(IDN_BETA * (s - mean) / std) for s in scores]
        scale = rate / (sum(raw) / n)
        return [min(0.95, scale * r) for r in raw]
    if noise_type == "boundary":
        margins = clean_margins(features, labels, k_classes)
        raw = [math.exp(-max(m, 0.0) / BOUNDARY_TAU) for m in margins]
        scale = rate / (sum(raw) / n)
        return [min(0.95, scale * r) for r in raw]
    raise ValueError("unknown noise type: %s" % noise_type)


def flip_destination(features, labels, i, noise_type, rng,
                     k_classes=DEFAULT_CLASSES, rivals=None):
    """The wrong label a flipped point receives."""
    y = labels[i]
    if noise_type == "class_dependent":
        return CLASS_PAIR[y]
    if noise_type == "boundary":
        if rivals is None:
            rivals = nearest_rival(features, labels, k_classes)
        return rivals[i]
    others = [c for c in range(k_classes) if c != y]
    return others[rng.randrange(len(others))]


def apply_noise(features, labels, noise_type, rate, rng,
                k_classes=DEFAULT_CLASSES, probs_override=None):
    """Apply one noise process to clean labels.

    Returns (noisy_labels, flipped_mask, info).  Used both for the
    planted ground truth and for the reference bank (where ``labels``
    are posterior-sampled base labels).  ``probs_override`` replaces
    the process's own flip probabilities (already normalised to the
    requested rate); the bank uses it for the boundary process, whose
    probabilities it computes from the fitted posterior geometry
    instead of from sampled-label centroids.
    """
    if probs_override is not None:
        probs = probs_override
    else:
        probs = flip_probabilities(features, labels, noise_type, rate,
                                   rng, k_classes)
    rivals = None
    if noise_type == "boundary":
        rivals = nearest_rival(features, labels, k_classes)
    noisy = list(labels)
    flipped = [False] * len(labels)
    for i in range(len(labels)):
        if rng.random() < probs[i]:
            noisy[i] = flip_destination(features, labels, i, noise_type,
                                        rng, k_classes, rivals)
            flipped[i] = noisy[i] != labels[i]
    info = {"expected_rate": sum(probs) / float(len(labels))}
    return noisy, flipped, info


def build_features(rng, n, d, k_classes):
    """Gaussian-mixture features + clean labels + duplicate pairs."""
    centres = [[rng.gauss(0.0, CENTRE_SCALE) for _ in range(d)]
               for _ in range(k_classes)]
    features = []
    labels = []
    for _ in range(n):
        y = rng.randrange(k_classes)
        x = [centres[y][j] + rng.gauss(0.0, SIGMA) for j in range(d)]
        features.append(x)
        labels.append(y)
    dup_pairs = []
    n_dup = int(round(DUP_FRACTION * n))
    for _ in range(n_dup):
        src = rng.randrange(len(features))
        copy = [v + rng.gauss(0.0, DUP_JITTER) for v in features[src]]
        features.append(copy)
        labels.append(labels[src])
        dup_pairs.append((src, len(features) - 1))
    return features, labels, dup_pairs


def build_dataset(seed, noise_type=None, rate=0.0, n=DEFAULT_N,
                  d=DEFAULT_D, k_classes=DEFAULT_CLASSES):
    """One seeded dataset; ground truth recorded on the Dataset."""
    rng = random.Random(seed)
    features, clean_labels, dup_pairs = build_features(rng, n, d,
                                                       k_classes)
    if noise_type is None or rate <= 0.0:
        return Dataset(features, list(clean_labels), clean_labels,
                       [False] * len(clean_labels), CLEAN, 0.0,
                       dup_pairs, seed, aux={})
    noisy, flipped, info = apply_noise(features, clean_labels,
                                       noise_type, rate, rng, k_classes)
    aux = {"expected_rate": info["expected_rate"]}
    return Dataset(features, noisy, clean_labels, flipped, noise_type,
                   rate, dup_pairs, seed, aux=aux)


def demo_dataset():
    """The fixed demo: boundary noise at 20% on the demo seed."""
    return build_dataset(7, "boundary", 0.20)
