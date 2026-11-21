"""Baseline: cleanlab-style noise-matrix inspection.

The standard practice this tool is compared against: estimate a
class-conditional noise matrix (here from leave-one-out kNN votes,
the model-free analogue of confident learning's confident joint),
inspect it, and call the noise class-dependent if any off-diagonal
pair rate stands out, uniform otherwise.

The heuristic is structurally blind to two of the four processes:
instance-dependent and boundary noise leave no single dominant
class pair, so the matrix inspection can only ever answer "uniform"
or "class_dependent".  That blindness is the baseline's finding,
not a bug in it.
"""

from .fingerprint import loo_predictions, neighbour_cache

PAIR_THRESHOLD = 0.05   # off-diagonal pair rate that "stands out"


def estimated_noise_matrix(features, labels, cache=None):
    """rate[a][b]: share of points labelled a whose LOO vote is b."""
    if cache is None:
        cache = neighbour_cache(features)
    preds = loo_predictions(labels, cache[0])
    classes = sorted(set(labels))
    counts = dict((a, 0) for a in classes)
    matrix = dict((a, dict((b, 0) for b in classes)) for a in classes)
    for i, y in enumerate(labels):
        counts[y] += 1
        matrix[y][preds[i]] += 1
    rates = {}
    for a in classes:
        rates[a] = {}
        for b in classes:
            rates[a][b] = (matrix[a][b] / float(counts[a])
                           if counts[a] else 0.0)
    return rates


def matrix_heuristic(features, labels, cache=None):
    """Diagnose by inspecting the estimated noise matrix.

    Returns (predicted_type, max_off_diagonal_pair_rate).
    """
    rates = estimated_noise_matrix(features, labels, cache)
    best = 0.0
    for a in rates:
        for b in rates[a]:
            if a != b and rates[a][b] > best:
                best = rates[a][b]
    if best >= PAIR_THRESHOLD:
        return "class_dependent", best
    return "uniform", best
