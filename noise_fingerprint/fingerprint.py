"""The fingerprint: model-free statistics of a labelled dataset.

Every component is computed from (features, labels) alone -- no model
is trained.  A leave-one-out kNN vote supplies the working notions of
"disagreement" and "margin"; the components then describe *where*
the disagreement lives (margin profile, class pairs, duplicate
pairs, feature regions, neighbour shells), which is what separates
the four noise processes.
"""

import math

from .models import FINGERPRINT_KEYS

K_LOO = 10        # neighbours for the leave-one-out prediction
K_MARGIN = 15     # neighbours for the vote margin
K_MAX = 25        # cached neighbours per point
DUP_EPS = 0.15    # distance under which two points count as duplicates
N_CLUSTERS = 6    # feature-space regions for the residual component


def _dist2(a, b):
    total = 0.0
    for x, y in zip(a, b):
        diff = x - y
        total += diff * diff
    return total


def neighbour_cache(features, k_max=K_MAX):
    """Per point: neighbour indices and distances, nearest first.

    Depends only on the features, so one cache serves the target
    dataset and every reference-bank relabelling of it.
    """
    n = len(features)
    all_idx = []
    all_dist = []
    for i in range(n):
        dists = []
        xi = features[i]
        for j in range(n):
            if j == i:
                continue
            dists.append((_dist2(xi, features[j]), j))
        dists.sort()
        top = dists[:k_max]
        all_idx.append([j for _, j in top])
        all_dist.append([math.sqrt(d2) for d2, _ in top])
    return all_idx, all_dist


def majority_label(ordered_labels):
    """Majority vote; ties broken by the nearest supporter."""
    counts = {}
    first_seen = {}
    for pos, lab in enumerate(ordered_labels):
        counts[lab] = counts.get(lab, 0) + 1
        if lab not in first_seen:
            first_seen[lab] = pos
    best = None
    best_key = None
    for lab, c in counts.items():
        key = (c, -first_seen[lab])
        if best_key is None or key > best_key:
            best_key = key
            best = lab
    return best


def loo_predictions(labels, cache_idx, k=K_LOO):
    """Leave-one-out kNN prediction for every point."""
    out = []
    for i in range(len(labels)):
        neigh = [labels[j] for j in cache_idx[i][:k]]
        out.append(majority_label(neigh))
    return out


def vote_margins(labels, cache_idx, k=K_MARGIN):
    """Neighbourhood decisiveness margin, in [0, 1].

    (votes for the neighbourhood's majority label - votes for the
    runner-up label) / k.  This is a property of the point's
    *neighbourhood*, deliberately independent of the point's own
    label: a point deep inside one class's territory has margin near
    1, a point in the contested zone near 0.  (A margin defined from
    the point's own vote share would be tautological -- a suspect is
    by construction a point its neighbours outvote.)
    """
    out = []
    for i in range(len(labels)):
        counts = {}
        for j in cache_idx[i][:k]:
            counts[labels[j]] = counts.get(labels[j], 0) + 1
        ranked = sorted(counts.values(), reverse=True)
        top = ranked[0]
        second = ranked[1] if len(ranked) > 1 else 0
        out.append((top - second) / float(k))
    return out


def detect_dup_pairs(cache_idx, cache_dist, eps=DUP_EPS):
    """Near-duplicate pairs, found via the nearest neighbour."""
    pairs = []
    for i in range(len(cache_idx)):
        if cache_dist[i] and cache_dist[i][0] < eps:
            j = cache_idx[i][0]
            if i < j:
                pairs.append((i, j))
    return pairs


def _terciles(values):
    """Index sets of the low / mid / high thirds of ``values``."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    n = len(order)
    third = n // 3
    return (set(order[:third]), set(order[third:2 * third]),
            set(order[2 * third:]))


def _margin_deciles(margins):
    """Decile index (0 = lowest margin) per point, by rank."""
    order = sorted(range(len(margins)), key=lambda i: margins[i])
    n = len(order)
    dec = [0] * n
    for rank, idx in enumerate(order):
        dec[idx] = min(9, int(10 * rank / n))
    return dec


def _clusters(features, k=N_CLUSTERS, rounds=15):
    """Deterministic k-means (init: evenly spaced sorted points)."""
    n = len(features)
    d = len(features[0])
    order = sorted(range(n),
                   key=lambda i: features[i][0] + features[i][1])
    centres = []
    for c in range(k):
        src = features[order[int((c + 0.5) * n / k)]]
        centres.append(list(src))
    assign = [0] * n
    for _ in range(rounds):
        changed = False
        for i in range(n):
            best = 0
            best_d = None
            for c in range(k):
                dc = _dist2(features[i], centres[c])
                if best_d is None or dc < best_d:
                    best_d = dc
                    best = c
            if assign[i] != best:
                assign[i] = best
                changed = True
        sums = [[0.0] * d for _ in range(k)]
        counts = [0] * k
        for i in range(n):
            counts[assign[i]] += 1
            for j in range(d):
                sums[assign[i]][j] += features[i][j]
        for c in range(k):
            if counts[c]:
                centres[c] = [s / counts[c] for s in sums[c]]
        if not changed:
            break
    return assign


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def compute_fingerprint(features, labels, cache=None, dup_pairs=None,
                        clusters=None):
    """The fingerprint vector of one labelled dataset, as a dict.

    ``clusters`` (a per-point region assignment) depends only on the
    features; callers fingerprinting many relabellings of one feature
    set pass it in to avoid recomputing the k-means.
    """
    n = len(labels)
    if cache is None:
        cache = neighbour_cache(features)
    cache_idx, cache_dist = cache
    preds = loo_predictions(labels, cache_idx)
    suspect = [preds[i] != labels[i] for i in range(n)]
    disagree = sum(suspect) / float(n)
    margins = vote_margins(labels, cache_idx)
    low_set, _mid_set, high_set = _terciles(margins)

    n_suspect = sum(suspect)
    if n_suspect:
        low_share = sum(1 for i in low_set if suspect[i]) / float(n_suspect)
    else:
        low_share = 0.0
    rate_low = (sum(1 for i in low_set if suspect[i])
                / float(len(low_set)))
    rate_high = (sum(1 for i in high_set if suspect[i])
                 / float(len(high_set)))
    margin_contrast = _clip((rate_low - rate_high) / (disagree + 1e-9),
                            -3.0, 3.0)
    high_ratio = _clip(rate_high / (disagree + 1e-9), 0.0, 3.0)

    # Class-pair confusion among suspects: observed label -> LOO vote.
    pair_counts = {}
    for i in range(n):
        if suspect[i]:
            key = (labels[i], preds[i])
            pair_counts[key] = pair_counts.get(key, 0) + 1
    if n_suspect and pair_counts:
        dom_pair, dom_count = max(pair_counts.items(),
                                  key=lambda kv: kv[1])
        pair_concentration = dom_count / float(n_suspect)
        back = pair_counts.get((dom_pair[1], dom_pair[0]), 0)
        pair_asymmetry = abs(dom_count - back) / float(dom_count + back)
    else:
        pair_concentration = 0.0
        pair_asymmetry = 0.0

    # Duplicate-pair label conflict.
    if dup_pairs is None:
        dup_pairs = detect_dup_pairs(cache_idx, cache_dist)
    if dup_pairs:
        conflicts = sum(1 for (i, j) in dup_pairs
                        if labels[i] != labels[j])
        dup_conflict = conflicts / float(len(dup_pairs))
    else:
        dup_conflict = 0.0
    dup_ratio = _clip(dup_conflict / (disagree + 1e-9), 0.0, 3.0)

    # Region residual: does local disagreement exceed what the
    # margin profile predicts?  (The instance-dependent signature.)
    dec = _margin_deciles(margins)
    dec_rate = []
    for b in range(10):
        members = [i for i in range(n) if dec[i] == b]
        if len(members) >= 5:
            dec_rate.append(
                sum(1 for i in members if suspect[i]) / float(len(members)))
        else:
            dec_rate.append(disagree)
    assign = clusters if clusters is not None else _clusters(features)
    residuals = []
    for c in range(N_CLUSTERS):
        members = [i for i in range(n) if assign[i] == c]
        if len(members) < 15:
            continue
        observed = sum(1 for i in members if suspect[i]) / float(len(members))
        predicted = sum(dec_rate[dec[i]] for i in members) / float(len(members))
        residuals.append(abs(observed - predicted))
    region_residual = (sum(residuals) / float(len(residuals))
                       if residuals else 0.0)

    # Neighbour-shell disagreement: nearest shell vs far shell.
    near_dis = 0
    far_dis = 0
    for i in range(n):
        near = [labels[j] for j in cache_idx[i][:3]]
        far = [labels[j] for j in cache_idx[i][8:15]]
        if majority_label(near) != labels[i]:
            near_dis += 1
        if majority_label(far) != labels[i]:
            far_dis += 1
    shell_decay = (near_dis - far_dis) / float(n)

    values = {
        "disagree_rate": disagree,
        "low_margin_share": low_share,
        "margin_contrast": margin_contrast,
        "high_margin_rate_ratio": high_ratio,
        "pair_concentration": pair_concentration,
        "pair_asymmetry": pair_asymmetry,
        "dup_conflict_rate": dup_conflict,
        "dup_ratio": dup_ratio,
        "region_residual": region_residual,
        "shell_decay": shell_decay,
    }
    return dict((key, values[key]) for key in FINGERPRINT_KEYS)


def cluster_assignment(features):
    """The deterministic feature-space regions (see _clusters)."""
    return _clusters(features)


def fingerprint_vector(fingerprint):
    """The fingerprint dict as an ordered list of floats."""
    return [fingerprint[key] for key in FINGERPRINT_KEYS]


def margin_profile(features, labels, cache=None):
    """Suspect rate by margin decile (for reports and the dashboard).

    Returns a list of (mean_margin, suspect_rate, count) per decile,
    lowest margin first.
    """
    n = len(labels)
    if cache is None:
        cache = neighbour_cache(features)
    cache_idx, _ = cache
    preds = loo_predictions(labels, cache_idx)
    margins = vote_margins(labels, cache_idx)
    dec = _margin_deciles(margins)
    out = []
    for b in range(10):
        members = [i for i in range(n) if dec[i] == b]
        if not members:
            out.append((0.0, 0.0, 0))
            continue
        mean_margin = sum(margins[i] for i in members) / float(len(members))
        rate = (sum(1 for i in members if preds[i] != labels[i])
                / float(len(members)))
        out.append((mean_margin, rate, len(members)))
    return out
