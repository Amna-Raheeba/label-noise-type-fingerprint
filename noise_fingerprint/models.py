"""Shared constants and data structures for the noise fingerprinter.

The four planted noise processes (the diagnosis targets):

- ``uniform``           -- every label flips with the same probability
                           to a uniformly random other class.
- ``class_dependent``   -- flip probability and destination depend on
                           the source class only (fixed class pairs,
                           asymmetric rates).
- ``instance_dependent``-- flip probability depends on the instance's
                           features (a random direction in feature
                           space), not on its class or its margin.
- ``boundary``          -- flip probability concentrates near the
                           decision boundary (low clean margin);
                           the destination is the rival class.
"""

NOISE_TYPES = (
    "uniform",
    "class_dependent",
    "instance_dependent",
    "boundary",
)

CLEAN = "clean"

# Rates planted by the headline experiment (brief: 5-30%).
DEFAULT_RATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DEFAULT_SEEDS = (11, 23, 37)

# Reference-bank rate grid (fingerprints are matched at the estimated
# rate, so the bank spans the full plausible range).
BANK_RATES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
BANK_SEEDS = (0, 1, 2)

DEFAULT_N = 420
DEFAULT_D = 6
DEFAULT_CLASSES = 4

# Fingerprint components, in vector order.  Every component is a
# model-free statistic of the labelled data itself (see
# fingerprint.py for the exact definitions).
FINGERPRINT_KEYS = (
    "disagree_rate",
    "low_margin_share",
    "margin_contrast",
    "high_margin_rate_ratio",
    "pair_concentration",
    "pair_asymmetry",
    "dup_conflict_rate",
    "dup_ratio",
    "region_residual",
    "shell_decay",
)

# The remedy each diagnosed type implies.  This mapping is the point
# of the tool: the four noise types need four different clean-ups.
REMEDIES = {
    "uniform": (
        "Prune or down-weight the flagged points uniformly: the noise "
        "is unbiased, so confident-learning style pruning and robust "
        "losses are safe; no class pair or region needs special care."
    ),
    "class_dependent": (
        "Fix the dominant class pairs first: estimate the class-pair "
        "noise matrix, relabel or reweight those pairs, and only then "
        "prune. Uniform pruning would waste clean points from the "
        "quiet classes."
    ),
    "instance_dependent": (
        "Target the hot feature regions: relabel or collect more data "
        "where the local disagreement concentrates. Global pruning "
        "removes good points elsewhere and leaves the bad region bad."
    ),
    "boundary": (
        "Do not prune: boundary points carry the decision signal. "
        "Re-annotate the low-margin cases, add features or training "
        "data near the boundary, and expect some disagreement to be "
        "genuine class overlap rather than error."
    ),
    CLEAN: (
        "No planted-noise signature detected: keep the labels, and "
        "spend the cleaning budget elsewhere."
    ),
}


class Dataset(object):
    """One seeded labelled dataset with planted-noise ground truth."""

    def __init__(self, features, labels, clean_labels, flipped,
                 noise_type, planted_rate, dup_pairs, seed, aux=None):
        self.features = features          # list of list of float
        self.labels = labels              # observed (noisy) labels
        self.clean_labels = clean_labels  # ground-truth labels
        self.flipped = flipped            # list of bool, per point
        self.noise_type = noise_type      # planted process or CLEAN
        self.planted_rate = planted_rate  # requested flip rate
        self.dup_pairs = dup_pairs        # list of (i, j) near-duplicate pairs
        self.seed = seed
        self.aux = aux or {}

    @property
    def n(self):
        return len(self.labels)

    @property
    def actual_rate(self):
        if not self.flipped:
            return 0.0
        return sum(1 for f in self.flipped if f) / float(len(self.flipped))


class Diagnosis(object):
    """The fingerprinter's verdict on one labelled dataset."""

    def __init__(self, predicted_type, confidence, estimated_rate,
                 fingerprint, distances, remedy):
        self.predicted_type = predicted_type
        self.confidence = confidence
        self.estimated_rate = estimated_rate
        self.fingerprint = fingerprint    # dict key -> float
        self.distances = distances        # noise type -> bank distance
        self.remedy = remedy
