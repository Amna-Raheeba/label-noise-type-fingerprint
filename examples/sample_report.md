Label-noise fingerprint audit
=============================

Dataset: 470 points, seed 7
Planted ground truth: boundary at rate 0.20 (actual flipped 0.183)

Fingerprint:
  disagree_rate          0.2128
  low_margin_share       0.6300
  margin_contrast        1.5709
  high_margin_rate_ratio 0.3272
  pair_concentration     0.1800
  pair_asymmetry         0.0909
  dup_conflict_rate      0.3261
  dup_ratio              1.5326
  region_residual        0.0303
  shell_decay            0.0213

Bank distances (lower = closer match):
  uniform                0.9152
  class_dependent        0.8669
  instance_dependent     0.8442
  boundary               0.6414

Diagnosis: boundary (confidence 0.240)
Estimated noise rate: 0.188
Remedy: Do not prune: boundary points carry the decision signal. Re-annotate the low-margin cases, add features or training data near the boundary, and expect some disagreement to be genuine class overlap rather than error.
