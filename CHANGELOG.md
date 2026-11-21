# Changelog

## 0.1.0 -- 2026-11-22

- First release: label-noise type fingerprinting for tabular
  datasets.
- Model-free 10-component fingerprint: neighbour disagreement,
  margin-shell profile, class-pair concentration/asymmetry,
  near-duplicate conflict rate and excess, cluster-residual
  disagreement, distance-shell decay.
- Parametric-bootstrap reference bank: proxy clean labels by
  iterated neighbour smoothing, Gaussian class model, posterior
  label sampling, per-type noise planting on the target's own
  geometry; joint (type, rate) nearest-fingerprint matching with
  confidence margin and per-type remedies.
- Seeded generator with four planted noise processes (uniform,
  class-dependent, instance-dependent, boundary) and planted
  near-duplicates.
- Cleanlab-style noise-matrix inspection baseline.
- Evaluation on 72 seeded datasets: type accuracy 0.736 vs 0.250
  baseline; rate MAE 0.046; accuracy by rate reported honestly
  (0.417 at 5% noise to 1.000 at 20%).
- Streamlit dashboard (Diagnose / Signatures / Evaluation),
  CLI (demo / evaluate), reproduction guard for the headline
  numbers, 60 tests.
