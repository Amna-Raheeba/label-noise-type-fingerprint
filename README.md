# Label Noise Type Fingerprint

## 1. What it is

Every label-cleaning method assumes a noise process. Confident
learning assumes flips depend only on the true class. Pruning assumes
flips are roughly unbiased. Boundary-aware relabelling assumes the
errors sit near the decision surface. Pick the wrong assumption and
the "cleaning" damages the dataset: pruning a boundary-noisy dataset
deletes exactly the points that carry the decision signal.

This project diagnoses **which kind** of label noise a dataset has
before any cleaning happens:

- **uniform** -- random flips, independent of everything;
- **class_dependent** -- flips concentrated in specific class pairs
  (the classic noisy-channel model);
- **instance_dependent** -- flip probability varies with the point's
  features (some regions of the data are labelled worse than others);
- **boundary** -- flips concentrated where classes genuinely meet.

The tool reads only the labelled data itself -- no model training,
no clean validation set -- and returns the diagnosed type, an
estimated noise rate, a confidence, and the remedy that matches the
type. It is validated against planted ground truth: datasets where
each noise process was applied at a known rate.

## 2. The novel mechanism

Three pieces, none of which is a trained classifier:

1. **A model-free fingerprint.** Ten statistics of the labelled
   data: leave-one-out neighbour disagreement, disagreement by
   neighbourhood-margin shell (how outvoted labels distribute
   between ambiguous and decisive neighbourhoods), class-pair
   concentration and asymmetry of the disagreements, the label
   conflict rate among near-duplicate points, the excess of
   duplicates over a permutation null, how much disagreement a
   clustering leaves unexplained, and the disagreement decay across
   distance shells. Each noise process distorts a different subset
   of these statistics.
2. **A parametric-bootstrap reference bank.** The tool cannot see
   clean labels, so it reconstructs an approximate clean world from
   the noisy data itself: iterated neighbour smoothing gives proxy
   labels, a Gaussian class model is fitted to them, and plausible
   clean label sets are *sampled* from the fitted posterior at the
   dataset's own points. Each of the four noise processes is then
   planted on those samples across a rate grid, producing reference
   fingerprints **on the same feature distribution, with the same
   class overlap, as the target**. Matching against references built
   this way cancels the dataset's own geometry -- the part that
   makes fingerprints from different datasets incomparable.
3. **Joint (type, rate) matching with a named remedy.** The target
   fingerprint is compared to every bank cell; each type scores by
   its closest cell, the winner's rate is refined by inverting its
   disagreement curve, and confidence is the relative margin
   between the best and second-best type. The output is a diagnosis
   plus the cleaning strategy that type calls for (prune,
   reweight class pairs, relabel the hot regions, or re-annotate
   the boundary instead of pruning it).

## 3. Prior art and the differentiator

- **Cleanlab / confident learning** (Northcutt et al.) finds and
  removes label errors and estimates a class-conditional noise
  matrix. It *assumes* the noise family (class-conditional) instead
  of diagnosing it, and says nothing about instance-dependent or
  boundary noise. Used naively -- "inspect the noise matrix, and if
  a pair stands out, call it class-dependent" -- it degenerates: in
  the evaluation below, that heuristic answered "class-dependent"
  for **all 72 datasets**, scoring 1.000 on class-dependent noise
  and 0.000 on everything else.
- **Noise taxonomies** (e.g. Frenay & Verleysen's survey) define the
  noise types carefully but ship no offline diagnostic validated
  against planted ground truth.
- **Noise-rate estimators** estimate *how much* noise there is,
  assuming the type is already known.

The differentiator: this tool answers the prior question -- *which
process generated the noise* -- from observable signatures alone,
with the type diagnosis and the rate estimate validated against
planted ground truth, and each diagnosis mapped to the remedy that
is safe for that type.

## 4. Measured headline results

72 seeded datasets (4 noise types x rates 5-30% x 3 seeds), 470
points each (420 generated + 50 planted near-duplicates), 4-class
Gaussian mixtures with genuine class overlap. Full per-dataset
records in `results/evaluation.json`; summary in
`results/evaluation_summary.md`.

| measure | fingerprinter | matrix-inspection baseline |
|---|---|---|
| noise-type accuracy (overall) | **0.736** | 0.250 |
| uniform | 0.722 | 0.000 |
| class_dependent | 0.778 | 1.000 |
| instance_dependent | 0.667 | 0.000 |
| boundary | 0.778 | 0.000 |
| noise-rate estimation MAE | **0.046** | n/a |

Accuracy by planted rate (baseline is 0.250 at every rate):

| rate | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
|---|---|---|---|---|---|---|
| accuracy | 0.417 | 0.583 | 0.750 | 1.000 | 0.833 | 0.833 |

Reading the results honestly:

- The baseline's perfect class-dependent score is the degeneracy,
  not a strength: it can only ever answer two of the four types,
  and on data with genuine overlap some class pair always stands
  out, so it answers "class-dependent" every time.
- The fingerprinter's errors concentrate exactly where they should:
  at 5-10% planted noise, where the noise is smaller than the
  dataset's own ~12% genuine-overlap disagreement. At 20% noise the
  diagnosis is perfect (12/12).
- The residual confusion is structured: uniform is sometimes read
  as class-dependent (4/18) when random flips happen to cluster on
  a pair, and instance-dependent noise is sometimes read as
  boundary noise (4/18) -- the planted feature direction partly
  aligns with the class boundary, so the two processes genuinely
  resemble each other.
- Rate estimation is tightest for uniform and class-dependent
  noise (MAE 0.036-0.037) and loosest for boundary noise (0.061).

## 5. How to run

```bash
pip install -r requirements.txt

# Diagnose the built-in demo dataset (boundary noise at 20%)
python -m noise_fingerprint demo

# Re-run the full evaluation and rewrite results/
python -m noise_fingerprint evaluate

# Interactive dashboard
streamlit run app.py

# Regenerate the seeded case files under data/
python scripts/build_cases.py

# Verify the shipped numbers reproduce (headline-drift guard)
python scripts/reproduce_results.py

# Tests
pytest
```

## 6. Honest limits

- All validation is on **synthetic** seeded data with four planted
  noise processes on Gaussian-mixture features. Real annotation
  noise mixes processes (an annotator can be both class-confused
  and boundary-sloppy), and mixed noise is out of scope for this
  version -- the tool reports the single closest process.
- At low noise rates (5-10%) the diagnosis is unreliable on data
  with substantial genuine class overlap: the noise signal is
  smaller than the overlap signal, and no statistic of the labelled
  data alone can fully separate them. The confidence score reflects
  this -- low-rate verdicts come out with low confidence.
- The reference bank assumes the class-conditional feature
  distribution is roughly Gaussian (the fitted model is per-class
  centroids with a shared variance). Strongly non-Gaussian geometry
  (manifolds, heavy tails) would bias the bank; the disagreement
  statistics themselves are model-free, but the matching references
  are not.
- Instance-dependent noise is modelled with a single latent feature
  direction. Real instance dependence (annotator fatigue, image
  quality, dialect) need not be a single direction, and is the
  hardest type here (0.667) partly for that reason.
- Near-duplicates are planted copies with small jitter. Real
  duplicate structure (exact duplicates, grouped records) differs;
  the duplicate-conflict statistic is used as one component among
  ten, not as a standalone detector.
- Diagnosis costs a few seconds per dataset (the bank is rebuilt
  per target so the references match its geometry). This is a
  dataset-audit tool, not a per-batch monitor.

## 7. Repository layout

```
noise_fingerprint/
  models.py       noise types, fingerprint keys, remedies, records
  generate.py     seeded generator + the four planted noise processes
  fingerprint.py  model-free fingerprint statistics
  diagnose.py     proxy labels, class model, reference bank, matching
  baselines.py    cleanlab-style noise-matrix inspection heuristic
  evaluate.py     planted-ground-truth evaluation harness
  report.py       text reports and the headline summary
  cli.py          demo / evaluate commands
data/             seeded case files (scenarios + demo case)
examples/         one end-to-end audit + a captured sample report
notebooks/        runnable demo and evaluation walkthroughs (.py)
scripts/          case builder + headline-drift reproduction guard
results/          evaluation.json/csv, summary, HEADLINE.md
tests/            pytest suite
app.py            Streamlit dashboard (Diagnose / Signatures / Evaluation)
```

## 8. Paper angle

"Diagnose before you clean: fingerprinting the label-noise process
in tabular datasets." The contribution is the diagnostic framing --
noise-type identification as a precondition for safe label
cleaning -- plus the parametric-bootstrap bank that makes
fingerprints comparable across datasets, validated against planted
ground truth with an honest low-rate failure analysis. Natural
homes: data-centric AI workshops and venues (NeurIPS DCAI, ICML
DataPerf-style tracks) and data-quality journals.

## 9. Author and licence

Author: Amna Raheeba. MIT licence (see `LICENSE`).
