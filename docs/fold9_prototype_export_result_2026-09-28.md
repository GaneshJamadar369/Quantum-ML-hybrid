# Frozen hybrid prototype export and Fold-9 calibration result

**Date:** 2026-09-28  
**Kaggle job:** `swayamjeetbhagat4/aquire-med-frozen-hybrid-prototype-export`, version 1  
**Artifact model version:** `aquire-hybrid-q4-v1`  
**Artifact source commit:** `78cc754ba8af063863a79a52ba5e2d7889b56731`

## Outcome

The export completed and produced a verified 18-artifact inference bundle.
Every accepted ECG executes both frozen predictors before nonnegative logistic
fusion:

```text
ECG -> Transformer h128 -> PLS-q4 -> three-restart 4-qubit VQC ensemble -> sQ
  \-> 106 deployable morphology features -> calibrated HGB --------------> sC

P(MI pattern) = Platt(sigmoid(b0 + bQ*sQ + bC*sC))
```

There is no model-selection gate, residual target, dynamic routing or
single-route fallback.

## Data separation

| Role | Folds | Primary ECGs | Use |
|---|---:|---:|---|
| Development | 1--8 | 17,348 | Fit all predictive components and OOF fusion |
| Calibration | 9 | 2,174 | Fit final Platt calibration and 90% specificity threshold |
| Locked test | 10 | Not accessed | Reserved for one-time final evaluation |

The bundle report and runner both record `fold_10_accessed: false`.

## Fold-9 result

| Metric | Value |
|---|---:|
| AUPRC | 0.804604 |
| AUROC | 0.917037 |
| Brier score | 0.093850 |
| Frozen threshold | 0.435313 |
| Sensitivity at threshold | 73.929% |
| Specificity at threshold | 90.043% |

AUPRC and AUROC measure the ranking of the already-frozen upstream hybrid on
Fold 9; monotonic Platt calibration does not change that ranking. The reported
Brier score and operating point are calibration-set estimates because Fold-9
labels fit the Platt map and threshold. They are not a final generalization
estimate. Fold 10 remains required for that purpose.

## Frozen fusion

The final nonnegative fusion was selected from development OOF predictions.
Its standardized weights are:

| Parameter | Value |
|---|---:|
| Classical HGB weight | 0.683729 |
| Quantum VQC weight | 2.219862 |
| Intercept | -2.421202 |
| Selected regularization C | 1.0 |

Both branch weights are positive. This proves that the exported fusion uses
both scores; it does not establish quantum advantage. The strongest matched
all-classical development control remains slightly stronger in AUPRC.

## Integrity and serving parity

- All 18 registered artifacts passed path, size and SHA-256 verification.
- The three VQC checkpoints contain the retained `narrow_js`, 60-epoch,
  2,000-patient-per-class protocol with seeds 42, 31415 and 27182.
- The API startup golden test ran all eight signed Fold-9 ECG fixtures through
  both routes and fusion.
- Maximum Kaggle-GPU-to-local-CPU probability error was `0.00013327`, below
  the frozen `0.0002` tolerance.
- One real upload call returned HTTP 200 with both routes marked active and a
  single fused MI-pattern result.
- Serving is pinned to scikit-learn 1.6.1 and NeuroKit2 0.2.13, matching the
  export environment.

## Decision

**GO for the SIH prototype.** The system now has a real executable hybrid
bundle and honest calibration boundary. Do not open Fold 10 during UI tuning
or demo preparation. The remaining release work is latency measurement,
explanations, three rehearsed demo cases and container/browser validation.
