# G6Q-NR — NeurIPS idea-transfer scout result

**Run date:** 2026-09-27  
**Status:** completed on two parallel Kaggle T4 jobs  
**Decision:** STOP both scouts; retain the existing aligned q4 VQC

## Data and execution validity

- 17,348 ECGs and 14,958 patients from official folds 1–8.
- Eight complete outer folds; zero missing outputs or duplicate ECG IDs.
- Fold 9 and Fold 10 remained sealed.
- Both scouts reproduced the retained fold-local PLS-q4 coordinates with
  maximum absolute error `5.96e-08`.
- Each outer fold used 4,000 patient-unique training records.
- All training losses decreased and all recorded gradients were finite.

## Primary comparison

| Model | AUPRC | Delta versus retained VQC | Patient-bootstrap 95% CI |
|---|---:|---:|---:|
| Retained aligned q4 VQC | 0.829379 | — | — |
| Layerwise-frequency q4 | 0.827496 | -0.001884 | [-0.002813, -0.000995] |
| Tied-equilibrium q4 | 0.828705 | -0.000674 | [-0.001610, +0.000211] |

Layerwise frequency scaling caused a small but statistically supported decline.
The tied-equilibrium score was statistically indistinguishable from the
retained VQC, but it did not improve it and remained far below the frozen
`+0.005` promotion effect.

The equilibrium arm still beat the weaker identical-q4 controls:

- versus q4 logistic: `+0.00322` AUPRC, 95% CI `[+0.00111,+0.00553]`;
- versus q4 MLP: `+0.00386` AUPRC, 95% CI `[+0.00181,+0.00605]`.

This does not justify promotion because the retained VQC already provides the
stronger quantum reference.

## System comparison

| System | AUPRC |
|---|---:|
| Frozen all-classical ceiling | 0.838015 |
| Retained quantum fusion | 0.835456 |
| Tied-equilibrium fusion | 0.835219 |
| Layerwise-frequency fusion | 0.834731 |

Neither scout beat the retained fusion or the classical ceiling. The frequency
fusion decline versus retained quantum fusion was supported by the paired
interval `[-0.001320,-0.000182]`. The equilibrium-fusion interval crossed
zero `[-0.000785,+0.000294]`.

## Secondary results

- Frequency VQC: AUROC `0.921203`, Brier `0.092724`, sensitivity at 90%
  specificity `0.76809`, hard-negative FPR `0.15701`.
- Equilibrium VQC: AUROC `0.921300`, Brier `0.092554`, sensitivity at 90%
  specificity `0.76854`, hard-negative FPR `0.15792`.
- Retained VQC: AUROC `0.921799`, Brier `0.092544`, sensitivity at 90%
  specificity `0.76717`, hard-negative FPR `0.15664`.

The small sensitivity changes do not compensate for lower AUPRC, lower AUROC
and slightly higher hard-negative false-positive rates.

## Mechanistic audit

The layerwise-frequency circuit trained 57 parameters. Learned scale values
remained finite and within the frozen bounds, ranging from `0.3471` to
`0.8273`; the failure was not caused by scale saturation.

The tied-equilibrium circuit trained 49 parameters and required roughly twice
the runtime of the frequency arm (`~1,306 s` versus `~629 s`). It did not
approach a fixed point after three iterations:

- per-fold median final residuals ranged from `0.174` to `0.210`;
- no held-out sample had residual below `0.05`;
- the required implicit-QDEQ residual is at most `1e-4`.

Therefore this arm is only a finite tied recurrence and fails the prerequisite
for an implicit deep-equilibrium experiment.

## Decision

1. Retain the supervised Transformer → fold-local PLS-q4 → aligned two-layer
   VQC as the best quantum head.
2. Do not run more seeds, restarts or merged frequency/equilibrium circuits.
3. Do not describe the tied recurrence as QDEQ or as converged equilibrium.
4. Keep anatomy-aware entanglement blocked until a fixed-identity anatomical
   representation passes its classical information gate.
5. Keep Lie/Hadamard gradient experiments deferred until the predictive
   architecture is frozen; those methods target shot efficiency rather than
   predictive score.

Artifacts:

- `kaggle_outputs/neurips_frequency_20260927/`
- `kaggle_outputs/neurips_equilibrium_20260927/`
