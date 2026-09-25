# G6Q-OPT1 result — nested q4 optimization and five-seed confirmation

## Verdict

**STOP for the current raw-score circuit. All three promotion gates failed.**

Five prespecified seeds completed on folds 1–8 for 17,348 ECGs from 14,958
patients. Folds 9 and 10 remained sealed. The experiment used fold-local
PLS-q4, nested inner-fold hyperparameter/epoch selection, identical-q4
classical controls, a no-entanglement ablation and 5,000-replicate paired
patient-cluster bootstraps.

## Five-seed ensemble

| Model | AUPRC | AUROC | Brier |
|---|---:|---:|---:|
| VQC | 0.82750 | 0.92165 | **0.10472** |
| VQC, no entanglement | 0.82691 | 0.92130 | 0.10515 |
| q4 logistic | 0.82673 | **0.92202** | 0.11077 |
| q4 MLP | 0.82710 | 0.92196 | 0.10984 |
| q4 RBF | 0.78817 | 0.91055 | 0.11292 |
| clinical + VQC fusion | 0.83607 | 0.92715 | 0.08945 |
| clinical + q4 MLP fusion | **0.83766** | **0.92765** | **0.08898** |

The VQC ensemble is numerically `+0.00042` AUPRC above the q4 MLP ensemble,
but the 95% patient-bootstrap interval is `[-0.00123, +0.00203]`. This is not a
validated improvement and is far below the prespecified `+0.005` effect-size
gate.

The quantum fusion is `-0.00158` AUPRC below the matched classical fusion, with
95% interval `[-0.00277, -0.00042]`. It also remains below the previously frozen
all-classical ceiling of `0.83802`.

## Seed stability

| Seed | VQC | strongest identical-q4 control | Delta | quantum fusion | classical fusion |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.81440 | 0.82586 | -0.01146 | 0.82904 | 0.83664 |
| 14142 | 0.82022 | 0.82722 | -0.00700 | 0.83195 | 0.83669 |
| 16180 | 0.82602 | 0.82660 | -0.00058 | 0.83529 | 0.83727 |
| 27182 | 0.82329 | 0.82647 | -0.00318 | 0.83407 | 0.83598 |
| 31415 | 0.82733 | 0.82654 | +0.00078 | 0.83503 | 0.83567 |

The VQC won only one of five seeds. Seed averaging improves the VQC because its
errors are partly initialization-dependent, but that does not satisfy the
required per-seed stability.

## Entanglement and clinical operating point

VQC minus no-entanglement AUPRC was `+0.00059`, with 95% interval
`[-0.00061, +0.00183]`. The entanglement gate therefore failed.

At approximately 90% specificity:

- VQC sensitivity: `0.76671`; hard-negative FPR: `0.15682`.
- q4 MLP sensitivity: `0.76694`; hard-negative FPR: `0.15627`.
- quantum-fusion sensitivity: `0.78068`; hard-negative FPR: `0.17076`.
- classical-fusion sensitivity: `0.77953`; hard-negative FPR: `0.16893`.

The quantum fusion gains about `0.00115` sensitivity at this descriptive
threshold, but loses AUPRC/AUROC/Brier and raises hard-negative FPR. That is not
enough to claim a better clinical model.

## Bottleneck trace

- The inner search selected the narrow-bandwidth JS circuit in 36/40 outer
  fits; baseline and ranking-only each won 2/40.
- The chosen epoch hit the 30-epoch cap in 23/40 fits, motivating a longer
  scheduled-training diagnostic.
- VQC and q4 classical scores remain highly correlated (`0.95–0.97` Spearman),
  so the quantum head mostly reproduces the same ordering.
- q4 effective rank is stable at about `3.90`; quantum-observable effective
  rank varies from `2.03` to `4.09`, showing seed-dependent observable geometry.
- Macro within-fold VQC-minus-logistic AUPRC is approximately `-0.00041`, while
  pooled single-seed deficits are larger. Fold-to-fold score-scale drift is a
  measurable secondary bottleneck.
- Diagnostic normalization using each held-out distribution can recover some
  pooled loss, but it is transductive and prohibited in the product pipeline.

The final targeted follow-up therefore uses only training-score reference
distributions, longer scheduled training and fixed restart averaging. It does
not reopen wider-qubit, deeper-circuit or broad representation searches.

## Claim boundary

This experiment evaluates predictive accuracy on a classical state-vector
simulator. Even a statistically positive result would establish predictive
value for this benchmark, not computational quantum advantage.
