# Adaptive non-local observable q4 results

**Verdict: STOP. Neither locality improves the retained quantum predictor or
the full system.**

## Validity checks

Both independent Kaggle version-1 jobs completed successfully. Each scored all
17,348 PRIMARY development ECGs from 14,958 patients across official folds
1–8. Folds 9 and 10 remained sealed. The reconstructed PLS-q4 coordinates
matched the retained seed-42 artifact with maximum absolute error
`5.96e-08`, well inside the frozen `1e-5` tolerance.

The k=2 adaptive and parameter-matched MLP arms each contained 139 trainable
parameters. The k=3 arms contained 297 and 295 trainable parameters,
respectively. All eight folds showed finite gradients and declining training
losses, so the result is not explained by a crashed or frozen optimizer.

## Primary comparison

| Predictor | AUPRC | AUROC | Brier | Sensitivity at 90% specificity |
|---|---:|---:|---:|---:|
| Retained VQC | **0.829379** | 0.921799 | **0.092544** | **0.767170** |
| ANO k=2 | 0.828900 | 0.921486 | 0.092683 | 0.766712 |
| k=2 fixed-Pauli | 0.828584 | 0.920981 | 0.092704 | 0.764652 |
| k=2 matched MLP | **0.829814** | **0.922592** | 0.092626 | 0.764423 |
| ANO k=3 | 0.828428 | 0.921496 | 0.092724 | 0.765339 |
| k=3 fixed-Pauli | 0.828550 | 0.921329 | 0.092567 | 0.766026 |
| k=3 matched MLP | **0.829908** | **0.922653** | 0.092563 | **0.767857** |

Paired 2,000-replicate patient-cluster bootstrap results:

- k=2 ANO minus retained VQC: `-0.00048`, 95% CI
  `[-0.00201, +0.00078]`.
- k=3 ANO minus retained VQC: `-0.00095`, 95% CI
  `[-0.00244, +0.00043]`.
- k=2 ANO minus its matched MLP: `-0.00092`, 95% CI
  `[-0.00185, -0.00011]`.
- k=3 ANO minus its matched MLP: `-0.00148`, 95% CI
  `[-0.00249, -0.00051]`.

The adaptive k=2 readout is only `+0.00029` AUPRC above its fixed-Pauli
ablation, with CI `[-0.00201, +0.00256]`. The k=3 adaptive readout is
`-0.00019` below fixed Pauli, with CI `[-0.00244, +0.00248]`. There is no
validated adaptive-measurement effect.

## Full-system comparison

| Fusion system | AUPRC | AUROC | Brier |
|---|---:|---:|---:|
| Retained VQC fusion | 0.835456 | 0.926258 | 0.089717 |
| ANO k=2 fusion | 0.835246 | 0.925998 | 0.089832 |
| ANO k=3 fusion | 0.835094 | 0.925953 | 0.089867 |
| k=2 matched-MLP fusion | 0.835942 | 0.926932 | 0.089751 |
| k=3 matched-MLP fusion | **0.836072** | **0.926978** | **0.089707** |
| Frozen all-classical ceiling | **0.838015** | — | — |

Both quantum fusions are below the retained quantum fusion and remain roughly
0.0028–0.0029 AUPRC below the frozen classical ceiling. Their small sensitivity
increases at 90% specificity do not compensate for lower AUPRC, AUROC and Brier
performance and were not prespecified promotion criteria.

## Interpretation

The experiment isolates measurement capacity while preserving the input,
circuit and training contract. Adaptive k-local measurements trained normally,
but additional measurement expressivity did not extract better generalizable
MI information from PLS-q4. The parameter-matched MLP significantly beats both
ANO variants, showing that the additional parameters are more useful in an
ordinary nonlinear q4 head than in this adaptive quantum measurement.

ANO does beat the older q4 logistic and retained q4 MLP-ensemble AUPRC controls,
but it does not beat the stronger retained VQC. Those legacy wins therefore do
not constitute an improvement to the current system.

All five frozen promotion gates failed for both localities. Do not launch
multi-seed confirmation, combine k=2 and k=3 post hoc, or replace the retained
VQC with ANO.

## Artifacts

- Kaggle k=2: <https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-ano-k2-q4>
- Kaggle k=3: <https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-ano-k3-q4>
- Local ignored downloads: `artifacts/ano_k2_q4/` and
  `artifacts/ano_k3_q4/`
