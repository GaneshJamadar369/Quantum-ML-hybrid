# Phase 6Q-A matched-kernel result

> **Superseded by Phase 6Q-B.** Phase 6Q-B discovered that the original RBF
> control did not receive the QSVM's additional train-fitted angle-coordinate
> transformation. With identical coordinates, RBF, Laplacian and
> product-cosine classical kernels all outperformed IQP-QSVM. See the
> [Phase 6Q-B result](phase6q_b_kernel_controls_result.md).

**Run date:** 2026-09-21

**Execution:** Kaggle CPU, eight patient-safe development folds 1–8

**Runtime:** approximately 4.8 minutes for environment checks and the complete job; model benchmark 190.2 seconds

**Input:** fold-local median imputation, robust scaling and PCA-whitened `z8`

**Training budget:** the same balanced 1,000-record training subset per fold for both models

**Calibration:** inner five-fold, training-only Platt calibration

## Result

| Metric | IQP-QSVM | Matched RBF-SVC | Direction |
|---|---:|---:|---|
| AUPRC | 0.4357 | 0.3426 | IQP higher |
| AUROC | 0.7040 | 0.6127 | IQP higher |
| Sensitivity at 90% specificity | 0.2935 | 0.2072 | IQP higher |
| F1 at threshold 0.5 | 0.4883 | 0.3730 | IQP higher |
| Brier score | 0.2206 | 0.2403 | IQP lower/better |
| Hard-negative false-positive rate | 0.4688 | 0.3481 | IQP worse |
| MI-versus-hard-negative AUPRC | 0.5659 | 0.4805 | IQP higher |
| Inference time per record | 6.216 ms | 0.184 ms | IQP about 34 times slower |

The paired patient-cluster bootstrap used 2,000 valid resamples:

- Delta AUPRC: **+0.0931**, 95% CI **[+0.0779, +0.1083]**.
- Delta AUROC: **+0.0912**, 95% CI **[+0.0801, +0.1023]**.
- Delta Brier score: **-0.0197**, 95% CI **[-0.0220, -0.0173]**.

The correct interpretation is **PASS_MATCHED_KERNEL_ACCURACY_DELTA**. The IQP
kernel predicted better than this particular matched RBF-SVC control on `z8`.
This is not evidence of computational quantum advantage: both kernels ran on
classical hardware, only one classical kernel family was tested, and the IQP
path was substantially slower.

## Numerical and clinical limitations

The IQP Gram matrices were positive semidefinite within numerical tolerance,
but their condition numbers ranged from approximately **8.9e7 to 3.7e10**.
This indicates near-linear dependence and possible sensitivity to regularization
or small data changes. The IQP model also increased the hard-negative false-
positive rate. It therefore cannot be promoted as the deployed champion.

Both `z8` models are far below the provisional full-waveform ResNet AUPRC of
0.7913 and the development classical feature champion AUPRC of 0.7198. Those
figures are not yet a final ranking because the ResNet requires repeated-seed
validation and the fusion result must be rerun after its architecture repair.

## Decision and next run

**MODIFY; do not launch VQC/HQNN yet.** Run polynomial, Laplacian,
product-cosine and scalable kernel approximations using the identical `z8`,
folds, sample budget and calibration. Add kernel regularization/conditioning
sensitivity and patient-bootstrap subgroup deltas. Only then decide whether the
IQP mapping contains evidence beyond ordinary classical kernel engineering.

Use a **CPU** for these kernel comparisons. Use a **GPU** for the repeated-seed
1D ResNet/pretrained ECG encoder and corrected fusion experiments. For a later
HQNN, train the ECG encoder on GPU, freeze it, and run the small quantum head on
the analytic CPU simulator before any hardware experiment.
