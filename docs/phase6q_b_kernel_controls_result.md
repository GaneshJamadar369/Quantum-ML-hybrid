# Phase 6Q-B stronger matched-kernel controls

**Run date:** 2026-09-21

**Execution:** Kaggle CPU, development folds 1–8 only

**Comparison:** identical records, balanced 1,000-record training budget per
fold, fold-local `z8`, train-fitted `StandardScaler → tanh → π` coordinates and
inner five-fold Platt calibration

## Why this rerun was required

Phase 6Q-A compared IQP-QSVM against an RBF-SVC on the same PCA `z8`, but the
QSVM then applied an additional train-fitted standardization and nonlinear angle
map internally. The RBF model did not receive that transformation. Phase 6Q-B
gives every control the exact coordinate map received by the IQP circuit.

## Results

| Model | AUPRC | AUROC | Brier | Hard-negative FPR | ms/record |
|---|---:|---:|---:|---:|---:|
| Laplacian SVC | **0.4933** | **0.7563** | **0.2024** | 0.4639 | **0.123** |
| Angle-matched RBF-SVC | 0.4743 | 0.7386 | 0.2097 | 0.4767 | 0.162 |
| Product-cosine SVC | 0.4615 | 0.7316 | 0.2121 | 0.4824 | 0.302 |
| IQP-QSVM | 0.4357 | 0.7040 | 0.2206 | 0.4688 | 6.296 |
| Polynomial SVC | 0.4010 | 0.6878 | 0.2325 | **0.4320** | 0.141 |

Paired 2,000-resample patient-cluster bootstrap comparisons:

| IQP-QSVM minus control | Delta AUPRC | 95% CI | Decision |
|---|---:|---:|---|
| Laplacian | **-0.0577** | `[-0.0702, -0.0448]` | Classical wins |
| Angle-matched RBF | **-0.0385** | `[-0.0513, -0.0259]` | Classical wins |
| Product-cosine | **-0.0257** | `[-0.0386, -0.0125]` | Classical wins |
| Polynomial | +0.0346 | `[+0.0202, +0.0481]` | IQP wins only this control |

## Verdict

**NO QUANTUM KERNEL UTILITY DEMONSTRATED.** Three appropriately matched
classical kernels outperform IQP-QSVM, and the best control is approximately 51
times faster per record. The earlier Phase 6Q-A result was caused by an unfair
preprocessing mismatch and is superseded.

This is a useful negative result: the `z8` representation is kernel-sensitive,
but its structure is handled better by a Laplacian similarity than by the
tested IQP fidelity map. Do not spend the next compute budget on VQC/HQNN as a
performance route. Preserve QML as a research ablation and move the main effort
to repeated-seed 1D ECG encoders and corrected multimodal fusion on a GPU.

Folds 9 and 10 remain untouched.
