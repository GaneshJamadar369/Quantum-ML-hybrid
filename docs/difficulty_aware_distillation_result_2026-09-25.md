# G6Q-KD4 — Difficulty-aware distillation result

## Execution audit

- Kaggle kernel: `swayamjeetbhagat4/aquire-med-difficulty-aware-kd-q4`
- Completed version: 2
- Pinned source: `eb65cc23ae8b3d3270827e904e17b99d7a00c547`
- Cohort: 17,348 ECGs from 14,958 patients
- Development folds: 1–8 only
- Fold 9/10 access: none
- Device: CUDA exact PyTorch statevector simulator
- Hard-label records filtered or reweighted: none

Version 1 stopped before model training because a redundant audit compared
reliability bins computed before and after float32 conversion. Version 2
removed that invalid duplicate comparison; the original inner-OOF reliability
calculation and experiment contract were unchanged.

## Standalone predictors

| Model | AUPRC | AUROC | Brier | ECE10 |
|---|---:|---:|---:|---:|
| Full-data q4 MLP assistant | **0.83085** | **0.92231** | **0.09310** | **0.03527** |
| Sample-matched q4 MLP | 0.82884 | 0.92182 | 0.10963 | 0.07862 |
| Curriculum difficulty VQC | **0.82724** | 0.92117 | 0.10426 | 0.06663 |
| Uniform-JS VQC | 0.82718 | 0.92144 | 0.10541 | 0.06989 |
| q4 logistic | 0.82686 | 0.92187 | 0.11034 | 0.08434 |
| Dynamic-difficulty VQC | 0.82672 | 0.92064 | 0.10442 | **0.06560** |
| Hard-negative-difficulty VQC | 0.82639 | 0.92060 | 0.10457 | 0.06657 |
| Hard-label VQC | 0.82567 | 0.92046 | 0.11202 | 0.08518 |
| Agreement-weighted VQC | 0.82543 | 0.92155 | 0.10780 | 0.07639 |

Curriculum weighting was the best difficulty-aware arm. Compared with hard
VQC, its paired AUPRC delta was `+0.00155`, 95% patient-bootstrap CI
`[+0.00032, +0.00269]`. The result is statistically positive but below the
prespecified `+0.005` practical promotion threshold.

Curriculum versus uniform JS was only `+0.00006`, 95% CI
`[-0.00020, +0.00027]`. Difficulty weighting therefore did not improve
standalone discrimination beyond ordinary JS in a resolved way.

Curriculum VQC versus the sample-matched q4 MLP was `-0.00163`, 95% CI
`[-0.00341, +0.000003]`; versus q4 logistic it was `+0.00033`, 95% CI
`[-0.00190, +0.00256]`. Neither is a quantum predictive win.

Difficulty-aware JS did materially improve probability quality versus hard
VQC. Curriculum reduced Brier by approximately `0.00775` with its entire paired
interval below zero and reduced ECE from `0.08518` to `0.06663`. This remains a
training/calibration benefit rather than a primary discrimination win.

## Clinical fusion

| Fusion | AUPRC | Brier | ECE10 |
|---|---:|---:|---:|
| Clinical + q4 MLP | **0.83802** | **0.08895** | **0.01210** |
| Clinical + curriculum VQC | **0.83697** | 0.08927 | 0.01553 |
| Clinical + uniform-JS VQC | 0.83672 | 0.08934 | 0.01539 |
| Clinical + hard-negative VQC | 0.83635 | 0.08935 | 0.01522 |
| Clinical + dynamic VQC | 0.83632 | 0.08941 | 0.01529 |
| Clinical + agreement VQC | 0.83594 | 0.08945 | 0.01592 |
| Clinical + hard VQC | 0.83579 | 0.08933 | 0.01431 |

Curriculum fusion improved over hard-VQC fusion by `+0.00117`, 95% CI
`[+0.00044, +0.00192]`. It also improved over uniform-JS fusion by a very small
`+0.00025`, 95% CI `[+0.00013, +0.00038]`.

However, curriculum fusion remained below all-classical fusion by `-0.00107`,
95% CI `[-0.00224, +0.00002]`, and had a significantly worse Brier score. The
system-level quantum-value gate therefore failed.

## Final decision

**STOP knowledge-distillation development.** Difficulty-aware curriculum is a
valid positive ablation over hard VQC and provides better probability quality,
but it does not reach the practical promotion threshold, improve meaningfully
over uniform JS, beat the identical-q4 MLP, or beat the all-classical fusion.

No five-seed difficulty run, multiplier search, clipping search or additional
teacher routing should be launched on the development OOF labels. Preserve the
hard VQC as the conservative required quantum core, report JS/curriculum as
training ablations, and proceed only after the final architecture decision to
frozen fold-9 calibration and one-time fold-10 evaluation.
