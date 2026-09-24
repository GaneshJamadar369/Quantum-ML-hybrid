# G6Q-KD3 — Clinical concept and teacher-assistant distillation result

## Execution audit

- Kaggle kernel: `swayamjeetbhagat4/aquire-med-concept-distillation-q4`, Version 1
- Pinned source: `5bbb1bc005c38c2b41313179d7bfc777883088f2`
- Status: complete
- Runtime: approximately 35 minutes
- Device: CUDA exact PyTorch statevector simulator
- Cohort: 17,348 ECGs from 14,958 patients
- Development folds: 1–8 only
- Folds 9 and 10 accessed: no
- Quantum fits: 48, covering six arms across all eight folds
- Non-finite outputs or gradients: none

## Primary comparison

| Model | AUPRC | AUROC | Brier | ECE10 |
|---|---:|---:|---:|---:|
| Full-data q4 MLP assistant | **0.83085** | **0.92231** | **0.09310** | **0.03527** |
| Sample-matched q4 MLP | 0.82884 | 0.92182 | 0.10963 | 0.07862 |
| VQC: assistant JS for all 30 epochs | **0.82718** | 0.92144 | 0.10541 | 0.06989 |
| VQC: concept two-stage | 0.82695 | 0.92094 | 0.11182 | 0.08427 |
| q4 logistic | 0.82686 | 0.92187 | 0.11034 | 0.08434 |
| VQC: concept + assistant two-stage | 0.82671 | 0.92087 | 0.11185 | 0.08424 |
| VQC: reliability-weighted combined | 0.82641 | 0.92088 | 0.11177 | 0.08412 |
| VQC: hard labels | 0.82567 | 0.92046 | 0.11202 | 0.08518 |
| VQC: assistant two-stage | 0.82558 | 0.92062 | 0.11201 | 0.08480 |

The best quantum arm was continuous answer-level JS distillation. Its paired
patient-cluster bootstrap delta over hard-label VQC was `+0.00149`, 95% CI
`[+0.00033, +0.00254]`. This is statistically positive under the prespecified
bootstrap but is below the required `+0.005` practical-improvement threshold.

The same quantum arm lost to the sample-matched q4 MLP by `-0.00169`, 95% CI
`[-0.00340, -0.00004]`. It therefore fails the quantum predictive-advantage
gate as well as the five-seed promotion gate.

## What the concept supervision learned

Concept supervision changed the auxiliary predictions from essentially no
association with held-out measurements to a mean held-out correlation of
approximately `0.20`. The strongest individual correlations for the
concept-only arm were:

- frontal-axis proxy: `0.478`
- global T-inversion count: `0.376`
- high-lateral ST mean: `0.346`
- global negative-ST count: `0.190`
- precordial transition lead: `0.171`

Inferior ST mean (`0.046`) and anterior ST mean (`0.069`) were poorly recovered
from the eight q4 quantum observables. Mean concept MAE changed only from
approximately `0.449` for the unused random auxiliary head to `0.448` after
concept training. Thus the auxiliary task created some rank association but
did not reconstruct the concept vector accurately.

Concept-only two-stage training improved AUPRC over hard VQC by `+0.00127`, but
its interval `[-0.00014, +0.00255]` crossed zero. Combining concepts with the
assistant, with or without reliability weights, was weaker. The ten concepts
therefore provide interpretability evidence, not a validated predictive gain.

## Interpretation

The full-data q4 assistant reached AUPRC `0.83085`, showing that additional
training examples help a small classical q4 learner. The four-qubit student,
trained on the frozen patient-unique 2,000-per-class budget, inherited only a
small fraction of that benefit. Two-stage hard-label refinement erased most of
the assistant gain; continuous JS guidance across all 30 epochs worked better.

This result also does not surpass the earlier clinical-teacher JS VQC
(`0.82747`) or the all-classical clinical+q4 fusion (`0.83798`). Natural-language
rationales, autoregressive on-policy distillation and token filtering remain
inapplicable to this fixed-vector biomedical classifier.

## Decision

**STOP G6Q-KD3. Do not launch five seeds or tune concept weights, concept sets,
teacher temperatures or reliability formulas on pooled development OOF.**

Retain the hard-label q4 VQC as the conservative quantum candidate. The JS
result may be reported as a positive training ablation, but not as the final
champion and not as quantum advantage. Continue to frozen model selection and
fold-9 calibration only after the remaining classical/repeated-seed gates are
resolved.
