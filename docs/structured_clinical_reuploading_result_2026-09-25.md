# G6Q-R2 — Clinically structured re-uploading result

**Date:** 2026-09-25

**Protocol:** `docs/structured_clinical_reuploading_plan_2026-09-25.md`

**Cohort:** 17,348 primary ECGs, 14,958 patients, official folds 1–8

**Verdict:** **STOP BEFORE QUANTUM**

## Validation

- all eight outer folds were represented exactly once;
- patient overlap across fold roles was zero;
- 106 signed deployable features were available;
- every imputer, scaler, PCA and quantile transform was fit on outer-training
  records only;
- the concept encoder never received MI labels;
- folds 9 and 10 were not accessed;
- 16 focused implementation tests passed.

## Primary conditional-information gate

| Model | OOF AUPRC | Brier |
|---|---:|---:|
| q4 HistGradientBoosting | 0.830324 | 0.106989 |
| q4 + four concept coordinates HistGradientBoosting | **0.833537** | **0.104739** |

Observed AUPRC improvement: `+0.003214`. The 2,000-replicate paired
patient-cluster bootstrap estimate was `+0.003236` with 95% interval
`[+0.001724, +0.004741]`.

The interval is positive, so the four concept coordinates contain a small
amount of conditional predictive information beyond q4. The improvement did
not reach the frozen `+0.005` practical-effect threshold. Stage B therefore
did not run.

## Prespecified family analysis

| Concept added to q4 HGB | AUPRC | Bootstrap delta vs q4 HGB (95% CI) |
|---|---:|---:|
| T-wave inversion burden | 0.832425 | +0.002100 `[+0.001066, +0.003121]` |
| Anterior/global ST burden | 0.831774 | +0.001455 `[+0.000328, +0.002551]` |
| R-wave progression | 0.830544 | +0.000226 `[-0.000682, +0.001148]` |
| aVR/reciprocity | 0.830456 | +0.000133 `[-0.000764, +0.000984]` |

The family analysis is descriptive. It does not justify selecting only the
two favorable groups and rerunning the same folds because that would convert
outer-fold evidence into a post-hoc feature-selection loop.

The secondary linear check agreed with the direction but not a large effect:
q4 logistic AUPRC was `0.826981`; q4+c4 logistic was `0.829367`, a bootstrap
delta of `+0.002396` `[+0.001287, +0.003461]`.

## Decision

No Kaggle GPU job was submitted. The frozen protocol required both a positive
interval and at least `+0.005` AUPRC before circuit training. Relaxing that
threshold after observing `+0.00321` would be outcome-driven experimentation.

The structured circuit and complete gated Stage B implementation remain in
the repository for an independently justified future cohort or sealed-fold
study. They are not promoted as the current architecture. The retained system
result remains the existing q4 quantum demonstrator alongside the stronger
classical performance ceiling, with no quantum-advantage claim.
