# Research plan: evidence-routed classical/quantum ECG fusion

## Decision before the new experiment

**MODIFY the representation; do not promote the current q4 VQC.** The fast
OOF junction audit used all 17,348 accepted ECGs from 14,958 patients in
official development folds 1–8. Folds 9 and 10 were not accessed.

| OOF model | AUPRC | AUROC | Brier |
|---|---:|---:|---:|
| Transformer + clinical + current q4 VQC, diagnostic stack | 0.84111 | 0.92818 | 0.08804 |
| Transformer + clinical, diagnostic stack | 0.84102 | 0.92817 | 0.08805 |
| Transformer head | 0.83250 | 0.92320 | 0.10327 |
| Current q4 VQC | 0.81476 | 0.91723 | 0.10875 |
| Clinical HistGradientBoosting | 0.71980 | 0.87221 | 0.11791 |

The q4 VQC and Transformer scores have Spearman correlation **0.9510**. In the
three-input diagnostic stack, the mean coefficient on the q4 score is only
**0.0262**, whereas the Transformer and clinical coefficients are 1.4663 and
0.3624. The q4 score improves AUPRC by about **0.00010** over the two-classical-
expert stack. This is insufficient evidence of useful quantum complementarity.
The audit stack is not a final performance estimate because its meta-level
evaluation is not fully nested.

The practical conclusion is that the supervised PLS q4 representation mostly
repeats the Transformer's strongest MI direction. More qubits cannot repair
that redundancy. The new quantum representation must target information left
after a strong classical prediction.

## Scientific basis

The routing strategy follows five findings from primary sources:

1. ECG evidence for MI is spatial and relational: contiguous-lead ST changes,
   reciprocal changes, Q/QS patterns and evolving T-wave changes must be
   interpreted together. These motivate anatomical feature families rather
   than independent lead values. See the
   [Fourth Universal Definition of Myocardial Infarction](https://www.ahajournals.org/doi/10.1161/CIR.0000000000000617).
2. PTB-XL+ contains ECGDeli and commercial measurements, but 12SL and Uni-G are
   closed-source and therefore reference-only. The deployable path must use
   locally reproducible measurements. See the
   [official PTB-XL+ documentation](https://physionet.org/content/ptb-xl-plus/1.0.0/).
3. A high-dimensional quantum space does not itself imply predictive advantage;
   strong classical learners can recover the same geometry from data. Quantum
   and classical kernels must be compared on the same samples and inputs. See
   [Power of data in quantum machine learning](https://www.nature.com/articles/s41467-021-22539-9).
4. Deep, expressive and highly entangled quantum embeddings can make kernels
   concentrate and lose input information. Local observables, shallow circuits
   and explicit kernel-health checks are required. See
   [Exponential concentration in quantum kernel methods](https://www.nature.com/articles/s41467-024-49287-w).
5. Fusion weights must be trained from inner OOF predictions. The 2026 clinical
   stacking study demonstrates the right nesting structure, although our study
   keeps every dimensional reduction fold-local and evaluates continuous
   scores rather than copying its global-PCA/binary-meta-feature choices. See
   [Adaptive quantum kernel selection via leakage-free stacking](https://doi.org/10.1038/s41598-026-56928-1).

Nested selection is essential in biomedical prediction because selecting
features outside the resampling loop biases performance estimates. See
[nestedcv](https://pmc.ncbi.nlm.nih.gov/articles/PMC10125905/) and
[consensus nested cross-validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC7776094/).

## Feature roles

The existing signed manifest remains the hard boundary: 106 locally
reproducible measurements are eligible and 20 failed/constant measurements are
excluded. No pooled audit can override the measurement-validity gate.

### Classical expert

The classical expert receives:

- all 106 approved deployable measurements;
- 128 fold-coherent Transformer coordinates;
- no diagnosis codes, reports, commercial diagnostic statements, patient IDs,
  QC-failure proxies or excluded interval/amplitude measurements.

The clinical branch uses shallow histogram boosting because it handles
nonlinear thresholds and missing values. The waveform branch uses regularized
logistic regression on h128. A two-score logistic stack is learned from inner
OOF predictions. Main-effect features belong here because classical models
already estimate them accurately and cheaply.

High-confidence classical families from the pooled audit include inferior and
lateral QRS amplitude/R:S balance, regional ST summaries, reciprocal ST
contrast, anterior/lateral morphology, and global ST/T extent. These are not
deleted from the quantum candidate bank until nested ablation proves that
duplication is harmful.

### Quantum residual expert

The quantum branch receives **four residual coordinates**, not four raw ECG
columns. For every outer fold:

\[
r_i = y_i - \hat p_{C,i}^{\mathrm{inner\ OOF}}
\]

PLS is fit only in the outer-training data to find four directions that
covary with the classical residual. Each coordinate is quantile-mapped to
\([-\pi,\pi]\) and encoded into one qubit. Three source banks are tested:

1. **h128 residual:** waveform directions missed by the classical stack;
2. **clinical residual:** combinations of the 106 validated measurements missed
   by clinical boosting;
3. **combined residual:** joint h128 and clinical directions.

This is soft evidence routing. It is preferable to assigning “ST to quantum”
or “amplitude to classical” by intuition because a quantum circuit operates on
interactions among coordinates, and the same clinical measurement can have a
strong main effect plus a useful residual interaction.

The pooled audit generated hypotheses for nested testing. Eleven features had
stable associations with both classical residual and current quantum/classical
loss difference without a strong stable marginal effect. Examples include
aVL R amplitude/R:S ratio, lead-I range, V1 R amplitude/R:S ratio, V2/V4 ST60,
aVF range and high-lateral R/R:S composites. Thirty further measurements,
especially inferior QRS, T-polarity and regional morphology features, require
shared-branch ablation. These are **hypotheses only**; no feature is routed in
the final model from pooled OOF statistics.

## Statistical decisions at each junction

| Junction | Question | Required analysis | Gate |
|---|---|---|---|
| J0 measurement | Can a new ECG reproduce the value? | coverage, missingness, ECGDeli agreement, failure rate, Bland–Altman error | failed measurements never enter either branch |
| J1 marginal evidence | Is the feature associated with MI? | Cliff's delta, Mann–Whitney p, BH-FDR, univariate AUROC/AUPRC, eight-fold direction stability | descriptive evidence only; no automatic deletion |
| J2 redundancy | Does it duplicate another measurement? | Spearman clusters, clinical meaning, reliability, fold-local ablation | keep the most reliable member or group-regularize |
| J3 residual utility | Does it explain classical errors? | association with cross-fitted residual, conditional MI, fold-sign stability, family-level permutation | candidate only when repeated inside outer training |
| J4 quantum geometry | Does the encoding preserve useful differences? | observable variance, effective dimension/rank, kernel-target alignment, off-diagonal variance, PSD and shot-noise sensitivity | stop concentrated/identity-like maps |
| J5 model contribution | Does the quantum model add unique prediction? | VQC vs matched MLP; classical-only vs fusion; quantum removal/shuffle; three residual source banks | patient-bootstrap ΔAUPRC CI must exceed zero |
| J6 clinical value | Is the gain useful and safe? | sensitivity at 90% specificity, hard-negative FPR, calibration, decision curve, sex/age/QC/MI-subtype strata | no material subgroup regression; calibration non-inferior |
| J7 reproducibility | Does it survive randomness and hardware constraints? | ≥5 seeds, circuit-depth/shot/noise ablations, latency and cost | stable direction and practical resource budget |

Feature-level testing uses hierarchical multiplicity control: test anatomical
families first, then apply BH-FDR within a family only if its group test passes.
Selection frequency across inner folds and seeds is reported; a feature is not
called stable from one fit. Correlated features are interpreted as groups, so
SHAP ranks are never presented as causal effects.

## Frozen experiment now implemented

For each held-out official fold 1–8:

1. Load the fold-specific Transformer encoder representation; training and
   validation patients are disjoint.
2. Use the other seven official folds as the inner loop.
3. Generate inner-OOF h128-logistic and clinical-HistGB predictions.
4. Fit the classical score stack on those OOF predictions and score the outer
   fold after refitting both base experts on all outer-training patients.
5. Compute outer-training residuals and select a patient-unique sample of 500
   MI and 500 non-MI ECGs with hard-negative representation.
6. Fit each of the three four-coordinate residual PLS representations using
   only the selected outer-training sample.
7. Train a two-layer q4 exact-statevector VQC with residual-weighted BCE.
8. Train a parameter-count-matched MLP on the identical coordinates, patients,
   weights and epochs.
9. Apply prespecified additive logit fusion weights 0.10, 0.25 and 0.50. The
   primary comparison is 0.25; weights are not selected from outer results.
10. Save every OOF score, fold audit, gradient range, patient/hash boundary and
    paired patient-cluster bootstrap result.

Primary endpoint: patient-pooled OOF AUPRC. Secondary endpoints: AUROC, Brier,
log loss, MI-versus-hard-negative AUPRC, sensitivity at 90% specificity,
latency, circuit cost and subgroup performance.

## Promotion rule

A residual quantum source advances only when its **0.25 fusion** satisfies all
of the following:

- VQC fusion minus classical-only ΔAUPRC has a 95% patient-cluster bootstrap
  interval entirely above zero;
- VQC fusion minus identical-input matched-MLP fusion also has an interval
  entirely above zero;
- gain is at least 0.005 AUPRC or improves a prespecified clinical operating
  point without degrading calibration;
- all eight folds have finite gradients and no single fold supplies the gain;
- circuit-removal and feature-shuffle controls reduce performance;
- the direction survives at least five prespecified seeds.

If the fixed-weight screen passes, the next run replaces fixed fusion with a
fully nested learned nonnegative residual weight. If it fails, no fusion is
called a winner; the honest result is that the quantum branch adds no unique
information to the current representation.

