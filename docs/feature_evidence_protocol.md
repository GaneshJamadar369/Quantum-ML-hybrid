# Feature Evidence and Clinical Validation Protocol

## Objective

Establish whether each locally reproducible ECG measurement is valid, stable,
clinically interpretable and incrementally useful for MI-pattern classification
before it is admitted to classical or quantum modeling.

This protocol never treats statistical significance as proof of clinical value.
Feature decisions require measurement validity, development-fold stability and
patient-level out-of-fold ablation evidence.

## Data boundary

- Development folds 1–8 are the only folds available to this protocol.
- Fold 9 remains reserved for later calibration/validation.
- Fold 10 remains locked until final evaluation.
- QC-failed and structurally rejected records are analysed for failure patterns
  but cannot enter the primary predictive cohort.
- All comparisons and bootstraps preserve the patient boundary.

## Evidence before feature extraction

1. Confirm MI prevalence, hard-negative prevalence, annotation quality, age,
   sex and fold balance.
2. Verify that no patient crosses an official fold.
3. Characterize every lead's range, RMS amplitude, baseline offset and valid
   sample fraction for MI and non-MI records.
4. Quantify how much accepted correction changes the minimally processed
   signal.
5. Audit clear MI, uncertain MI, normal, abnormal non-MI and QC-failed ECGs.

## Candidate feature ontology

The existing deployable measurements comprise rhythm, PR/QRS/QT/QTc timing,
lead-level voltage, R and S amplitudes, R/S balance, ST displacement, T polarity
and measurement-validity features.

Prespecified clinical composites summarize anatomically related lead groups:

- inferior: II, III, aVF;
- high lateral: I, aVL;
- lateral: I, aVL, V5, V6;
- anterior: V1–V4;
- reciprocal inferior/high-lateral and anterior/inferior ST contrasts;
- global ST magnitude and positive/negative lead counts;
- regional T-wave inversion;
- V1–V6 R-wave progression and transition;
- frontal-axis proxy from leads I and aVF.

Q-wave measurements remain deferred until Q onset, offset and amplitude can be
validated against an independent delineation reference. Age/sex/lead-specific
clinical ST thresholds remain deferred until the clinical protocol is frozen.

## Measurement validity

Every local measurement mapped to PTB-XL+ ECGDeli is evaluated by coverage,
correlation, mean absolute error, median absolute error and failure rate.
Commercial 12SL and Uni-G measurements are oracle benchmarks and are never
required for a new submitted ECG.

## Post-extraction statistical evidence

For every original and derived feature, record:

- coverage, missingness, unique values and variance;
- median, IQR and MI/non-MI median difference;
- Mann–Whitney test with Benjamini–Hochberg correction;
- Cliff's delta with a stratified bootstrap 95% interval;
- univariate AUROC, best-direction AUPRC and mutual information;
- predictive signal carried by missingness;
- effect direction and AUROC stability across folds 1–8;
- Spearman redundancy pairs at absolute rho ≥ 0.95.

These statistics generate recommendations only. No feature is automatically
deleted.

## Clinical-composite ablation

A patient-safe eight-fold out-of-fold logistic model compares:

1. the existing deployable measurements;
2. clinical composites alone;
3. existing measurements plus clinical composites.

The combined group is promoted only when the patient-bootstrap 95% interval
shows an AUPRC gain above zero and no Brier-score deterioration. This model is
an evidence screen, not the classical champion selection.

## Decision states

- `KEEP_FOR_NESTED_ABLATION`: sufficient measurement coverage and stability.
- `REDUNDANCY_REVIEW`: highly correlated; retain the more reliable and
  interpretable member only after ablation.
- `REVIEW_UNSTABLE`: effect direction is inconsistent across development folds.
- `REPAIR_OR_EXCLUDE`: measurement coverage is below the prespecified floor.
- `EXCLUDE_NONINFORMATIVE`: constant or near-zero variance.

Final inclusion is frozen in an approved feature manifest before classical
champion selection. The same manifest and fold-local transformations are then
used for all classical and quantum comparisons.

