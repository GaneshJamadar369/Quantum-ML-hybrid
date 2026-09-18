# G5F Feature Evidence Findings

## Decision

**MODIFY before classical champion training.** The clinical-composite group is
promoted to nested evaluation, but the current local measurement extractor is
not yet suitable for freezing.

The evidence run used 17,348 accepted ECGs from 14,958 patients in official
folds 1–8. Fold 9 and fold 10 were not accessed. MI prevalence was 25.18%, and
5,452 records were abnormal hard negatives.

## Feature inventory

- 94 original numeric deployable fields were evaluated.
- 33 prespecified multi-lead clinical composites were added.
- 127 total fields were evaluated for coverage, effect size, FDR, mutual
  information, fold stability and redundancy.

## Clinical-composite ablation

Patient-safe eight-fold logistic screening produced:

| Feature group | AUPRC | AUROC | Brier | Sensitivity at 90% specificity |
|---|---:|---:|---:|---:|
| Existing 94 | 0.5709 | 0.8011 | 0.1838 | 0.4650 |
| Clinical composites only | 0.5100 | 0.7569 | 0.2005 | 0.3853 |
| Existing + clinical | 0.5864 | 0.8105 | 0.1777 | 0.4755 |

The patient-bootstrap median AUPRC gain was 0.0158 with 95% interval
[0.0105, 0.0204]. The median Brier change was -0.0061 with 95% interval
[-0.0071, -0.0050]. The clinical-composite group therefore advances to nested
model evaluation. This is not a champion-model result.

## Measurement blockers

1. NeuroKit2 was unavailable for all 17,348 original extractions. PR, QRS, QT
   and three QTc fields consequently have zero coverage.
2. Local RR median does not agree with ECGDeli: correlation -0.062 and median
   absolute error 270 ms. The simple absolute-peak detector is detecting
   non-R-wave extrema and cannot be used for rhythm measurements.
3. R-amplitude agreement is moderate in I, II and V4–V6, but weak in V1–V3.
   Local QRS alignment and baseline estimation require repair.
4. All twelve per-lead valid-fraction fields are effectively constant in the
   accepted cohort. They remain useful as extraction/QC metadata but should not
   be predictor columns in this cohort.

## Stability and redundancy review

Seven fields showed inconsistent MI/non-MI effect direction across folds:

- I R/S ratio;
- lead II ST60;
- V1 S amplitude;
- V2 range and S amplitude;
- V3 range;
- anterior regional ST mean.

They are retained only for nested ablation or measurement repair, not accepted
as individually established predictors.

Three redundancy pairs exceeded absolute Spearman rho 0.95:

- heart rate and median RR;
- aVF R/S ratio and inferior regional median R/S;
- V3 R/S ratio and anterior regional median R/S.

The final manifest should keep one member of each pair based on measurement
reliability, clinical interpretation and nested ablation.

## Required next action

Replace the absolute-peak extractor with a common multilead/lead-II QRS time
base, install the delineation dependency, measure ST60 from a delineated J
point, and use beat-specific pre-QRS baselines. Re-extract features from the
existing primary HDF5; the preprocessing pipeline does not need to be rerun.
Then repeat ECGDeli agreement and the G5F evidence gate before freezing the
feature manifest.

## Repair implementation

Extractor v0.3 now uses one NeuroKit2-delineated beat timebase for rhythm,
interval and all-lead morphology measurements. It never fabricates interval
values from fallback windows. Re-extraction is restartable and reads directly
from the accepted HDF5, so the verified signal preprocessing is preserved.

The next computation is deliberately limited to a patient-stratified sample of
1,024 ECGs. Frozen timing and amplitude agreement thresholds decide whether a
full 17,348-record extraction is permitted. Classical baseline training remains
blocked until this calibration and the repeated feature-evidence gate pass.
