# AQUIRE-Med implementation and gate ledger

This file is the operational source of truth. A checked **implementation** item means the code exists and is covered by local tests. A data gate remains unchecked until the named artifact has been generated from verified PTB-XL v1.0.3 data and reviewed.

## Implementation complete

- [x] Git baseline preserved before research hardening.
- [x] Locked environment and package metadata added.
- [x] Kaggle script reduced to a thin package driver; GPU disabled.
- [x] PTB-XL identity checks enforce 21,799 rows and checksum-manifest agreement.
- [x] Waveform paths come from `filename_lr` / `filename_hr` and paired files are checked.
- [x] MI codes are derived dynamically and asserted against all 14 v1.0.3 MI codes.
- [x] MI overlap, likelihood metadata and hard-negative groups are retained.
- [x] Fold 10 access guard is active; Fold 9 is excluded from tuning and QC calibration.
- [x] PTB-XL+ joins enforce unique one-to-one keys and an explicit description-based allowlist.
- [x] Commercial/reference and deployable features are separated.
- [x] `ValidatedECG` and `ProcessedECG` contracts preserve masks and provenance.
- [x] 100 Hz powerline detection/correction is disabled.
- [x] 500→100 Hz conversion uses anti-alias filtering and morphology validation.
- [x] QC failures are quarantined outside the primary training HDF5.
- [x] Morphology gate uses polarity-aware lead-II/consensus QRS detection and all matched beats.
- [x] Fold-local masked normalization and lazy HDF5 datasets are implemented.
- [x] Development-only augmentation boundary is enforced.
- [x] Classical tabular OOF baselines and a 1D ResNet architecture are implemented.
- [x] AUPRC-first champion rule with Brier score and inference-cost tie breakers is implemented.
- [x] Pre/post-extraction feature evidence gate and clinical-composite generator are implemented.
- [x] Feature-level effect sizes, FDR, mutual information, redundancy and fold stability are implemented.
- [x] Patient-bootstrap OOF ablation for clinical feature groups is implemented.

## G0–G5 evidence gates

- [ ] **G0 — task contract approved.** Review `docs/scientific_task_contract.md`.
- [ ] **G1 — dataset identity.** Generate `dataset_identity.json`, `file_registry.json` and `waveform_reference_report.json`; require zero errors.
- [ ] **G2 — immutable phenotype/cohorts.** Generate `patient_manifest.csv`, `fold_report.json` and `cohort_report.json`; manually review overlap and uncertain-MI sensitivity cohorts.
- [ ] **G3 — PTB-XL+ deployability.** Generate `ptbxl_plus_audit.json`, coverage tables and `deployable_vs_reference.csv`; approve the allowlist and commercial benchmark-only boundary.
- [ ] **G4 — QC and morphology.** Generate QC operating curves, threshold report, morphology table and visual audit; review false positives. Threshold choice must not use model accuracy.
- [ ] **G5 — reproducibility and leakage.** Generate fold-local normalizers, package/Kaggle parity report and passing real-data test report. Fold 10 access tests must pass.

## G5F — feature evidence gate (required before Phase 6)

- [ ] Generate the development-cohort and raw-waveform statistical audits.
- [ ] Review extractor failures and local-versus-ECGDeli measurement agreement.
- [ ] Review every feature's coverage, effect size, FDR, mutual information and fold stability.
- [ ] Resolve every `REPAIR_OR_EXCLUDE`, `EXCLUDE_NONINFORMATIVE` and `REVIEW_UNSTABLE` recommendation.
- [ ] Review rho ≥ 0.95 redundancy clusters using reliability and clinical interpretability.
- [ ] Accept clinical composites only if patient-bootstrap OOF ablation supports them.
- [ ] Freeze and sign the final approved feature manifest.

## Phase 6 — blocked until G0–G5 pass

- [ ] Generate 8-fold patient-safe OOF predictions on folds 1–8.
- [ ] Compare logistic regression, RBF-SVC, random forest, XGBoost, small MLP and 1D ResNet.
- [ ] Compare waveform, deployable-feature and fused branches.
- [ ] Keep 12SL/Uni-G reference performance in a separate oracle table.
- [ ] Select the provisional champion using pooled OOF AUPRC, then Brier score and latency when within 0.005 AUPRC.
- [ ] Freeze the classical champion and OOF residuals before any `z4/z8` or quantum experiment.
