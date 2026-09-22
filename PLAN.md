# AQUIRE-Med implementation and gate ledger

This file is the operational source of truth. A checked **implementation** item means the code exists and is covered by local tests. A data gate remains unchecked until the named artifact has been generated from verified PTB-XL v1.0.3 data and reviewed.

The final research protocol, model ladder, quantum stop rule and definition of
completion are frozen in
[`docs/research_completion_roadmap.md`](docs/research_completion_roadmap.md).
The nested representation/circuit search for any further quantum-kernel work is
specified in
[`docs/quantum_kernel_optimization_plan.md`](docs/quantum_kernel_optimization_plan.md).
The completed direct VQC and fold-coherent waveform q4 screens, including
their negative overall comparison and next decision gate, are recorded in
[`docs/quantum_core_screen_2026-09-22.md`](docs/quantum_core_screen_2026-09-22.md).
The subsequent QSVC and fusion HQNN screens are recorded in
[`docs/qsvc_hqnn_screen_2026-09-22.md`](docs/qsvc_hqnn_screen_2026-09-22.md).
The compact Transformer experiment and its anti-overfit protocol are frozen in
[`docs/transformer_quantum_plan_2026-09-22.md`](docs/transformer_quantum_plan_2026-09-22.md).
The completed eight-fold encoder and quantum-head results are in
[`docs/transformer_quantum_result_2026-09-22.md`](docs/transformer_quantum_result_2026-09-22.md).
The label-free JEPA → q4/q8 experiment and the gated S4D/Mamba follow-ups are
frozen in
[`docs/jepa_ssm_quantum_representation_plan_2026-09-22.md`](docs/jepa_ssm_quantum_representation_plan_2026-09-22.md).

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
- [x] v0.3 delineated-beat feature extractor and restartable HDF5 re-extraction are implemented.
- [x] Patient-stratified ECGDeli calibration gate and acceptance thresholds are frozen in code.

## G0–G5 evidence gates

- [ ] **G0 — task contract approved.** Review `docs/scientific_task_contract.md`.
- [ ] **G1 — dataset identity.** Generate `dataset_identity.json`, `file_registry.json` and `waveform_reference_report.json`; require zero errors.
- [ ] **G2 — immutable phenotype/cohorts.** Generate `patient_manifest.csv`, `fold_report.json` and `cohort_report.json`; manually review overlap and uncertain-MI sensitivity cohorts.
- [ ] **G3 — PTB-XL+ deployability.** Generate `ptbxl_plus_audit.json`, coverage tables and `deployable_vs_reference.csv`; approve the allowlist and commercial benchmark-only boundary.
- [ ] **G4 — QC and morphology.** Generate QC operating curves, threshold report, morphology table and visual audit; review false positives. Threshold choice must not use model accuracy.
- [ ] **G5 — reproducibility and leakage.** Generate fold-local normalizers, package/Kaggle parity report and passing real-data test report. Fold 10 access tests must pass.

## G5F — feature evidence gate (required before Phase 6)

- [x] Generate the development-cohort and raw-waveform statistical audits.
- [x] Review extractor failures and local-versus-ECGDeli measurement agreement. **MODIFY:** intervals absent; RR invalid; V1–V3 R amplitude weak.
- [x] Review every feature's coverage, effect size, FDR, mutual information and fold stability.
- [ ] Resolve every `REPAIR_OR_EXCLUDE`, `EXCLUDE_NONINFORMATIVE` and `REVIEW_UNSTABLE` recommendation.
- [ ] Review rho ≥ 0.95 redundancy clusters using reliability and clinical interpretability.
- [x] Accept clinical composites only if patient-bootstrap OOF ablation supports them. **Promoted to nested evaluation:** ΔAUPRC 0.0158 [0.0105, 0.0204].
- [x] Run v0.3 on the 1,024-patient stratified calibration sample. **STOP/REPAIR:** RR passed; four interval fields failed; five of eight amplitudes passed.
- [x] Exclude unvalidated 100 Hz PR/QRS/QT/QTc families from predictor eligibility rather than relaxing their failed thresholds.
- [x] Run signed-fiducial v0.4 on 1,024 previously unseen patients. **PASS:** RR and six of eight R-amplitude pairs passed; V2/V3 remain under exclusion/review.
- [x] If calibration passes, re-extract all 17,348 accepted ECGs and repeat the full evidence gate. **PASS:** Full 17,348 re-extraction completed on Kaggle (350826460); clinical composites promoted (ΔAUPRC 0.0164 [0.0122, 0.0208]).
- [x] Freeze and sign the final approved feature manifest. **DONE:** `configs/approved_feature_manifest_v0_4.json` signed (106 approved, 20 excluded).

## G5.5 — Pre-Modelling Data Conditioning (Advanced Feature Engineering)

- [x] **Stage 1 & 2:** Fold-local P1/P99 outlier clipping and Yeo-Johnson power transforms to guard scalers.
- [x] **Stage 3 & 4:** Pre-specified clinical interaction feature engineering (5 cardiologically motivated pairs) and fold-local zero-variance filter.
- [x] **Stage 5:** Hybrid ANOVA-F + Mutual Information fold-local selection.
- [x] **Stage 6:** Layered class imbalance conditioning (SMOTE-ENN on training folds + hard-negative sample weights).
- [x] **Stage 7:** Redundancy cluster resolution (`configs/redundancy_resolution.json`).
- [x] **Stage 8 & 10:** Fold-local Platt probability calibration and subgroup/hard-negative sensitivity analysis.
- [x] **Stage 9:** Post-OOF SHAP attribution audit implementation.
- [x] **Stage 11:** Split conformal prediction sets (RAPS, 90% coverage) implementation.

## Phase 6 — blocked until G0–G5 pass

- [x] Generate 8-fold patient-safe classical OOF predictions on folds 1–8 using the G5.5 conditioned feature matrix. **Artifact winner:** HistGradientBoosting, AUPRC 0.719802.
- [ ] Complete the promised classical comparison. Logistic regression, random forest, HistGradientBoosting, XGBoost and MLP ran; **RBF-SVC is missing from `_models()`**.
- [x] Run the provisional waveform ResNet benchmark. **Development result:** AUPRC 0.791308 from one seed; uncertainty/repeated-seed validation remains open.
- [ ] Rerun multimodal fusion. The 2026-09-21 audit proved the original one-key attention ignored the tabular query, so its 0.797322 result is invalid as fusion evidence.
- [ ] Keep 12SL/Uni-G reference performance in a separate oracle table.
- [ ] Select the provisional champion using pooled OOF AUPRC, then Brier score and latency when within 0.005 AUPRC.
- [ ] **BLOCK:** Champion selection is blocked until Stage 9 SHAP attribution audit passes clinical plausibility checks.
- [ ] Freeze the classical champion and OOF residuals before any `z4/z8` or quantum experiment.

## G6R — modeling repair gate added after research audit (2026-09-21)

- [x] Record the exact 12-hour timeout and latest metric-print failure in `docs/research_code_audit_2026-09-21.md`.
- [x] Correct multimodal fusion so both tabular and waveform inputs affect predictions; add modality-dependence regression tests.
- [x] Replace the ineffective fixed-unitary kernel with a data-dependent IQP feature map and kernel PSD diagnostics.
- [x] Define fold-local training-only PCA `z8` and a matched RBF-SVC comparator.
- [x] Add per-fold atomic checkpoints, resume support and a real quantum preflight test.
- [x] Remove automatic Fold 9/10 evaluation from the quantum Kaggle runner.
- [x] Run Phase 6Q-A (IQP-QSVM versus matched RBF-SVC) and download complete artifacts. **Completed on Kaggle in 4.8 minutes:** IQP-QSVM AUPRC 0.4357 versus RBF-SVC 0.3426.
- [x] Compute paired patient-cluster bootstrap confidence intervals for delta AUPRC and delta Brier. **Delta AUPRC +0.0931 [0.0779, 0.1083]; delta Brier -0.0197 [-0.0220, -0.0173].** Subgroup uncertainty remains open.
- [x] Compare the IQP kernel against stronger matched classical kernels on identical angle coordinates. **Phase 6Q-B reverses the apparent Phase 6Q-A win:** Laplacian AUPRC 0.4933, angle-RBF 0.4743, product-cosine 0.4615, IQP-QSVM 0.4357, polynomial 0.4010.
- [x] Evaluate polynomial, Laplacian and product-cosine controls on identical fold-local `z8`, records, sample budget and calibration. **QSVM minus Laplacian ΔAUPRC -0.0577 [-0.0702, -0.0448].**
- [x] Record the Phase 6Q kernel branch as a valid negative result against stronger classical kernels. Subsequent direct-q4 VQC screens are separately marked exploratory; prioritize repeated-seed waveform encoders and corrected fusion before champion selection.
- [x] Audit VQC/HQNN execution truth. Neither model has a real-data OOF result; both previously failed batched backpropagation under parameter-shift.
- [x] Repair VQC/HQNN simulator gradients with adjoint/backprop differentiation and add batched forward/backward regression tests.
- [x] Export fold-coherent waveform embeddings using one encoder per outer fold; verify patient isolation and complete 17,348-record OOF coverage. **Representation OOF AUPRC 0.7878; supervised-encoder ablation.**
- [x] Run two direct q4 VQC screens on Kaggle: approved clinical features and fold-coherent waveform embeddings, each against matched-input MLP and RBF controls. **Clinical VQC 0.5087 vs RBF 0.5133; waveform VQC 0.7325 vs MLP 0.7583. No overall quantum win.**
- [x] Run fold-coherent q4 IQP-QSVC with RBF and Laplacian controls. **IQP-QSVC AUPRC 0.6572 vs RBF 0.7433 and Laplacian 0.7244; negative result with patient-bootstrap intervals below zero.**
- [x] Run q4 fusion HQNN with waveform and approved clinical features against same-input fusion MLP. **HQNN AUPRC 0.7424 vs MLP 0.7526; finite gradients, but no primary-metric win.**
- [ ] Repeat the quantum-head experiment with inner-fold-selected representations, patient-unique sample-size/regularization ablations and at least five seeds; add exact parameter-count and circuit-removal controls. Do not expand to q8 based on the narrow waveform VQC–RBF comparison alone.
- [ ] Run Phase 6C-R: corrected ResNet/fusion with modality controls and five prespecified seeds on GPU.
- [x] Implement a 271,041-parameter patch Transformer with 100 time tokens, width 96, four heads, three pre-norm blocks, patient-separated inner early stopping and full-outer-training retrain. Synthetic shape, gradient, patient-split and training smoke tests pass.
- [x] Complete the eight-fold Kaggle GPU Transformer representation export and audit selected epochs and train/inner/outer gaps. **Raw OOF AUPRC 0.8325 versus CNN 0.7878; mean train/outer AUPRC gap 0.0684 versus CNN 0.2109.**
- [x] Run the same q4 VQC/MLP/RBF head screen as the CNN study. **Transformer-input VQC 0.8148, MLP 0.8112, RBF 0.7747. VQC–MLP 95% patient-bootstrap interval crosses zero; no established quantum win.**
- [x] Compare Transformer q4 against stronger classical and quantum heads on identical patients and coordinates. **q4 logistic 0.8224 beat VQC 0.8148 (VQC minus logistic ΔAUPRC -0.0075 [-0.0125, -0.0030]); IQP-QSVC 0.3190 failed; h128 logistic ceiling 0.8247.**
- [x] Run Transformer-input fusion HQNN against its identical-input MLP. **HQNN 0.8049 vs MLP 0.8164; ΔAUPRC -0.0114 [-0.0167, -0.0061]. Trained negative result with finite gradients.**
- [ ] Repeat the Transformer quantum-head result with five seeds and stronger matched classical controls, including exact parameter-count and classical circuit-removal ablations; keep all tuning in inner folds.
- [ ] Screen xResNet/Inception and reproducible pretrained ECG encoders under the same patient-safe protocol.
- [ ] Run VQC/HQNN only as matched frozen-embedding ablations after the classical development champion is selected.
- [ ] Implement the projected-quantum-kernel and trainable sparse-IQP search only under the nested protocol; do not tune against pooled outer-fold labels.
- [x] Freeze a label-free Transformer-JEPA experiment that isolates the representation objective before changing the backbone.
- [x] Implement and locally validate masked token/global latent prediction with an EMA target encoder and explicit collapse monitoring.
- [ ] Complete the eight-fold label-free Transformer-JEPA Kaggle export and pass the J1 acceptance gate.
- [ ] Compare unsupervised PCA/quantile q4 and q8 using VQC, logistic, matched MLP and RBF controls on identical patients.
- [ ] Run Conv-S4D-JEPA only if the label-free representation gate supports further backbone work; hold Mamba until the S4D decision.
- [ ] Rerun corrected fusion with tabular-only, waveform-only and modality-shuffle ablations over at least three seeds.
- [ ] Replace the invalid development-fold conformal claim with Fold-9 calibration and one-time Fold-10 coverage evaluation after champion freeze.
