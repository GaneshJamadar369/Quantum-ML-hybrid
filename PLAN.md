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
Its completed negative result and stop decision are recorded in
[`docs/jepa_quantum_result_2026-09-23.md`](docs/jepa_quantum_result_2026-09-23.md).
The supervised h128→q8/q12/q16 direct-qubit experiment is frozen in
[`docs/supervised_qubit_scaling_plan_2026-09-23.md`](docs/supervised_qubit_scaling_plan_2026-09-23.md).
Its completed cross-width result and stop decision are in
[`docs/supervised_qubit_scaling_result_2026-09-23.md`](docs/supervised_qubit_scaling_result_2026-09-23.md).
The isolated q4-representation/16-qubit-capacity experiment is frozen in
[`docs/q4_on_16_qubit_capacity_plan_2026-09-23.md`](docs/q4_on_16_qubit_capacity_plan_2026-09-23.md).
Its completed result and stop decision are recorded in
[`docs/q4_on_16_qubit_capacity_result_2026-09-23.md`](docs/q4_on_16_qubit_capacity_result_2026-09-23.md).
The completed independent direct-label dual-route experiment and its negative
quantum-value decision are recorded in
[`docs/independent_dual_route_fusion_result_2026-09-24.md`](docs/independent_dual_route_fusion_result_2026-09-24.md).

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
- [x] Complete the eight-fold label-free Transformer-JEPA Kaggle export and pass the J1 acceptance gate. **17,348 ECGs/14,958 patients; zero overlap; finite full-rank h128; eight distinct encoders; folds 9/10 sealed.**
- [x] Compare unsupervised PCA/quantile q4 and q8 using VQC, logistic, matched MLP and RBF controls on identical patients. **Negative: q4 VQC 0.3865; q8 VQC 0.3835; q8 matched MLP 0.4412. q8 helped classical heads but not VQC.**
- [x] Apply the frozen stop rule to Conv-S4D-JEPA and Conv-Mamba-JEPA. **Not launched: label-free VQC fell by about 0.428 AUPRC versus the supervised VQC reference and failed both representation gates.**
- [x] Implement the controlled supervised Transformer qubit-scaling screen: fold-local PLS q8/q12/q16, matching 8/12/16-qubit exact VQCs, identical-input controls and GPU statevector execution.
- [x] Complete and audit the independent q8, q12 and q16 Kaggle jobs; promote a width only if it improves q4 by ≥0.005 AUPRC with a positive patient-bootstrap interval. **STOP: q8 0.7532, q12 0.7939 and q16 0.7805 all fell below q4 VQC 0.8148 with paired intervals entirely below zero.**
- [x] Apply the gate to full-h128 re-uploading and amplitude encoding. **Not launched because the required positive direct-width trend was absent.**
- [x] Implement a capacity-isolation VQC that keeps the winning q4 representation and replicates its four angles across a 16-qubit ring, with a parameter-matched q4 MLP control.
- [x] Complete the q4-on-16-qubit Kaggle OOF screen and require ≥0.005 AUPRC over the four-qubit VQC with a positive patient-bootstrap interval before promotion. **STOP: 16-qubit VQC 0.7979 versus four-qubit VQC 0.8148; paired Δ -0.0168 [-0.0233, -0.0105].**
- [ ] Rerun corrected fusion with tabular-only, waveform-only and modality-shuffle ablations over at least three seeds.
- [ ] Replace the invalid development-fold conformal claim with Fold-9 calibration and one-time Fold-10 coverage evaluation after champion freeze.

## G6Q-R — evidence-routed residual quantum fusion (2026-09-23)

- [x] Audit OOF score complementarity across the Transformer, clinical HistGB and q4 VQC. **Finding:** VQC–Transformer Spearman rho 0.9510; adding VQC to the Transformer+clinical diagnostic stack changes AUPRC only 0.84102→0.84111.
- [x] Generate pooled development routing hypotheses using measurement validity, marginal stability, classical residual association and quantum/classical loss difference. Treat them as hypotheses, not final feature assignments.
- [x] Freeze the three residual q4 source banks, matched-MLP control, fixed fusion weights and promotion rule in `docs/feature_routing_quantum_fusion_plan_2026-09-23.md`.
- [x] Implement and smoke-test the fold-safe residual routing runner with inner official-fold classical OOF predictions and outer patient isolation.
- [x] Complete the eight-fold Kaggle GPU screen for h128-residual, clinical-residual and combined-residual q4 VQCs. **Completed:** 17,348 ECGs/14,958 patients; all eight folds; finite gradients; folds 9/10 sealed.
- [x] Apply the promotion gate. **STOP:** classical expert AUPRC 0.837996; primary h128/clinical/combined VQC fusions 0.837933/0.837174/0.837544. All paired AUPRC intervals versus classical and matched MLP include or lie below zero.
- [x] Apply the conditional follow-up rule. **Not launched:** five seeds, learned fusion and hardware/noise expansion require a positive fixed-weight result, which was absent. Result recorded in `docs/feature_routing_quantum_fusion_result_2026-09-23.md`.

## G6Q-I — independent dual-route fusion (completed 2026-09-24)

Research protocol: `docs/independent_dual_route_fusion_plan_2026-09-24.md`.
This is a new direct-label experiment and contains no residual target,
classical-error weighting or uncertainty gate.

- [x] Freeze Route A: approved clinical measurements to the classical expert;
  Transformer h128 → direct-label PLS-q4 to the quantum expert.
- [x] Freeze Route B: QRS/rhythm features to the classical expert and disjoint
  ST/T spatial features to the quantum expert.
- [x] Implement fold-local route transforms and prove folds 9/10 remain sealed.
- [x] Train both branches independently on MI/non-MI and save raw OOF logits.
- [x] Train a one-neuron logistic fusion using meta-fold cross-fitting only.
- [x] Run same-input logistic/MLP/RBF/Laplacian, all-classical fusion,
  route-swap, quantum-removal and quantum-shuffle controls.
- [x] Apply the promotion gate. **STOP/AVOID:** Route A VQC fusion 0.8010 lost
  to matched classical fusion 0.8212 (paired delta -0.0201
  [-0.0252, -0.0149]); Route B VQC fusion 0.6792 lost to matched MLP fusion
  0.6863 and the all-feature classical oracle 0.7125. Quantum removal hurt,
  but identical-input classical learners used the same information better.

## G6Q-AF — advanced quantum-input fusion and constrained adapters (completed 2026-09-24)

Protocol and result:
[`docs/advanced_quantum_input_fusion_protocol_2026-09-24.md`](docs/advanced_quantum_input_fusion_protocol_2026-09-24.md),
[`docs/advanced_quantum_input_fusion_result_2026-09-24.md`](docs/advanced_quantum_input_fusion_result_2026-09-24.md).

- [x] Implement eight clinically grouped tokens with exact coverage of all 106
  approved features and preserved missing-value masks.
- [x] Run FiLM, rank-4 low-rank bilinear, clinical-query patch attention and
  cross-attention-plus-bilinear q4 screens in parallel with identical-q4
  logistic/MLP/RBF and no-entanglement controls. **STOP:** VQC AUPRC
  0.7414/0.7224/0.7287/0.6382; every arm failed its representation and quantum
  gates.
- [x] Run zero-initialized residual angle adapters using h128 alone and
  h128+clinical inputs. **STOP:** adapted VQCs 0.7700/0.7637 versus unchanged
  two-layer VQC 0.8228.
- [x] Run six-parameter information-preserving orthogonal q4 mixers with
  three-layer ring and ladder VQCs. **STOP:** 0.8094/0.8106 versus identical-q4
  logistic 0.8286/0.8306.
- [x] Re-evaluate the unchanged q4 VQC at 2,000 patient-unique examples per
  class and 30 epochs. **Best retained quantum candidate:** VQC 0.82284,
  logistic 0.82731, matched MLP 0.82426, RBF 0.77560. VQC minus logistic
  patient-bootstrap interval `[-0.00681, -0.00206]`; no quantum win.
- [x] Apply the stop rule: no more input-fusion, width, depth or topology
  searches on pooled development OOF. Proceed only to five-seed confirmation,
  then frozen fold-9 calibration and one-time fold-10 evaluation.

## G6Q-KD2 — calibrated bidirectional-divergence distillation (2026-09-25)

Protocol:
[`docs/divergence_distillation_protocol_2026-09-25.md`](docs/divergence_distillation_protocol_2026-09-25.md).

- [x] Freeze the retained Transformer→PLS-q4→four-qubit two-layer ring VQC,
  2,000-per-class sample and 30-epoch budget.
- [x] Implement tested hard-label, temperature-scaled forward KL, 25% reverse
  mixture, symmetric KL, Jensen-Shannon and pure reverse-KL objectives.
- [x] Replace in-sample teacher targets with inner-fold OOF clinical-teacher
  probabilities and outer-training sigmoid calibration.
- [x] Implement identical-q4 logistic/MLP controls, cross-fitted meta-fusion,
  complete OOF export and paired patient-bootstrap decisions.
- [x] Pass the real 17,348-record/14,958-patient preflight with all eight
  Transformer archives; folds 9 and 10 remain sealed.
- [x] Submit the pinned single-seed screen to Kaggle GPU as
  `swayamjeetbhagat4/aquire-med-divergence-kd-q4` Version 1; source revision
  `9d128ad185fbf8779ed730612e3ac551b8afb894`.
- [x] Complete the prespecified single-seed Kaggle GPU divergence screen.
  **Result:** JS VQC 0.82747, hard VQC 0.82567, q4 logistic 0.82686; JS-minus-hard
  +0.00180 `[-0.00014, 0.00358]`. Pure reverse KL 0.82538 improved Brier but
  did not improve AUPRC.
- [x] Apply the promotion gate. **STOP:** no objective reached +0.005 AUPRC
  with a positive paired interval. Matched all-classical fusion remained best
  at 0.83798 versus JS quantum fusion 0.83563.
- [x] Apply the conditional five-seed rule. **Not launched:** the single-seed
  promotion gate failed. Retain the hard-label VQC and defer probability
  calibration to frozen fold 9.

## G6Q-KD3 — clinical concept and teacher-assistant distillation (2026-09-25)

Protocol:
[`docs/concept_rationale_distillation_plan_2026-09-25.md`](docs/concept_rationale_distillation_plan_2026-09-25.md).

- [x] Freeze ten deployable, non-diagnostic ECG concept targets and their
  outer-fold-only missingness, imputation and robust-scaling contract.
- [x] Implement an inner-official-fold OOF q4 MLP teacher assistant with
  outer-training sigmoid calibration and calibration-aware reliability.
- [x] Implement equal-budget hard, answer-only, two-stage concept and combined
  concept/assistant VQC arms. The production MI readout remains unchanged and
  the auxiliary concept head is training-only.
- [x] Add identical-q4 logistic/MLP controls, concept fidelity metrics, complete
  OOF exports and paired patient-cluster bootstrap gates.
- [x] Pass the 17,348-record real-data preflight with folds 9/10 sealed.
  **PASS:** 17,348 ECGs, 14,958 patients, eight complete folds and no access to
  folds 9/10.
- [x] Complete the pinned Kaggle GPU screen and apply the +0.005 AUPRC and
  positive-interval gate before any five-seed follow-up. **STOP:** answer-only
  JS was the best VQC at 0.82718 versus hard VQC 0.82567 (paired delta
  +0.00149 `[+0.00033, +0.00254]`), below the +0.005 gate and below the
  identical-q4 MLP at 0.82884. Concept supervision learned weak physiological
  associations but did not produce a validated AUPRC gain. Result:
  `docs/concept_rationale_distillation_result_2026-09-25.md`.

## G6Q-KD4 — final difficulty-aware distillation screen (2026-09-25)

Protocol:
[`docs/difficulty_aware_distillation_plan_2026-09-25.md`](docs/difficulty_aware_distillation_plan_2026-09-25.md).

- [x] Freeze a binary-classification adaptation of difficulty-aware KD that
  never removes records or reweights the hard-label loss.
- [x] Implement teacher-agreement, dynamic student difficulty, curriculum and
  prespecified hard-negative soft-loss weighting with bounded influence.
- [x] Retain equal-budget hard/uniform-JS controls, identical-q4 classical
  controls, clinical fusion, full OOF exports and patient-bootstrap gates.
- [x] Pass the real-data preflight with 17,348 ECGs and folds 9/10 sealed.
  **PASS:** 17,348 ECGs, 14,958 patients, folds 1–8 only, zero hard-label
  records removed.
- [x] Complete the pinned Kaggle GPU screen and apply the final KD stop rule.
  **STOP:** curriculum VQC 0.82724 versus hard VQC 0.82567 (paired delta
  +0.00155 `[+0.00032, +0.00269]`) and q4 MLP 0.82884. Curriculum fusion
  0.83697 remained below all-classical fusion 0.83802. Full result:
  `docs/difficulty_aware_distillation_result_2026-09-25.md`.
