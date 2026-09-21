# AQUIRE-Med research completion roadmap

**Checkpoint date:** 2026-09-21

## Scientific scope

The primary task is contemporaneous **MI-pattern versus non-MI-pattern
classification** from a submitted 10-second, 12-lead ECG. It is not future
cardiovascular-event prediction, a definitive clinical diagnosis, or autonomous
triage. The primary cohort includes overlapping diagnoses; the prespecified
hard-negative cohort contains abnormal non-MI ECGs that can mimic infarction.

The goal is a reproducible, externally challenged research model. “Best” means
best under this exact phenotype, patient partition, operating point and
uncertainty protocol. Scores reported for different label sets in other papers
are references, not directly comparable leaderboard entries.

## Truthful checkpoint

| Component | Status | Evidence | Decision |
|---|---|---|---|
| PTB-XL development preprocessing | Executed | 17,348 primary ECGs and 70 quarantined ECGs; 100 Hz waveforms and provenance artifacts | Retain; complete formal G0–G5 sign-off |
| Deployable features | Executed | 106 approved and 20 excluded features in signed v0.4 manifest | Retain as deployable tabular branch |
| Classical feature models | Executed, provisional | HistGradientBoosting pooled OOF AUPRC 0.7198 | Retain as strong tabular comparator; rerun under final protocol |
| ECGResNet1D | Executed once | Pooled OOF AUPRC 0.7913, one seed | Promising, not a frozen champion |
| Original multimodal fusion | Executed but invalid as fusion evidence | AUPRC 0.7973; one-key attention ignored the tabular query | Discard result; corrected architecture must be rerun |
| IQP-QSVM | Fully executed | Eight-fold OOF plus patient bootstrap | Negative result against stronger matched kernels |
| VQC | Prototype only | No real-data OOF run; batched backpropagation was broken before the 2026-09-21 repair | Smoke-testable now; scientifically unvalidated |
| HQNN | Prototype only | No real-data OOF run; same prior gradient defect; waveform stem is not the validated ResNet champion | Smoke-testable now; scientifically unvalidated |
| Fold 9 | Sealed calibration set | Not used by valid model-selection jobs | Keep sealed until one champion is frozen |
| Fold 10 | Sealed final test | Not used | Open exactly once after Fold-9 calibration |
| External validation | Not done | No independent labeled cohort result | Required before strong generalization claims |

## Why VQC and HQNN were halted

They were excluded from the Phase 6Q jobs by design after the matched-kernel
gate failed. Running slower quantum models after IQP-QSVM lost to Laplacian,
RBF and product-cosine controls would not be a rational performance search.

The code audit found an additional implementation problem: the PennyLane
`parameter-shift` path could execute a batched forward pass but failed during
backpropagation through broadcasted trainable inputs. The implementation now
uses adjoint differentiation on `lightning.qubit` and backpropagation on
`default.qubit`; batched VQC and HQNN forward/backward regression tests pass.
This proves software trainability only. It does not provide medical performance,
scalability, noise robustness or quantum-utility evidence.

VQC and HQNN remain in the study as bounded ablations. They will run only after
the classical waveform representation is frozen, against parameter- and
data-matched classical heads. They cannot become the clinical champion unless
they pass the same uncertainty, subgroup, calibration and latency gates.

## Final model-development protocol

### R0 — Reconcile data gates

- Sign G0–G5 using the immutable Kaggle artifacts and checksums.
- Recompute patient counts, label prevalence and fold isolation from artifacts.
- Resolve every remaining feature-manifest recommendation.
- Keep folds 9 and 10 inaccessible to all development commands.

**Pass:** zero patient overlap, zero unresolved waveform paths, verified
PTB-XL v1.0.3 identity, signed phenotype and feature manifests.

### R1 — Repair the deep-learning experiment engine

- Use lazy HDF5 loading instead of copying the complete waveform tensor to RAM.
- Fit tabular imputation and robust scaling inside each outer training fold.
- Apply ECG normalization and augmentation inside training folds only.
- Add AMP, fold checkpoints, restart support and deterministic seed logging.
- Save raw logits, model checkpoints, optimizer settings and learning curves.
- Do not call development probabilities clinically calibrated.

**Pass:** package/Kaggle parity, restart test, modality-dependence tests and one
128-record end-to-end fixture.

### R2 — Strong supervised waveform baselines

Run the following under identical folds, augmentations and stopping rules:

1. Existing compact ECGResNet1D as the continuity control.
2. xResNet1D101 or the official PTB-XL residual benchmark.
3. InceptionTime/Inception1D as a different temporal inductive bias.
4. A compact structured state-space or transformer baseline only if its public
   implementation and preprocessing can be reproduced.

Use a development-only two-stage budget: one screening seed for all candidates,
then five prespecified seeds for the two finalists. Hyperparameters are selected
within folds 1–8; no Fold-9 inspection.

### R3 — Pretrained ECG representation track

Audit weights, license, input sampling rate, normalization and pretraining-data
overlap before use. Prioritize reproducible public checkpoints:

1. ECG-JEPA ViT-XS/ViT-S: frozen linear probe, then discriminative fine-tuning.
2. ECGFounder or HeartLang: include only after checkpoint and license validation.
3. CPC/ST-MEM reference if integration cost is lower than the larger models.

For each encoder, compare random initialization, frozen probe and fine-tuning.
Pretraining on PTB-XL or PTB-derived data must be disclosed; such a model cannot
serve as clean external validation on an overlapping source.

### R4 — Correct multimodal fusion

Compare these models with identical waveform encoders:

- waveform-only;
- tabular-only;
- late concatenation;
- corrected two-token gated attention;
- fusion with shuffled tabular values;
- fusion with shuffled waveforms;
- fusion with missing/noisy modality masks.

Promote fusion only when patient-bootstrap delta AUPRC versus waveform-only is
greater than 0.005 with a 95% interval above zero, hard-negative FPR does not
worsen materially, and both modalities measurably influence predictions.

### R5 — Bounded quantum ablation

Freeze the best classical ECG encoder. Compare on the same frozen `z4`/`z8`
or residual representation:

- logistic/linear head;
- small MLP with matched trainable-parameter count;
- RBF and Laplacian heads;
- VQC with 4 qubits before 8 qubits;
- frozen-encoder HQNN quantum bottleneck;
- optional residual prediction `y - p_classical`, with matched classical
  residual controls.

Use at least five seeds for trainable heads, report gradient norms and failure
rates, and include ideal/noisy-simulator sensitivity. A real QPU execution is a
demonstration only unless it is repeated and statistically compared under a
matched time/shot budget.

**Stop rule:** if VQC/HQNN fails to exceed the best matched classical head with
the patient-bootstrap 95% delta-AUPRC interval above zero, record a negative
result and keep the classical champion for inference.

### R6 — Development selection

Primary metric: patient-pooled OOF AUPRC. Also report AUROC, sensitivity at 90%
and 95% specificity, specificity at fixed sensitivity, F1, log loss, Brier,
calibration error, decision-curve net benefit, latency and memory.

Every comparison uses patient-cluster bootstrap intervals. Report performance
by sex, age band, hard-negative diagnosis group, annotation certainty, QC group
and missing-lead/noise condition. Select the simplest model within 0.005 AUPRC
of the leader when uncertainty intervals overlap materially.

### R7 — Freeze, calibrate and test

1. Freeze architecture, preprocessing, weights-training recipe and threshold
   policy from folds 1–8.
2. Retrain the frozen recipe on folds 1–8 with prespecified seeds/ensemble.
3. Use Fold 9 once for natural-prevalence calibration, abstention thresholds
   and operating-point selection.
4. Lock all code and hashes.
5. Evaluate Fold 10 once; compute patient-bootstrap intervals and all subgroup,
   calibration, robustness and failure analyses.

Fold 10 cannot choose models, thresholds, ensembles or explanation settings.

### R8 — External and stress validation

- Select at least one independently sourced 12-lead ECG cohort with a defensible
  MI label mapping and documented license.
- Have a clinician review the mapping and a stratified error sample.
- Report transport without retuning, then a clearly separated recalibration
  analysis if allowed.
- Stress test baseline drift, muscle noise, lead dropout, lead swaps, polarity
  inversion, amplitude scaling, resampling and device/site shift.

### R9 — Interpretability, uncertainty and release

- Use lead/time occlusion and integrated gradients; require explanation
  stability across seeds and small input perturbations.
- Compare highlighted segments with QRS/ST/T regions and clinician review.
- Implement abstention for low confidence, failed QC and out-of-distribution
  embeddings.
- Publish model/data cards, intended-use statement, failure cases, environment
  lock, hashes and a one-command reproduction entrypoint.

## Exact execution order

1. Reconcile G0–G5 evidence and repair the deep runner.
2. Rerun corrected ECGResNet1D and corrected fusion over five seeds on GPU.
3. Add xResNet/Inception and screen public pretrained ECG encoders.
4. Run full ablations and choose one development champion.
5. Execute the bounded VQC/HQNN head study on frozen embeddings.
6. Freeze the final recipe.
7. Calibrate on Fold 9.
8. Evaluate Fold 10 once.
9. Perform external and robustness validation.
10. Complete clinician error review and release the reproducibility package.

## Completion definition

The project is research-complete only when the frozen model has a one-time
Fold-10 result, external-cohort result, patient-bootstrap uncertainty,
clinically meaningful operating points, subgroup and hard-negative analyses,
robustness tests, calibrated/abstaining output, reproducible code and an honest
quantum conclusion. A high development AUPRC alone is insufficient.

## Research anchors

- [PTB-XL deep-learning benchmark](https://arxiv.org/abs/2004.13701)
- [ECG-JEPA](https://arxiv.org/abs/2410.13867)
- [ECGFounder official implementation](https://github.com/PKUDigitalHealth/ECGFounder)
- [HeartLang official implementation](https://github.com/PKUDigitalHealth/HeartLang)
- [Systematic review of QML for digital health](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048600/)
