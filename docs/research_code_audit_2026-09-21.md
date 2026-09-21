# AQUIRE-Med research code audit — 2026-09-21

## Decision

**MODIFY. Do not open Fold 10 and do not claim a quantum or multimodal win.**

The preprocessing and feature-evidence work is substantially stronger than the
later prototype modeling code. The repository passes 70 local tests, but four
important tests are skipped locally (PyTorch plus three real-data tests), and
the tests that existed did not exercise the complete Kaggle quantum command.

## What failed in the latest Kaggle execution

The latest eight-fold QSVM calculation completed in roughly three minutes. It
then crashed while printing a non-existent metric field:

```text
AttributeError: 'ModelMetrics' object has no attribute 'sens_at_90_spec'
```

The correct field is `sensitivity_at_90_specificity`. The same stale names also
included `f1_optimal`, `expected_calibration_error`, and `hard_neg_auroc`.
Before the crash, the log reported **QSVM OOF AUPRC 0.3199 and AUROC 0.5987**.
These are log-only results because the script saved artifacts only after every
model completed.

The preceding 12-hour failure was a different issue: a nested Python loop made
millions of circuit calls and reached only QSVM fold 4 before Kaggle stopped the
job. Both failures were preventable with a preflight test and fold checkpoints.

## Scientific and implementation findings

### Critical

1. **The old quantum kernel was not the claimed entangled feature map.** It
   applied an angle encoding followed by the same fixed CNOT ring to every
   sample. A shared data-independent unitary cancels from the fidelity
   `|<psi(x)|psi(y)>|²`; the CNOT layers therefore changed no kernel value.
2. **The QSVM did not consume the “top eight SHAP features.”** It sliced the
   first eight manifest columns: heart rate, RR summaries, and early lead-I
   measurements. The artifact label `quantum_top8_multimodal` was false.
3. **The reported multimodal fusion ignored the tabular input.** Attention had
   one query and one key. Softmax over one key is always one, so the attention
   output was independent of the query. The reported hybrid AUPRC of 0.7973
   cannot be interpreted as a tabular-plus-waveform fusion result.
4. **The holdout workflow violated the frozen protocol.** It was configured to
   run automatically after QML, combined Folds 9 and 10 in its description,
   retrained four candidate models, and compared them on Fold 10. Fold 9 is the
   calibration/validation set; Fold 10 is a one-time final test after one model
   and all decisions are frozen.
5. **The development HDF5 contains Folds 1–8 only.** The old holdout script was
   pointed at that file, so it could not supply Fold 9 or Fold 10 waveforms.

### High priority

1. The downloaded classical artifact identifies
   `hist_gradient_boosting` as champion at **0.719802 AUPRC / 0.872210 AUROC**.
   The narrative reports instead claim XGBoost at 0.7251 / 0.8754. Reports must
   be generated from immutable artifacts rather than handwritten values.
2. The classical plan promised an RBF-SVC, but `_models()` contains no RBF-SVC.
3. The standalone ResNet result (**0.791308 AUPRC**) remains a useful
   development result, but it is one seed without paired patient-bootstrap
   confidence intervals. It is a provisional benchmark, not state of the art.
4. The 0.0060 AUPRC difference between the broken hybrid and ResNet is small and
   has no uncertainty estimate. Even with a correct fusion block it would need
   paired patient-cluster bootstrap testing and repeated seeds.
5. Quantum installation pulled SHAP into the environment, which downgraded
   NumPy and broke the declared lock. SHAP is unrelated to QML execution.
6. `lightning.qubit` is an analytic CPU simulator. A Kaggle GPU allocation does
   not make QSVM quantum execution run on GPU, and it is not quantum hardware.
7. QSVM used `SVC(probability=True)` and then applied another Platt calibrator
   across outer OOF predictions. This was double calibration. The outer OOF
   calibrator was also not a strict nested calibration design.
8. The script loaded about 794 MB of waveforms even for tabular-only QSVM.
9. No fold checkpoint was written, so every failure discarded completed work.
10. The current conformal report calibrates and reports coverage on Fold 8
    itself and mixes OOF models trained on overlapping folds. It is exploratory,
    not a valid independent coverage claim. Final calibration belongs on Fold 9
    and coverage evaluation on Fold 10.
11. SHAP is an exploratory audit on Fold 8 and its global CSV averages several
    model rankings. It is not proof of causal or clinical validity and must not
    be used globally to select outer-fold predictors.

## Corrections implemented in this audit

- Replaced the ineffective quantum feature map with a data-dependent IQP map.
- Added kernel symmetry, diagonal, PSD, and conditioning diagnostics.
- Added train-only, internally cross-fitted QSVM probability calibration.
- Defined `z8` fold-locally using training-only median imputation, robust
  scaling, and PCA whitening.
- Added an RBF-SVC control with the identical `z8`, class balance, sample budget,
  folds, and calibration design.
- Split QML execution by model and made QSVM plus the matched control the first
  gate; VQC/HQNN are no longer launched automatically.
- Avoided loading waveform HDF5 for the QSVM stage.
- Added atomic per-fold checkpoints and resume support.
- Added a quantum end-to-end preflight test before real data is touched.
- Removed automatic Fold 9/10 evaluation from the Kaggle quantum runner.
- Removed SHAP/XGBoost from the quantum environment and pinned PennyLane.
- Replaced the one-key fusion with two-token multimodal attention and added
  regression tests proving that each modality affects the prediction.

## Research-grounded direction

The strongest performance route is a validated ECG encoder, then a carefully
tested fusion layer. PTB-XL benchmarking found ResNet/Inception-family waveform
models stronger than feature-only approaches and used the official
patient-respecting splits at 100 Hz. A modern extension is self-supervised ECG
pretraining, evaluated as an ablation rather than assumed to help.

The quantum branch is a falsifiable representation experiment:

```text
training-only feature processing -> z8
                                 -> matched RBF-SVC
                                 -> IQP quantum kernel SVC
                                 -> paired OOF comparison
```

Only if the IQP kernel improves pooled OOF AUPRC with a patient-bootstrap
confidence interval above zero should stronger matched-kernel controls be run.
VQC or a quantum residual model should proceed only after those controls and
kernel-conditioning checks pass. Simulation accuracy alone cannot establish
computational quantum advantage.
The practical QML literature requires comparison with strong classical models
and classical approximations of the quantum model.

Phase 6Q-B completed this immediate control gate. Once all models received the
QSVM's train-fitted angle coordinates, Laplacian, RBF and product-cosine SVCs
outperformed IQP-QSVM. The QML performance branch therefore stops here as a
valid negative result; the repeated-seed waveform encoder and repaired fusion
are the next performance experiments.

## Acceptance sequence

1. Run the new Phase 6Q-A preflight and eight-fold QSVM/RBF benchmark.
2. Report paired patient-bootstrap delta AUPRC, delta Brier, kernel diagnostics,
   runtime, and memory. Call a positive result a matched-kernel accuracy delta;
   do not call it quantum advantage. It requires the 95% CI of delta AUPRC to
   be above zero and no clinically material subgroup regression.
3. Rerun corrected fusion with a ResNet-only control, tabular-only control,
   modality-shuffle tests, at least three seeds, and paired patient bootstrap.
4. Add xResNet1D101/InceptionTime or a pretrained ECG encoder as the strong
   waveform reference. This is the most credible route to higher performance.
5. Freeze one architecture and its hyperparameters. Use Fold 9 once for
   probability calibration and threshold selection.
6. Preprocess Fold 10 with the frozen pipeline, evaluate the one frozen model
   once, and publish confidence intervals plus failure/subgroup analysis.

## Primary references

- [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
- [PTB-XL+ documentation](https://physionet.org/content/ptb-xl-plus/1.0.1/)
- [Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL](https://arxiv.org/abs/2004.13701)
- [Self-Supervised Pre-Training with JEPA Boosts ECG Classification](https://arxiv.org/abs/2410.13867)
- [Power of data in quantum machine learning](https://www.nature.com/articles/s41467-021-22539-9)
- [The Inductive Bias of Quantum Kernels](https://arxiv.org/abs/2106.03747)
- [Systematic review of QML for digital health](https://pmc.ncbi.nlm.nih.gov/articles/PMC12048600/)

The 2025 systematic review screened 4,915 records and found no consistent trend
supporting empirical QML utility over classical methods in digital health. Our
project should therefore report “quantum utility not demonstrated” unless the
matched, uncertainty-aware experiment rejects that null position.
