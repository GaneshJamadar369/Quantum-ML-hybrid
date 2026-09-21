# Direct quantum-core screening checkpoint — 2026-09-22

## Execution and data contract

Both private Kaggle jobs completed on repository commit `c2e79a5`:

| Task | Kaggle job | Local evidence |
|---|---|---|
| Clinical-feature q4 VQC | [Phase 6Q-D](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-phase-6q-d-direct-vqc) | `kaggle_outputs/quantum_core_20260922/quantum-core-q4-screen/` |
| Fold-coherent waveform q4 VQC | [Phase 6Q-E](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-phase-6q-e-waveform-vqc) | `kaggle_outputs/waveform_quantum_20260922/waveform-quantum-q4-screen/` |

Both used PTB-XL development folds 1–8 only, 17,348 held-out ECGs from
14,958 patients, 4,368 MI-positive ECGs (25.18%). Each outer training fold
sampled up to 500 patient-unique ECGs per class; fold-local scaling, supervised
PLS to four coordinates, and quantile-to-angle mapping were fitted on those
training samples. Fold 9 and fold 10 remained sealed. All eight prediction
files are finite, with one held-out score per ECG and no cross-fold patient
overlap. The waveform branch used the training and validation embeddings from
the **same** fold-specific encoder, fixing the previous incompatible-coordinate
pooling problem.

The direct classifier uses four qubits, two angle-data re-uploading layers,
trainable bounded feature scales, a sparse ring of Ising-ZZ entanglers,
single-qubit Z and edge ZZ observables, and a linear readout (45 trainable
parameters). The simulator is `lightning.qubit` on CPU, not a QPU. Scores are
uncalibrated sigmoid transforms of logits/margins, not clinical probabilities.

## Eight-fold out-of-fold results

| Input and head | AUPRC | AUROC |
|---|---:|---:|
| Clinical q4 RBF-SVC | **0.5133** | 0.7763 |
| Clinical q4 direct VQC | 0.5087 | 0.7705 |
| Clinical q4 small MLP | 0.5047 | **0.7778** |
| Waveform q4 small MLP | **0.7583** | **0.8675** |
| Waveform q4 direct VQC | 0.7325 | 0.8671 |
| Waveform q4 RBF-SVC | 0.7246 | 0.8614 |

The paired 2,000-resample patient-cluster bootstrap yielded:

- Clinical VQC minus RBF: delta AUPRC -0.0046, 95% interval
  [-0.0169, 0.0081]. VQC minus MLP: +0.0041 [-0.0048, 0.0134].
  Neither establishes a win.
- Waveform VQC minus RBF: +0.0080 [0.0005, 0.0159]. This is a narrow,
  exploratory win against **one** matched input/control on this simulator.
- Waveform VQC minus MLP: -0.0256 [-0.0349, -0.0155]. The best tested
  classical head wins decisively. This screen does not support an overall
  quantum predictive advantage.

The clinical VQC training loss decreased from about 0.68–0.72 to 0.54–0.58;
all reported gradient norms were finite and nonzero. Waveform VQC training
loss reached about 0.018–0.065, yet held-out AUPRC remained below the MLP.
This large training/validation contrast requires a larger-sample and
regularization study before circuit expansion. The waveform encoder itself is
**supervised on MI labels**; its 128-dimensional output is useful as a
representation ablation but cannot by itself establish that the quantum model
is the indispensable core source of MI discrimination. Its full 128-dimensional
supervised encoder OOF AUPRC was 0.7878, above every q4 head here.

The controls share the same q4 inputs, outer folds and per-class sample budget.
Their parameter counts, optimization and inductive biases are not identical;
`matched_mlp` means matched input and data, not exact parameter matching. The
clinical and waveform branches share the held-out cohort but may select
different training patients because their source row orders differ. Their
cross-branch AUPRC difference is therefore a representation signal, not a
causal estimate of quantum benefit. Bootstrap intervals do not account for
the exploratory choice among architectures or training seeds.

## Decision and next experiment

**MODIFY.** Keep the q4 direct VQC as a valid trainable baseline, not the
champion. Do not open folds 9/10 or report its scores as calibrated patient
risk. Do not claim quantum advantage. The current strongest tested q4 head is
the waveform MLP; the prior full-waveform ResNet remains a separate stronger
benchmark, subject to repeated-seed validation.

1. Freeze this screen as exploratory. Formal G0–G5 sign-off, unresolved
   feature-evidence recommendations, corrected fusion, and repeated-seed
   waveform baselines remain open in `PLAN.md`.
2. Build a fold-local **label-free** ECG representation arm (masked-waveform
   reconstruction or contrastive learning), with a supervised encoder kept as
   an explicitly labeled ablation. Train on outer-training patients only and
   export same-encoder training/validation coordinates with immutable hashes.
   Compare clinical-only, waveform-only and concatenated inputs.
3. Fit angle mappings and any supervised PLS **only inside the inner training
   split**. Predefine a compact search over q4/q6/q8, ring versus ladder,
   one versus two re-uploading layers, and a few bandwidth initializations.
   Include an independently tuned RBF/Laplacian SVC, parameter-count-matched
   MLP, logistic head, and an ablation that removes the circuit but keeps the
   readout. Use inner patient folds for choices; preserve outer-fold OOF for
   assessment. Do not tune on pooled OOF labels.
4. First test more patient-unique training samples (e.g. 500 versus 1,500 per
   class where available) and early stopping/weight decay on inner validation.
   Use at least five prespecified seeds for the finalists. Inspect train/val
   learning curves, gradient norms, angle saturation, prediction variance,
   subgroup errors, simulation runtime and circuit/shot costs. Advance to q8
   only if q4/q6 show stable learning and a credible gain over all controls.
5. Freeze a quantum candidate only if it exceeds the **best** matched
   classical head on the same representation with a patient-bootstrap delta
   AUPRC interval above zero across prespecified seeds, without a material
   hard-negative, calibration or latency regression. Then calibrate on fold 9
   and evaluate fold 10 exactly once under the frozen protocol. If it fails,
   retain QML as an honest demonstration while documenting the classical
   performance leader; the prototype may still run a quantum predictive core
   to satisfy the problem-statement deliverable, with the gap disclosed.

No QPU run, noise robustness result, external validation, probability
calibration or clinical utility claim follows from these two screens.
