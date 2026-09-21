# QSVC and fusion HQNN screening result — 2026-09-22

## Reproducible execution

Two private Kaggle jobs completed from pinned source revision `10f280d`:

| Experiment | Kaggle result | Local downloaded artifact |
|---|---|---|
| 4-qubit IQP-QSVC | [Phase 6Q-F](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-phase-6q-f-waveform-qsvc) | `kaggle_outputs/qsvc_20260922/qsvc-screen/` |
| 4-qubit fusion HQNN | [Phase 6Q-G](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-phase-6q-g-fusion-hqnn) | `kaggle_outputs/hqnn_20260922/hqnn-screen/` |

Both jobs used PTB-XL development folds 1–8, 17,348 held-out ECGs from
14,958 patients, and the same saved fold-coherent waveform encoders. Each
outer training fold selected 500 patient-unique MI ECGs and 500 patient-unique
non-MI ECGs, including hard negatives. There was no train/validation patient
overlap, non-primary ECG entry, or fold 9/10 access. The trained representation
for each held-out fold came from its matching outer-fold encoder. Source ECG
and patient IDs were cross-checked against immutable preprocessing metadata.
Each job wrote eight finite fold checkpoints, OOF scores, metrics, patient
bootstrap comparisons and per-fold audits. The source encoder was trained
with MI labels, so these are **supervised-representation ablations**, not
clean evidence that quantum processing alone found the MI pattern.

## QSVC: numerically sound kernel, weak predictive geometry

Fold-local robust scaling, supervised PLS and quantile angle mapping produced
the same four-coordinate waveform input for all three kernels. The IQP
feature map used four qubits and two repeats; SVC used `C=1` and balanced
class weights. RBF and Laplacian bandwidths were chosen from training-only
median distances. The IQP kernel was a valid state-fidelity Gram matrix;
128-by-128 per-fold PSD audits found no negative eigenvalues. All scores below
are **uncalibrated** and use pooled eight-fold OOF evaluation.

| Head | AUPRC | AUROC |
|---|---:|---:|
| IQP-QSVC | 0.6572 | 0.8096 |
| RBF-SVC | **0.7433** | 0.8676 |
| Laplacian-SVC | 0.7244 | **0.8737** |

Patient-cluster bootstrap, 2,000 resamples: IQP-QSVC minus RBF AUPRC
`-0.0862 [-0.0976, -0.0746]`; IQP-QSVC minus Laplacian
`-0.0673 [-0.0791, -0.0553]`. Both are clear losses. In a diagnostic on
outer fold 1, mean same-class minus different-class kernel similarity was
about `0.040` for IQP versus `0.313` for RBF, consistent with weaker class
separation. The fixed IQP map should be kept as a negative control; a later
bandwidth/topology search needs inner-fold selection and cannot use these OOF
labels for tuning. This result reinforces the earlier negative q8 kernel
branch, but the data representation differs from that earlier experiment.

## HQNN: trainable and modestly stronger than waveform RBF, below fusion MLP

The waveform and approved deployable clinical feature matrices were separately
reduced within each outer training fold to two supervised PLS angles each. A
trainable four-input mixer fed a direct four-qubit, two-layer quantum head
with a linear readout: 65 trainable parameters in total. The classical fusion
MLP consumed the same four angles and used the same training sample budget,
epochs and optimizer; it had 45 parameters. A waveform-only RBF-SVC was an
additional, weaker control and did **not** receive clinical features.

| Head | AUPRC | AUROC |
|---|---:|---:|
| Fusion HQNN | 0.7424 | **0.8716** |
| Fusion MLP | **0.7526** | 0.8644 |
| Waveform-only RBF-SVC | 0.7245 | 0.8529 |

Patient-cluster bootstrap: HQNN minus fusion MLP AUPRC
`-0.0101 [-0.0181, -0.0018]`; HQNN minus waveform RBF
`+0.0178 [0.0094, 0.0262]`. The first is the relevant same-modality
comparison: **HQNN did not win**. Every fold's training loss decreased to
roughly `0.012–0.046`, and gradient norms stayed finite and nonzero. This
demonstrates software trainability, while the near-zero fit loss from 1,000
patients per fold warns of overfitting. A paired, exploratory cross-run
comparison with the previous waveform-only direct VQC gave HQNN minus VQC
`+0.0098 [0.0007, 0.0186]`; the models do not share identical inputs and
the CI does not account for trying several candidate designs.

## Decision

**Neither QSVC nor HQNN is the development champion.** The IQP-QSVC loses
substantially to both matched classical kernels. HQNN is viable as a working
quantum predictive head but loses to the same-input fusion MLP on primary
AUPRC. The prior four-coordinate waveform MLP also scored 0.7583, and the
full 128-dimensional supervised ECG encoder scored 0.7878 in its own OOF
screen; those are distinct-capacity comparisons, not proofs about quantum
utility. The original full-waveform ResNet remains a separate benchmark
requiring repeated-seed validation.

The two new jobs ran on the CPU `lightning.qubit` statevector simulator, not
quantum hardware. Execution took approximately 174 seconds for QSVC and
1,376 seconds for HQNN, including setup and bootstrap. No output here is a
calibrated MI probability, prospective risk score, external-validation
result, or quantum-advantage claim. The HQNN classical control is input- and
training-matched but not exactly parameter-count-matched. These were
prespecified one-seed screens; repeated seeds and inner-fold circuit search
are still needed before deciding whether to retain a quantum model beyond
the problem-statement demonstration.

The next best use of compute is **representation and generalization repair**:
expand patient-unique training samples, apply inner-fold early stopping and
regularization, test label-free ECG representations alongside the supervised
encoder, and compare q4/q6 circuit variants with tuned classical heads on
identical fold-local coordinates. Do not deepen the current IQP or HQNN
circuits simply because they execute successfully. Preserve folds 9 and 10
until a single architecture and calibration protocol are frozen.
