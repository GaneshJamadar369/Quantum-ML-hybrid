# Label-free ECG representation plan for the quantum core

## Objective

Determine whether a label-free ECG encoder and a less destructive q4/q8 bottleneck improve VQC performance and make the quantum component a defensible predictive core. The current supervised Transformer plus supervised PLS representation is the frozen reference: Transformer head AUPRC 0.8325, q4 logistic 0.8224 and q4 VQC 0.8148.

## Scientific ordering

1. **Objective isolation — Transformer-JEPA:** retain the 271,041-parameter patch Transformer and replace MI-supervised training with masked latent prediction. This tests the representation objective without confounding it with a new backbone.
2. **Backbone isolation — Conv-S4D-JEPA:** if the label-free Transformer vectors pass the representation gate, replace attention with a shallow S4D temporal backbone while holding the self-supervised objective, output dimension and evaluation protocol fixed.
3. **Selective-state ablation — Conv-Mamba-JEPA:** run only if S4D or Transformer-JEPA shows a stable gain. Mamba is not promoted merely for being newer.
4. Keep xLSTM, Hyena and linear-attention models as later ablations. Do not allocate compute to RWKV/Griffin, Titans/TTT, Jamba/Zamba/MoE, RAG, diffusion or neural operators for this 10-second classification task.

## Phase J1 — Transformer-JEPA representation

For each held-out development fold 1–8:

- fit the existing fold-local waveform normalizer on the other seven folds;
- train an online Transformer using 60% random patch masking, mild gain/noise perturbation and no diagnostic labels;
- predict complete-waveform target tokens produced by an exponential-moving-average encoder;
- include a pooled global latent loss so the exported h128 head is trained, plus a small variance penalty to detect and discourage collapse;
- use a fixed 12-epoch schedule; neither MI labels nor outer-fold performance chooses the stopping point;
- freeze the EMA target encoder and export coherent train/validation h128 vectors;
- save patient/fold checks, finite-loss and feature-variance histories, model checksums and complete provenance.

Fold 9 remains unavailable for representation/model selection and Fold 10 remains inaccessible. Labels are retained in output artifacts only for subsequent held-out head evaluation.

### J1 acceptance gate

- 17,348 unique OOF ECGs and 14,958 patients;
- zero train/validation patient overlap in every fold;
- eight distinct finite encoder checkpoints;
- finite h128 embeddings with non-collapsed coordinate variance;
- package tests and Kaggle execution pass;
- no label tensor reaches the encoder optimizer.

## Phase J2 — q4/q8 bottleneck comparison

On each frozen fold-specific h128 representation, select the same patient-unique 500 MI and 500 non-MI training ECGs used by the earlier head screen. Fit every transformation on those training ECGs only.

Compare two bottleneck families:

| Bottleneck | Definition | Purpose |
|---|---|---|
| Unsupervised q4 | robust scaling → PCA(4) → quantile angles | direct label-free counterpart to current q4 |
| Unsupervised q8 | robust scaling → PCA(8) → quantile angles | test whether q4 removed useful interactions |

The first experiment deliberately avoids supervised PLS because PLS directly maximizes covariance with the MI label and may make the coordinates especially favorable to a linear classifier.

## Phase J3 — matched predictive heads

For q4, run the current direct data-reuploading VQC, logistic regression, parameter-matched MLP and RBF-SVC. For q8, run an eight-coordinate/four-qubit VQC that uploads coordinates 1–4, entangles, uploads 5–8, entangles again and applies a linear measurement readout. Give q8 classical controls exactly the same coordinates and training ECGs.

Primary metric is pooled OOF AUPRC. Also report AUROC, hard-negative performance, sensitivity at 90% specificity, parameter count, runtime, final loss and gradient range. Use five prespecified seeds only after the single-seed implementation screen passes.

### Decision gates

- **Representation GO:** label-free VQC exceeds the current 0.8148 AUPRC reference without evidence of collapse or leakage.
- **q8 GO:** q8 improves VQC over q4 with patient-bootstrap uncertainty and does not merely increase every classical score by the same amount.
- **Quantum win:** VQC minus the strongest identical-input classical control has a patient-cluster 95% interval entirely above zero across the repeated-seed confirmation.
- **MODIFY:** absolute VQC improves, but a matched classical model remains better.
- **STOP:** label-free VQC declines materially, representations collapse, or improvements disappear under repeated seeds.

Only after J1–J3 should Conv-S4D-JEPA be implemented. This prevents architecture, supervision and quantum-circuit changes from being mixed into one uninterpretable experiment.

## Execution status — 2026-09-22

Phase J1 completed on the private Kaggle GPU job [AQUIRE-Med Label-Free ECG JEPA](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-label-free-ecg-jepa), pinned to source revision `9fdb01a`.

- 17,348 unique OOF ECGs from 14,958 patients across folds 1–8;
- zero patient overlap in every outer fold;
- folds 9 and 10 not accessed;
- eight distinct target-encoder checksums;
- finite h128 vectors with all 128 coordinates non-collapsed and sample rank 128;
- median coordinate standard deviation 1.216 (minimum 0.751);
- masked/global training loss decreased in every fold, from 1.15–1.32 initially to 0.37–0.47 at epoch 12;
- five Kaggle representation tests passed before training.

The J1 integrity gate therefore passed. A diagnostic performed after freezing the encoders showed h128 logistic AUPRC 0.4986, unsupervised q4 logistic 0.3944 and unsupervised q8 logistic 0.4311 under the same 1,000-record-per-fold budget.

Phases J2/J3 subsequently completed on [AQUIRE-Med JEPA q4 q8 Quantum Screen](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-jepa-q4-q8-quantum-screen). q4 VQC reached 0.3865 AUPRC and q8/four-qubit VQC reached 0.3835, while the parameter-matched q8 MLP reached 0.4412. q8 improved the MLP and RBF controls but not VQC. The representation GO and q8 GO gates failed, so the branch terminates and the planned S4D/Mamba jobs are not launched. The complete analysis is in [`jepa_quantum_result_2026-09-23.md`](jepa_quantum_result_2026-09-23.md).
