# G6Q-R2 — Clinically structured four-qubit re-uploading

## Research question

Does the current supervised Transformer-to-PLS `q4` bottleneck discard
conditional information about T-wave inversion, ST burden, aVR/reciprocity and
R-wave progression, and can a four-qubit VQC use those preserved concepts
better than classical models receiving the identical coordinates?

This is a bounded representation experiment. It does not claim clinical or
computational quantum advantage. Folds 9 and 10 remain inaccessible.

## Frozen cohort and representation

- 17,348 primary PTB-XL development ECGs from official folds 1–8.
- Fold-coherent supervised Transformer `h128` and fold-local PLS `q4`.
- Four disjoint clinical feature families frozen in
  `configs/structured_reuploading_v1.json`.
- Each family is median-imputed, robust-scaled and compressed to one PCA
  coordinate using outer-training records only. PCA does not receive MI labels.
- PCA orientation is fixed by a prespecified anchor feature, followed by a
  training-only quantile angle transform to `[-pi/2, pi/2]`.
- The resulting quantum input is
  `z8 = [q_wave1..q_wave4, c_T, c_ST, c_aVR, c_Rprogression]`.

## Stage A — conditional-information gate

For every outer fold, compare fixed HistGradientBoosting models on `q4` versus
`q4 + c4`. Logistic regression is a secondary linear check. Also report each
single concept family as a descriptive ablation. All transforms and models are
fit only on the outer-training records.

The quantum screen runs only when the pooled OOF HistGradientBoosting delta is
at least `+0.005 AUPRC` and its 2,000-replicate paired patient-cluster bootstrap
95% interval lies above zero. The four concept families are never selected or
edited from outer-fold results.

## Stage B — matched predictive-head screen

The four-qubit structured VQC uploads the waveform block with `RY` rotations,
applies a shallow trainable/Ising-ZZ layer, uploads the clinical block through
non-commuting `RZ` and `RX` rotations, and applies the second shallow layer.
The readout sees only local `Z` and ring `ZZ` expectations.

Prespecified arms:

1. current waveform `q4` curriculum VQC;
2. structured `z8` hard-label VQC;
3. structured `z8` curriculum VQC;
4. structured `z8` curriculum VQC with entanglement disabled;
5. concept-only `c4` curriculum VQC;
6. identical-`z8` logistic regression, parameter-matched MLP and RBF-SVC;
7. clinical-expert plus quantum and all-classical cross-fitted fusion.

Quantum fits use the existing patient-unique, class-balanced sample budget,
30 epochs, one frozen seed and the same hard-label/curriculum schedule as the
latest q4 study. No five-seed follow-up is permitted unless the single-seed
gate passes.

## Promotion gates

- **Representation:** structured VQC minus current q4 VQC is at least `+0.005`
  AUPRC with a positive patient-bootstrap interval.
- **Quantum:** structured VQC beats the strongest identical-`z8` classical
  head with a positive interval.
- **System:** clinical-plus-structured-VQC fusion beats clinical-plus-`z8`-MLP
  fusion with a positive interval and exceeds the recorded all-classical
  reference AUPRC `0.83802`.
- **Stability:** complete eight-fold coverage, finite gradients and no patient
  overlap. Folds 9 and 10 must be reported as unaccessed.

If Stage A fails, stop before circuit training. If Stage A passes but any Stage
B promotion gate fails, retain the VQC as a required demonstrator and report
the corresponding classical model as the scientific performance ceiling.

## Build order

- [x] Freeze clinical groups and gates.
- [x] Implement and unit-test the fold-local concept encoder.
- [x] Implement and unit-test the exact statevector structured re-uploading VQC.
- [x] Implement Stage A and gated Stage B driver with complete OOF artifacts.
- [x] Run local real-data preflight and Stage A.
- [x] Apply the Kaggle submission gate. **Not submitted:** Stage A failed the
  frozen effect-size threshold, so GPU execution was prohibited by protocol.
- [x] Verify coverage and write the final verdict.
