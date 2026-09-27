# G6Q-DM — nonlinear manifold q4 representation screen

**Decision under test:** can an inductive Diffusion Map preserve useful local
geometry from the fold-coherent Transformer `h128` representation better than
the retained supervised PLS-q4 compressor, while keeping the quantum model and
all downstream evaluation unchanged?

## Frozen architecture

```text
PTB-XL 12x1000 -> existing QC/preprocessing -> frozen outer-fold Transformer h128
  -> outer-train-only landmark Diffusion Map q4 -> train-only angle quantiles
  -> existing 4-qubit, 2-layer ring VQC with Z/ZZ observables and JS training
  -> train-reference CDF score -> cross-fitted calibration -> P(MI)
  -> existing cross-fitted clinical fusion
```

The PLS-q4 VQC result from seed 42 is the paired reference. No waveform
preprocessing, Transformer weight, label definition, circuit, observable,
optimizer, distillation target, sampling budget, calibration, or fusion rule is
changed.

## Fold and leakage contract

- Use development folds 1–8 only. Folds 9 and 10 remain inaccessible.
- For each outer fold, fit robust scaling, patient-unique landmarks, kernel
  bandwidth, anisotropic graph normalization, eigenvectors, and angle quantiles
  from that outer-training partition only.
- Embed outer-validation ECGs only through the Nystrom extension.
- Landmark selection is label-free and includes at most one ECG per patient.
- Preserve the eight separately trained Transformer encoders. Never pool
  embeddings from different fitted encoders before outer-fold prediction.

## Frozen representation configuration

- 1,024 patient-unique landmarks
- RBF affinity with bandwidth equal to the median 15-nearest-landmark squared
  distance
- 32-neighbour symmetric landmark graph
- anisotropic normalization `alpha=1`
- four leading non-stationary, one-step diffusion coordinates
- outer-training quantile mapping to `[-pi/2, pi/2]`

## Controls and evidence

- Identical-q4 balanced logistic regression and three-restart MLP
- VQC with entanglement removed
- Paired PLS-q4 VQC and PLS-q4 fusion predictions from the retained run
- Same 2,000-per-class patient-unique quantum training sample
- Same three VQC restarts, 60 epochs, clinical teacher, train-CDF alignment,
  cross-fitted calibration, and nonnegative logistic fusion
- Patient-cluster bootstrap with 2,000 replicates
- Geometry audit: effective rank, pairwise-distance Spearman correlation, and
  15-neighbour overlap between h128 and q4

## Promotion gates

All gates must pass before any extra seeds or generative diffusion work:

1. **Representation:** Diffusion-Map VQC exceeds paired PLS-q4 VQC by at least
   0.003 AUPRC and the patient-bootstrap interval is above zero.
2. **Quantum:** Diffusion-Map VQC exceeds its strongest identical-q4 classical
   control by at least 0.003 AUPRC and the interval is above zero.
3. **System:** quantum fusion exceeds the frozen all-classical ceiling AUPRC
   `0.8380154`.

Failure of any gate stops this branch. Passing all gates promotes exactly two
additional prespecified seeds before fold 9 can be considered. A generative ECG
diffusion encoder, imputer, denoiser, or augmentation model is deferred because
it changes more than the compression bottleneck and is substantially more
expensive to validate.

