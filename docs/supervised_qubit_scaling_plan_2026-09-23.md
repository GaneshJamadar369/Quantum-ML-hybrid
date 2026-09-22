# Supervised Transformer qubit-scaling plan — 2026-09-23

## Question

The successful quantum path currently compresses each fold-specific Transformer
embedding from h128 to supervised PLS q4 before the four-qubit VQC. Its OOF
AUPRC is 0.8148, compared with 0.8224 for q4 logistic and 0.8247 for h128
logistic. This study tests whether that 128→4 bottleneck removed interactions a
wider directly encoded quantum circuit can use.

Increasing the qubit count while retaining the same four or eight coordinates
would not restore information. The experiment therefore increases the
fold-local representation width and physical simulator qubits together.

## Frozen first-stage experiment

Three independent private Kaggle jobs evaluate:

| Branch | Transformation | Quantum circuit | State size |
|---|---|---|---:|
| q8 | h128 → supervised PLS(8) → quantile angles | 8 qubits, two re-upload layers | 256 |
| q12 | h128 → supervised PLS(12) → quantile angles | 12 qubits, two re-upload layers | 4,096 |
| q16 | h128 → supervised PLS(16) → quantile angles | 16 qubits, two re-upload layers | 65,536 |

Every branch uses the frozen supervised Transformer representations, the same
patient-unique 500 MI and 500 non-MI records per outer fold, folds 1–8 only, 20
epochs and the prespecified random seed. Fold 9 remains unavailable and Fold 10
remains sealed.

The circuit uses RY input encoding, near-identity trainable Rot gates, sparse
ring IsingZZ interactions, local Z/ZZ measurements and a linear readout. The
8–16-qubit circuit is evaluated with an exact differentiable PyTorch
statevector simulator so the 16-qubit screen can use a Kaggle GPU. A numerical
test requires the simulator to match PennyLane to machine precision on the
same circuit before cloud submission.

Each width is compared against logistic regression, RBF-SVC and a
parameter-matched one-hidden-layer MLP receiving exactly the same coordinates.
The h128 logistic reference is regenerated on the same patients. Scores remain
uncalibrated until a champion is frozen.

## Metrics and gates

Primary metric is patient-pooled OOF AUPRC. Secondary outputs are AUROC,
sensitivity at 90% specificity, hard-negative AUPRC/AUROC, gradient range,
loss trajectory, parameter count and runtime. VQC differences against every
matched control use a paired patient-cluster bootstrap.

- **Width GO:** the wider VQC improves over q4 VQC by at least 0.005 AUPRC and
  the patient-bootstrap interval is above zero.
- **Quantum GO:** VQC minus the strongest identical-input classical control has
  a 95% interval above zero.
- **Stability GO:** every fold has finite nonzero gradients and no material
  train/validation failure; a promising width must then survive five seeds.
- **STOP:** no ordered q8→q12→q16 improvement, widening helps classical heads
  only, or simulation cost grows without predictive gain.

No claim of computational quantum advantage is permitted from simulator
results.

## Conditional second stage

Only if the first-stage width gate passes:

1. feed all h128 coordinates through a 16-qubit circuit in eight sequential
   16-coordinate uploads;
2. compare with full-h128 amplitude encoding on seven qubits, which preserves
   128 normalized amplitudes but has costly state preparation;
3. run depth/topology ablations inside inner folds and confirm the selected
   configuration over five seeds;
4. assess analytic, shot-based and noise-aware inference after freezing the
   circuit.

This ordering distinguishes information loss from qubit capacity and prevents
a large full-h128 circuit from being promoted merely because it is larger.

## Implementation status

- [x] Exact GPU-vectorized 8–16-qubit statevector VQC implemented.
- [x] PennyLane equivalence, gradient, metric and parameter-match tests pass.
- [x] End-to-end q8 one-fold smoke execution passes with atomic output.
- [ ] [q8 private Kaggle OOF job](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-supervised-q8-direct-vqc) completed and audited. **Submitted and running.**
- [ ] [q12 private Kaggle OOF job](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-supervised-q12-direct-vqc) completed and audited. **Submitted and running.**
- [ ] q16 private Kaggle OOF job completed and audited. **Runner validated and pinned; submission waits for one of Kaggle's two GPU-session slots.**
- [ ] Cross-width paired comparison completed.
- [ ] Five-seed confirmation or stop decision applied.
