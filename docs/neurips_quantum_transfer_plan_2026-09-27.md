# G6Q-NR — controlled transfer of NeurIPS quantum ideas

**Protocol date:** 2026-09-27  
**Status:** implementation complete; two one-seed scouts ready for parallel submission  
**Development data:** PTB-XL folds 1–8 only; folds 9 and 10 remain sealed

## Question

Can either layer-specific input frequencies or a weight-tied recurrent quantum
cell improve the retained supervised Transformer → fold-local PLS-q4 → VQC
path without changing its patients, labels, compression, loss, calibration or
clinical fusion?

This is a circuit-head experiment. It is not an ECG foundation-model
experiment and does not claim that PLS coordinates have anatomical meaning.

## Evidence boundary

The retained paired seed-42 reference is the score-alignment experiment:

- VQC AUPRC: `0.829379`;
- quantum/clinical fusion AUPRC: approximately `0.83546`;
- all-classical ceiling: `0.838015`.

Trainable scaling and repeated q4 upload already exist in the retained VQC.
The new scaling arm therefore tests **layer-specific** frequency and phase
parameters. Applying named anatomical edges to PLS-q4 is forbidden because
the four PLS components do not correspond to fixed lead territories.

## Parallel scout A — layer-specific quantum frequencies

The same q4 coordinates enter both circuit layers. Each layer learns its own
bounded scale and phase for every coordinate:

\[
\phi_{l,j}=\lambda_{l,j}q_j+b_{l,j},\qquad
\lambda_{l,j}\in[0.25,2.0].
\]

Everything else remains the retained two-layer four-qubit ring circuit with
local Z and neighbouring ZZ observables and a linear readout.

## Parallel scout B — tied recurrent quantum cell

The scout uses a finite QDEQ-inspired refinement, rather than claiming a full
implicit deep-equilibrium implementation. At every step a fresh q4 circuit
uploads the fixed q4 vector followed by the previous local-Z memory. The same
rotations and interactions are reused at every step:

\[
m_{t+1}=(1-\alpha)m_t+\alpha Z(F_\theta(q,m_t)).
\]

The scout fixes three refinement steps and `alpha=0.5`. It saves held-out
residual distributions. Repeated refinement is not free physical depth; every
step is another circuit execution.

## Frozen common protocol

- One prespecified seed: `42`.
- Eight official outer folds, with patient-unique training samples.
- `2,000` records per class, one restart and 40 epochs for the scout.
- Existing `narrow_js` objective, cosine schedule and train-reference CDF
  alignment.
- Identical fold-local PLS-q4 coordinates, checked numerically against the
  retained reference.
- Cross-fitted calibration and cross-fitted one-neuron clinical fusion.
- Paired patient-cluster bootstrap with 2,000 replicates.
- Full raw logits, q4 coordinates, quantum observables, learned scales,
  gradients and convergence residuals saved.

The one-restart/40-epoch scout is deliberately cheaper than the retained
three-restart/60-epoch confirmation. A scout may advance only after a positive
paired result; confirmation then repeats the exact architecture at the full
budget and multiple seeds.

## Promotion gates

Advance an arm only if all conditions hold:

1. Candidate VQC improves seed-matched retained VQC by at least `0.005`
   AUPRC and its paired patient-bootstrap interval is above zero.
2. Candidate beats retained identical-q4 logistic and MLP controls.
3. Candidate fusion exceeds `0.838015` and beats the retained quantum fusion.
4. Sensitivity at 90% specificity does not fall by more than `0.005`, Brier
   does not worsen by more than `0.002`, and hard-negative FPR does not rise by
   more than `0.01`.
5. The equilibrium scout additionally needs evidence of contraction. A full
   implicit QDEQ implementation is blocked until a separate solver/gradient
   gate demonstrates at least 99.5% convergence, residual at most `1e-4`, and
   agreement with long unrolling.

If neither scout passes, stop both circuit ideas and do not merge them.

## Deferred work

- An anatomy-aware circuit first requires a new fold-local representation
  with fixed territory identities and a classical information gate. It must
  include no-entanglement and scrambled-graph controls.
- Hadamard/Lie-algebra gradient estimation begins only after a predictive
  circuit is frozen. Its outcome is shot and wall-time efficiency, not AUPRC.
- ECG foundation pretraining is a separate encoder lane. It must first beat
  the supervised Transformer representation under the same folds before any
  quantum-head evaluation.
