# Adaptive non-local observable q4 experiment plan

**Status:** implementation complete; two independent seed-42 Kaggle scouts to
be submitted after local validation.

## Research question

The retained four-qubit VQC reaches 0.82938 development OOF AUPRC, while its
fusion reaches 0.83546 and the frozen all-classical system reaches 0.838015.
Increasing qubits, changing representations, distillation, extra re-uploading,
frequency scaling and tied equilibrium recurrence did not close that gap. This
experiment tests a narrower hypothesis: the retained circuit may learn useful
states whose information is lost by a fixed, low-capacity measurement.

Adaptive Non-Local Observables (ANO) parameterize a k-local Hermitian readout in
the Pauli basis instead of selecting a fixed Pauli-Z measurement. The source
paper reports improved forecasting results over fixed measurements, but those
results are not evidence of improved ECG classification. We therefore treat ANO
as a falsifiable measurement-layer hypothesis and preserve every upstream
component. Source: <https://arxiv.org/abs/2607.24399>.

The newer input-conditioned hypernetwork form is excluded from this scout. It
would add a strong classical input-to-parameter network and weaken the claim
that the quantum circuit is the core predictor. Source:
<https://arxiv.org/abs/2609.18655>.

## Frozen experimental contract

Both jobs use the same 17,348 fold-1-to-8 PRIMARY ECGs, patient groups, labels,
supervised Transformer h128 archives, fold-local PLS-q4 coordinates and
seed-42 q4 reference as the retained experiment. Folds 9 and 10 remain sealed.

The retained two-layer ring circuit, initial input bandwidth, patient-unique
balanced training sample, narrow-JS objective, AdamW schedule, 40 epochs,
train-reference CDF alignment, cross-fitted calibration, clinical teacher and
nonnegative cross-fitted fusion are unchanged. The q4 arrays must match the
retained artifact with maximum absolute error at most 1e-5.

## Separate jobs

### Job 1: ANO k=2

For every one of the six qubit pairs, learn a normalized arbitrary two-qubit
Hermitian observable from its 16-element Pauli basis. The quantum head therefore
returns six expectation values to a linear readout.

### Job 2: ANO k=3

For every one of the four three-qubit subsets, learn a normalized arbitrary
three-qubit Hermitian observable from its 64-element Pauli basis. The quantum
head returns four expectation values to a linear readout.

The jobs are independent. Their results must not be used to tune one another.

## Controls inside each job

1. **Adaptive ANO:** circuit and k-local Pauli coefficients are trainable.
2. **Fixed-Pauli ablation:** identical circuit and output subsets, with each
   measurement frozen to Z tensor Z (or Z tensor Z tensor Z).
3. **Parameter-matched q4 MLP:** same q4 input, sample, objective, optimizer and
   epoch budget, with its trainable parameter count within one percent of ANO.
4. **Frozen retained controls:** retained VQC, q4 logistic, q4 MLP ensemble,
   retained quantum fusion and matched all-classical fusion.

The experiment exports complete OOF probabilities, raw logits, q4 coordinates,
measurement outputs, normalized Pauli coefficients, loss/gradient histories,
calibration and fusion audits.

## Primary gates

ANO advances to confirmation only if every condition holds:

- paired patient-bootstrap AUPRC gain over the retained VQC is at least 0.005;
- that gain has a 95% interval above zero;
- ANO beats fixed-Pauli and parameter-matched MLP controls with intervals above
  zero;
- ANO clinical fusion exceeds 0.838015 development OOF AUPRC;
- all 17,348 records are scored once, q4 identity passes and folds 9/10 remain
  unopened.

A passing locality is then repeated with three restarts and multiple seeds. If
neither locality passes, adaptive measurement is stopped and the retained VQC
remains the quantum benchmark. There will be no post-hoc mixing of k=2 and k=3.

## Interpretation limits

This exact-statevector study evaluates predictive behavior and simulation cost.
It does not establish quantum advantage, hardware speedup or clinical utility.
Even a promoted result must later pass shot-noise, device-noise, calibration and
one-time held-out validation gates.
