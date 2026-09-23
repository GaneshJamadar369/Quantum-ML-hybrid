# q4 representation on a 16-qubit VQC — capacity-isolation plan

## Objective

Test whether the four-coordinate representation is already appropriate but the
four-qubit VQC lacks circuit capacity. This deliberately retains the successful
fold-local Transformer h128 → supervised PLS q4 → quantile-angle transformation
and changes only the quantum register from four to sixteen qubits.

This differs from the completed q16 experiment, which changed both the
representation and circuit:

| Experiment | Input | Quantum register |
|---|---|---|
| Completed direct-width q16 | PLS q16 | 16 qubits, one coordinate per qubit |
| Capacity-isolation experiment | **PLS q4** | **16 qubits, each coordinate encoded on four wires** |

## Circuit

The wire assignment is
`[q1,q2,q3,q4,q1,q2,q3,q4,q1,q2,q3,q4,q1,q2,q3,q4]`. The repeated classical
angles are legal data encoding; no unknown quantum state is cloned. Two layers
apply RY data encoding, near-identity trainable Rot gates and ring IsingZZ
interactions. Sixteen Z and sixteen adjacent ZZ expectations feed a linear
readout. The circuit has 177 trainable parameters and is evaluated with the
PennyLane-verified exact GPU statevector backend.

## Frozen comparison

- same eight patient-separated development folds;
- same patient-unique 500 MI and 500 non-MI training ECGs per fold;
- same q4 transformation, 20 epochs and fixed seed;
- q4 logistic, q4 RBF-SVC and a 175-parameter MLP receive identical inputs;
- existing 45-parameter four-qubit VQC AUPRC 0.8148 is the promotion reference;
- h128 logistic 0.8247 remains the strongest frozen-representation control;
- folds 9 and 10 remain sealed.

## Decision gate

Promote the 16-qubit core only if it exceeds the four-qubit VQC by at least
0.005 AUPRC with a patient-cluster 95% interval above zero, and does not lose to
the strongest identical-q4 classical control. Otherwise retain four qubits and
record circuit widening as a negative capacity result.

## Status

- [x] Replicated-input exact 16-qubit VQC implemented.
- [x] Sixteen-wire mapping, gradient and PennyLane-equivalence tests pass.
- [x] [Private Kaggle eight-fold job](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-q4-on-16-qubit-vqc) completed. **16-qubit VQC AUPRC 0.7979.**
- [x] Paired comparison with the existing four-qubit VQC completed. **ΔAUPRC -0.0168 [-0.0233, -0.0105].**
- [x] Promotion or stop decision applied. **STOP: retain the four-qubit VQC at 0.8148.**

The complete analysis is in
[`q4_on_16_qubit_capacity_result_2026-09-23.md`](q4_on_16_qubit_capacity_result_2026-09-23.md).
