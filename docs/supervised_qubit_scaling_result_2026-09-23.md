# Supervised Transformer qubit-scaling result — 2026-09-23

## Outcome

**STOP direct qubit scaling at q16. Retain q4/four-qubit VQC as the quantum
champion.** Increasing both the supervised PLS width and directly encoded
qubits did not recover predictive information. All wider VQCs remained below
the q4 VQC, and every wider width lost to its identical-input logistic control.

Completed private Kaggle jobs:

- [q8/eight-qubit screen](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-supervised-q8-direct-vqc)
- [q12/twelve-qubit screen](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-supervised-q12-direct-vqc)
- [q16/sixteen-qubit screen](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-supervised-q16-direct-vqc)

Each job used 17,348 outer-fold predictions from 14,958 patients, the same
patient-unique 500 MI and 500 non-MI training records per fold, and folds 1–8
only. Folds 9 and 10 remained sealed. The q16 run completed all eight folds in
approximately 37 minutes after Kaggle's alternate input mount was resolved.

## Cross-width results

| Representation/head | Qubits | Parameters | OOF AUPRC | OOF AUROC |
|---|---:|---:|---:|---:|
| h128 logistic | 0 | — | **0.8247** | **0.9195** |
| q4 logistic | 0 | 5 | 0.8224 | — |
| **q4 VQC** | **4** | 45 | **0.8148** | 0.9172 |
| q8 logistic | 0 | 9 | 0.8133 | 0.9159 |
| q12 logistic | 0 | 13 | 0.8079 | 0.9135 |
| q16 logistic | 0 | 17 | 0.8025 | 0.9113 |
| q12 VQC | 12 | 133 | 0.7939 | 0.9088 |
| q16 VQC | 16 | 177 | 0.7805 | 0.9024 |
| q8 VQC | 8 | 89 | 0.7532 | 0.8913 |

The classical sequence is informative: q4 logistic 0.8224, q8 logistic 0.8133,
q12 logistic 0.8079 and q16 logistic 0.8025. Extra PLS coordinates did not add
useful held-out discrimination; they progressively reduced it. The result is
therefore not simply that the quantum circuit lacked enough qubits.

## Paired patient-cluster comparisons

- q8 VQC minus q4 VQC: ΔAUPRC **-0.0615**, 95% CI [-0.0693, -0.0534].
- q12 VQC minus q4 VQC: **-0.0208** [-0.0272, -0.0147].
- q16 VQC minus q4 VQC: **-0.0342** [-0.0415, -0.0265].
- q12 VQC minus q8 VQC: +0.0407 [+0.0323, +0.0495].
- q16 VQC minus q12 VQC: -0.0133 [-0.0200, -0.0062].

At q16, VQC minus logistic was -0.0220 [-0.0290, -0.0155]. VQC minus the
parameter-matched MLP was +0.0017 [-0.0064, +0.0101], and VQC minus RBF-SVC
was +0.0081 [-0.0001, +0.0162]. Neither nonlinear-control interval established
a win, and logistic remained clearly stronger.

## Execution audit

The exact PyTorch statevector implementation matches PennyLane to machine
precision on the same circuit. All eight q16 folds trained 1,000 unique
patients, produced finite predictions and had nonzero gradients. q16 gradient
norms ranged from 0.269 to 0.466 and training loss decreased in every fold.
The negative result is therefore not explained by a halted optimizer, barren
gradient trace, missing fold or patient leakage.

## Decision

The prespecified width gate required a wider VQC to exceed q4 by at least
0.005 AUPRC with a patient-bootstrap interval above zero. q8, q12 and q16 all
failed decisively. Do not launch the conditional full-h128 16-qubit
eight-upload circuit or amplitude-encoding branch under this representation
recipe. Their rationale depended on a positive width trend that is absent.

The retained prototype remains supervised Transformer h128 → fold-local q4 →
four-qubit VQC. h128 and q4 logistic remain mandatory scientific controls, and
the evidence still does not establish predictive or computational quantum
advantage.
