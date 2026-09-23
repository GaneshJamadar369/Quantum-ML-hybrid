# q4 representation on a 16-qubit VQC — result

## Outcome

**STOP and retain the four-qubit VQC.** Replicating the winning q4 coordinates
across a sixteen-qubit entangling register reduced OOF performance. The wider
core did not overcome the current model.

Private Kaggle job:
[AQUIRE-Med q4 on 16-Qubit VQC](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-q4-on-16-qubit-vqc).

## Results

| Model on the frozen Transformer representation | Qubits | Parameters | OOF AUPRC | OOF AUROC |
|---|---:|---:|---:|---:|
| h128 logistic | 0 | — | **0.8247** | **0.9195** |
| q4 logistic | 0 | 5 | 0.8224 | 0.9195 |
| q4 matched MLP | 0 | 175 | 0.8167 | 0.9176 |
| **Existing q4/four-qubit VQC** | **4** | **45** | **0.8148** | **0.9172** |
| q4/sixteen-qubit VQC | 16 | 177 | 0.7979 | 0.9103 |
| q4 RBF-SVC | 0 | — | 0.7747 | 0.9035 |

The sixteen-qubit circuit encoded
`[q1,q2,q3,q4]` four times around the ring, used two RY/Rot/IsingZZ layers and
measured sixteen Z plus sixteen adjacent ZZ observables before a linear
readout.

## Paired patient-cluster evidence

- sixteen-qubit VQC minus four-qubit VQC: ΔAUPRC **-0.0168**, 95% CI
  [-0.0233, -0.0105];
- sixteen-qubit VQC minus q4 logistic: **-0.0244** [-0.0303, -0.0189];
- sixteen-qubit VQC minus 175-parameter q4 MLP: **-0.0187**
  [-0.0240, -0.0134];
- sixteen-qubit VQC minus q4 RBF-SVC: +0.0230 [+0.0147, +0.0313].

The model beats RBF-SVC, but RBF is not the strongest identical-input control.
It loses decisively to logistic, the parameter-matched MLP and the existing
four-qubit VQC.

## Execution audit

All 17,348 OOF ECGs and 14,958 patients were covered across folds 1–8. Every
fold trained on 1,000 unique patients. Folds 9 and 10 remained sealed. Loss
decreased in every fold, gradients were finite and nonzero, and all outputs
were finite. The 16-wire assignment was recorded in every fold audit. The
negative result is therefore not attributable to a stalled optimizer,
incorrect input mapping, incomplete folds or patient leakage.

## Interpretation

The q4 representation is not being limited by the number of physical
statevector qubits in this circuit family. Repeating four angles over sixteen
wires adds parameters and correlated quantum observables without adding new
patient information. It increases variance and optimization burden, reducing
held-out generalization.

The prototype should retain supervised Transformer h128 → fold-local PLS q4 →
four-qubit VQC. Further gains should come from repeated-seed/nested tuning of
the compact q4 circuit and representation, not a larger register. This remains
the best tested quantum path, while q4/h128 logistic and the Transformer head
remain stronger classical controls.
