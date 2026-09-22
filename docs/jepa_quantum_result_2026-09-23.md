# Label-free ECG-JEPA q4/q8 quantum result — 2026-09-23

## Outcome

**STOP this label-free branch; retain the supervised Transformer q4 VQC as the quantum reference.** The label-free encoder passed every integrity and non-collapse check, but its representations contained substantially less MI-discriminative information than the supervised Transformer representation. Increasing the label-free bottleneck from q4 to q8 improved classical heads but did not improve the VQC.

Completed private Kaggle jobs:

- [Label-free Transformer-JEPA representation](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-label-free-ecg-jepa)
- [JEPA q4/q8 matched head screen](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-jepa-q4-q8-quantum-screen)

The representation job was pinned to revision `9fdb01a`; the head job was pinned to revision `faa80ea`. It covered 17,348 unique OOF ECGs from 14,958 patients over development folds 1–8. Each head used the same patient-unique 500 MI and 500 non-MI training ECGs per outer fold. Patient overlap was zero, all outputs were finite, and folds 9 and 10 remained sealed. Eleven quantum/bottleneck tests passed before the head job ran.

## Representation audit

All eight label-free encoders had distinct checksums. The exported h128 matrix was finite and full-rank; coordinate standard deviations ranged from 0.751 to 1.884. Training loss decreased in every fold, from 1.15–1.32 initially to 0.37–0.47 after 12 epochs. Thus the negative predictive result cannot be explained by obvious representation collapse, a halted encoder or patient leakage.

The bottleneck transformation was fold-local robust scaling, whitened PCA and quantile-to-angle mapping. It never received labels. PCA q4 retained 0.679–0.788 of training-sample variance across folds; q8 retained 0.839–0.900.

## Matched results

| Representation and head | Parameters | OOF AUPRC | OOF AUROC |
|---|---:|---:|---:|
| Label-free q8 MLP | 51 | **0.4412** | 0.7198 |
| Label-free q8 RBF-SVC | — | 0.4383 | **0.7198** |
| Label-free q8 logistic | 9 | 0.4311 | 0.7169 |
| Label-free q4 RBF-SVC | — | 0.4273 | 0.7018 |
| Label-free q4 MLP | 25 | 0.4212 | 0.6985 |
| Label-free q4 logistic | 5 | 0.3944 | 0.6878 |
| Label-free q4 VQC | 45 | 0.3865 | 0.6605 |
| Label-free q8/four-qubit VQC | 49 | 0.3835 | 0.6608 |

Both VQCs trained for 20 epochs in every fold. Loss decreased and gradient norms were finite and nonzero. The q8 VQC used four qubits: coordinates 1–4 were uploaded and entangled, followed by coordinates 5–8 and a second entangling stage.

Paired patient-cluster results:

- q4 VQC minus q4 MLP: ΔAUPRC -0.0349, 95% CI [-0.0467, -0.0230].
- q4 VQC minus q4 logistic: -0.0081 [-0.0211, 0.0059].
- q8 VQC minus parameter-matched q8 MLP: **-0.0580 [-0.0717, -0.0443]**.
- q8 VQC minus q8 logistic: -0.0480 [-0.0618, -0.0336].
- q8 VQC minus q4 VQC: -0.0030 [-0.0159, 0.0091].
- q8 MLP minus q4 MLP: **+0.0202 [0.0090, 0.0316]**.
- q8 RBF-SVC minus q4 RBF-SVC: +0.0109 [0.0005, 0.0215].

The q8 coordinates therefore carried useful additional information that classical nonlinear heads exploited. The four-qubit re-uploading VQC did not exploit that information. This separates a pure dimensionality failure from a circuit/head failure.

## Comparison with the supervised reference

| System | OOF AUPRC |
|---|---:|
| Supervised Transformer classification head | **0.8325** |
| Supervised h128 logistic | 0.8247 |
| Supervised PLS-q4 logistic | 0.8224 |
| Supervised PLS-q4 VQC | **0.8148** |
| Label-free h128 logistic diagnostic | 0.4986 |
| Label-free PCA-q8 MLP | 0.4412 |
| Label-free PCA-q4 VQC | 0.3865 |
| Label-free PCA-q8/four-qubit VQC | 0.3835 |

Label-free q4 VQC minus supervised q4 VQC was -0.4276 [-0.4440, -0.4109]. Label-free q8 VQC minus supervised q4 VQC was -0.4308 [-0.4475, -0.4131]. At 90% specificity, sensitivity was 0.234 for label-free q4 VQC and 0.226 for q8 VQC, versus approximately 0.758 for the supervised q4 VQC.

## Interpretation and decision

This experiment tests PTB-XL-only, 12-epoch label-free pretraining with a compact Transformer. It does not disprove ECG-JEPA trained on much larger external ECG corpora. It does show that our present label-free recipe is unsuitable for the MI prototype and that merely replacing attention with S4D or Mamba under the same weak pretraining signal is not justified.

Do not launch the gated Conv-S4D-JEPA or Conv-Mamba-JEPA jobs. Keep the supervised Transformer representation and direct q4 VQC as the leading problem-statement-compliant quantum demonstration, while retaining q4 logistic as the stronger matched scientific control. The next quantum work, if pursued, should focus on repeated-seed/nested optimization of the supervised representation, bottleneck and shallow VQC—not on adding another sequence backbone.
