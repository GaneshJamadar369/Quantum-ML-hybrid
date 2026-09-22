# Transformer-input classical and quantum comparison

## Frozen experiment

The source is the completed patient-separated compact Transformer export on PTB-XL development folds 1–8. Its 128-dimensional supervised ECG embeddings are already MI-discriminative. We will score each outer fold only after its encoder was trained on the other seven folds. Folds 9 and 10 remain sealed.

Each head uses the same patient-unique 500 MI and 500 non-MI training ECGs per fold. Four-coordinate comparisons share the same fold-local robust scaling, supervised PLS and quantile-angle mapping. This gives a matched input and sample-budget test, not a claim that quantum processing discovered ECG morphology independently. The 128-coordinate classical arm uses the *same 1,000 ECGs* to measure information lost at the q4 bottleneck. It is a separate capacity comparison and cannot be presented as matched quantum evidence.

| Arm | Models | Input |
|---|---|---|
| Matched q4 | VQC, IQP-QSVC, small MLP, RBF-SVC, Laplacian-SVC, logistic regression, shallow histogram boosting | Four PLS/quantile angles |
| Wider classical | Logistic regression, RBF-SVC, shallow histogram boosting | Full 128-dimensional embedding |
| Fusion sensitivity | HQNN and matched MLP | Two waveform angles plus two locally extracted clinical-feature angles |

VQC/MLP/RBF results already exist in `docs/transformer_quantum_result_2026-09-22.md`. Three private Kaggle jobs completed the new classical comparison, IQP-QSVC and fusion HQNN. The fusion result is shown separately because the input contains additional clinical measurements.

Pre-specified single-seed settings avoid learning from outer-fold scores. Model choice remains provisional until repeated seeds, training-only hyperparameter searches, circuit-removal controls and a prospective fold-9 calibration pass are complete. Primary metric is patient-pooled OOF AUPRC; secondary AUROC and paired patient-cluster bootstrap differences. Scores are not calibrated probabilities. No quantum-advantage or clinical-deployment claim follows from this screen.

## Completed jobs and validation

- [Classical q4/h128 comparison](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-transformer-classical-comparison)
- [Transformer IQP-QSVC screen](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-transformer-qsvc-screen)
- [Transformer fusion HQNN screen](https://www.kaggle.com/code/swayamjeetbhagat4/aquire-med-transformer-hqnn-screen)

All jobs used pinned source revision `5202e07`, 17,348 unique held-out ECGs from 14,958 patients, and outer folds 1–8. Each fold trained its comparison heads on 1,000 patient-unique ECGs. Training and validation patient overlap was zero; folds 9 and 10 were not accessed. The classical and QSVC audits have identical sampled-training ECG hashes, validation hashes, and q4 representation contracts in every fold. Quantum test suites passed before the quantum jobs ran. Outputs were finite and complete.

## Results

The primary matched q4 comparison is:

| Model | Type | Input | OOF AUPRC | OOF AUROC |
|---|---|---:|---:|---:|
| Logistic regression | Classical | q4 | **0.8224** | **0.9195** |
| Direct VQC | Quantum | q4 | 0.8148 | 0.9172 |
| Small MLP | Classical | q4 | 0.8112 | 0.9158 |
| Histogram boosting | Classical | q4 | 0.8038 | 0.9139 |
| Median-bandwidth RBF-SVC | Classical | q4 | 0.7974 | 0.9123 |
| Fixed-gamma RBF-SVC | Classical | q4 | 0.7747 | 0.9035 |
| Laplacian-SVC | Classical | q4 | 0.7343 | 0.8951 |
| IQP-QSVC | Quantum kernel | q4 | 0.3190 | 0.5959 |

The direct VQC minus q4 logistic-regression delta AUPRC was **-0.0075**, with a 95% patient-cluster bootstrap interval of **[-0.0125, -0.0030]**. The interval is entirely below zero, so the best quantum model did not win this matched comparison. The VQC did beat the matched small MLP by 0.0036 in the earlier screen, but its interval crossed zero. The IQP kernel was PSD in every sampled audit, yet it used 815–889 of 1,000 training records as support vectors and generalized poorly. This is an empirical failure of this feature map and bandwidth, rather than an invalid-kernel failure.

The same-budget wider classical ceiling is:

| Model | Input | OOF AUPRC | OOF AUROC |
|---|---:|---:|---:|
| Logistic regression | h128 | **0.8247** | **0.9195** |
| Histogram boosting | h128 | 0.8196 | 0.9171 |
| RBF-SVC | h128 | 0.7934 | 0.9116 |

The q4 versus h128 logistic delta AUPRC was -0.0023 [-0.0061, 0.0015]. Thus q4 retained nearly all of the head-level discrimination available to logistic regression on h128. This comparison is not a matched quantum test because the input capacities differ. The Transformer's own supervised encoder head remains higher at AUPRC 0.8325, although it used the full outer-training set and therefore also has a different training budget.

The separate two-waveform-plus-two-clinical-coordinate fusion result is:

| Model | Input | OOF AUPRC | OOF AUROC |
|---|---:|---:|---:|
| Fusion MLP | 2 waveform + 2 clinical | **0.8164** | **0.9188** |
| Fusion HQNN | 2 waveform + 2 clinical | 0.8049 | 0.9152 |
| Waveform-only RBF control | 2 waveform | 0.7334 | 0.8940 |

HQNN minus identical-input fusion MLP delta AUPRC was **-0.0114 [-0.0167, -0.0061]**. Quantum gradients were finite and nonzero, and loss decreased in every fold, so this is a trained negative result rather than a halted or inert circuit.

## Decision

**MODIFY.** Keep the compact Transformer as the current best representation, but do not select a quantum champion from this single-seed screen. The ordering is: supervised Transformer head (0.8325), h128 logistic (0.8247), q4 logistic (0.8224), q4 VQC (0.8148), fusion HQNN (0.8049), and IQP-QSVC (0.3190), with the stated differences in inputs and sample budgets. The current evidence supports the Transformer and the compact q4 bottleneck; it does not support quantum superiority.

For the problem-statement-compliant prototype, retain the VQC as the leading quantum candidate while labeling it the **quantum-core demonstration model**. Before freezing it, run the prespecified repeated-seed, inner-fold tuning and label-free encoder experiments. Classical q4 logistic regression must remain the principal matched scientific control. Stop the current IQP-QSVC design and do not spend fold 9 or fold 10 on it.
