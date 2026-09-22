# Transformer-input classical and quantum comparison

## Frozen experiment

The source is the completed patient-separated compact Transformer export on PTB-XL development folds 1–8. Its 128-dimensional supervised ECG embeddings are already MI-discriminative. We will score each outer fold only after its encoder was trained on the other seven folds. Folds 9 and 10 remain sealed.

Each head uses the same patient-unique 500 MI and 500 non-MI training ECGs per fold. Four-coordinate comparisons share the same fold-local robust scaling, supervised PLS and quantile-angle mapping. This gives a matched input and sample-budget test, not a claim that quantum processing discovered ECG morphology independently. The 128-coordinate classical arm uses the *same 1,000 ECGs* to measure information lost at the q4 bottleneck. It is a separate capacity comparison and cannot be presented as matched quantum evidence.

| Arm | Models | Input |
|---|---|---|
| Matched q4 | VQC, IQP-QSVC, small MLP, RBF-SVC, Laplacian-SVC, logistic regression, shallow histogram boosting | Four PLS/quantile angles |
| Wider classical | Logistic regression, RBF-SVC, shallow histogram boosting | Full 128-dimensional embedding |
| Fusion sensitivity | HQNN and matched MLP | Two waveform angles plus two locally extracted clinical-feature angles |

VQC/MLP/RBF results already exist in `docs/transformer_quantum_result_2026-09-22.md`. The next three private Kaggle jobs run the new classical comparison, IQP-QSVC and fusion HQNN. The fusion result must be shown separately because the input contains additional clinical measurements.

Pre-specified single-seed settings avoid learning from outer-fold scores. Model choice remains provisional until repeated seeds, training-only hyperparameter searches, circuit-removal controls and a prospective fold-9 calibration pass are complete. Primary metric is patient-pooled OOF AUPRC; secondary AUROC and paired patient-cluster bootstrap differences. Scores are not calibrated probabilities. No quantum-advantage or clinical-deployment claim follows from this screen.
