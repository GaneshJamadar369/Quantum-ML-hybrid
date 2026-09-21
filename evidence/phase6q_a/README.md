# Phase 6Q-A evidence bundle

These files were downloaded from the completed Kaggle CPU run on 2026-09-21.
`quantum_oof_predictions.csv` contains the patient-safe fold 1–8 predictions
used for the paired 2,000-resample patient-cluster bootstrap. `SHA256SUMS.txt`
records the downloaded evidence hashes.

The raw bootstrap JSON contains the original label `PASS_QML_UTILITY`. That
label was too broad and is retained only to preserve the immutable run output.
The code and report now call the result
`PASS_MATCHED_KERNEL_ACCURACY_DELTA`, because comparison with one RBF kernel on
a classical simulator cannot establish computational quantum advantage.

See [`docs/phase6q_a_matched_kernel_result.md`](../../docs/phase6q_a_matched_kernel_result.md)
for the reviewed interpretation and next gate.
