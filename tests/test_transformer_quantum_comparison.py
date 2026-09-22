import json

import pandas as pd

from analyze_transformer_quantum_comparison import compare


def test_paired_encoder_comparison_aligns_ecgs_and_labels(tmp_path):
    rows = []
    for model in ("waveform_direct_vqc", "waveform_matched_mlp", "waveform_rbf_svc"):
        for ecg_id in range(1, 13):
            label = ecg_id % 2
            rows.append({
                "ecg_id": ecg_id, "patient_id": ecg_id,
                "fold": 1 + (ecg_id % 8), "label": label,
                "model": model, "score": 0.7 if label else 0.3,
            })
    frame = pd.DataFrame(rows)
    left = tmp_path / "transformer.csv"
    right = tmp_path / "cnn.csv"
    frame.to_csv(left, index=False)
    frame.to_csv(right, index=False)
    transformer_summary = tmp_path / "t.json"
    cnn_summary = tmp_path / "c.json"
    transformer_summary.write_text(json.dumps({"oof_auprc_raw": 0.8}))
    cnn_summary.write_text(json.dumps({"oof_auprc_raw": 0.7}))
    report = compare(left, right, transformer_summary, cnn_summary,
                     tmp_path / "report.json", iterations=25)
    assert report["held_out_ecgs"] == 12
    assert report["heads"]["waveform_direct_vqc"]["transformer_auprc"] == 1.0
    assert report["heads"]["waveform_direct_vqc"]["paired_transformer_minus_cnn"]["delta_auprc"]["mean"] == 0.0
