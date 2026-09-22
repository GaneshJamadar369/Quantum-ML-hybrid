"""Paired patient-level CNN/Transformer q4 head comparison.

This runs only after both independent eight-fold Kaggle jobs finish.  It does
not choose architecture hyperparameters or touch sealed folds 9/10.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _json_default


def _one_model(frame: pd.DataFrame, model: str) -> pd.DataFrame:
    selected = frame[frame.model.eq(model)].copy()
    if selected.empty or not selected.ecg_id.is_unique:
        raise ValueError(f"Missing or duplicate predictions for {model}")
    return selected.set_index("ecg_id").sort_index()


def compare(
    transformer_predictions: Path,
    cnn_predictions: Path,
    transformer_representation_summary: Path,
    cnn_representation_summary: Path,
    output_path: Path,
    iterations: int = 2000,
    seed: int = 20260922,
) -> dict:
    transformer = pd.read_csv(transformer_predictions)
    cnn = pd.read_csv(cnn_predictions)
    model_names = ("waveform_direct_vqc", "waveform_matched_mlp", "waveform_rbf_svc")
    results = {}
    for model in model_names:
        left = _one_model(transformer, model)
        right = _one_model(cnn, model)
        if not left.index.equals(right.index):
            raise ValueError(f"Different held-out ECG IDs for {model}")
        for column in ("patient_id", "fold", "label"):
            if not np.array_equal(left[column].to_numpy(), right[column].to_numpy()):
                raise ValueError(f"Mismatched {column} for {model}")
        if not np.isfinite(left.score).all() or not np.isfinite(right.score).all():
            raise ValueError("Non-finite prediction score")
        paired = _paired_patient_bootstrap(
            left.label.to_numpy(), left.patient_id.to_numpy(),
            left.score.to_numpy(), right.score.to_numpy(),
            iterations=iterations, seed=seed,
        )
        results[model] = {
            "transformer_auprc": float(average_precision_score(left.label, left.score)),
            "cnn_auprc": float(average_precision_score(right.label, right.score)),
            "paired_transformer_minus_cnn": paired,
        }
    transformer_summary = json.loads(transformer_representation_summary.read_text())
    cnn_summary = json.loads(cnn_representation_summary.read_text())
    result = {
        "patients": int(left.patient_id.nunique()),
        "held_out_ecgs": int(len(left)),
        "folds": sorted(int(value) for value in left.fold.unique()),
        "transformer_encoder_oof_auprc": transformer_summary["oof_auprc_raw"],
        "cnn_encoder_oof_auprc": cnn_summary["oof_auprc_raw"],
        "heads": results,
        "interpretation_boundary": (
            "Same held-out patients and head protocols, but independently trained supervised "
            "encoders. Improvements shared by classical and quantum heads are representation "
            "benefits, not quantum advantage. Scores are uncalibrated."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, default=_json_default))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-predictions", type=Path, required=True)
    parser.add_argument("--cnn-predictions", type=Path, required=True)
    parser.add_argument("--transformer-summary", type=Path, required=True)
    parser.add_argument("--cnn-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.transformer_predictions, args.cnn_predictions,
        args.transformer_summary, args.cnn_summary, args.output,
    )
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
