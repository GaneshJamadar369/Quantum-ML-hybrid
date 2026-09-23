"""Audit complementarity and generate feature-routing hypotheses.

This fast study uses already completed OOF predictions.  It is intentionally
separate from the final nested routing experiment: pooled OOF associations are
used to formulate hypotheses, never to fit a held-out production model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from aquire_preprocessing.feature_routing import (
    binary_log_loss_per_record,
    route_feature_hypotheses,
)


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict:
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "auprc": float(average_precision_score(labels, p)),
        "auroc": float(roc_auc_score(labels, p)),
        "brier": float(brier_score_loss(labels, p)),
        "log_loss": float(log_loss(labels, p)),
    }


def _load_quantum(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "model" in frame:
        frame = frame[frame.model.eq("waveform_direct_vqc")]
    score = "score" if "score" in frame else "waveform_direct_vqc"
    return frame.rename(columns={score: "q4_vqc", "record_id": "ecg_id"})[
        ["ecg_id", "patient_id", "fold", "label", "q4_vqc"]
    ]


def _load_transformer(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        return pd.DataFrame({
            "ecg_id": data["record_ids"],
            "transformer": data["raw_probability"],
        })


def _load_tabular(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame.model.eq("hist_gradient_boosting")]
    return frame[["ecg_id", "probability"]].rename(columns={"probability": "clinical_histgb"})


def _diagnostic_stack(frame: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[list[float]]]:
    """Leave-one-official-fold-out meta diagnostic, not final nested evidence."""

    labels = frame.label.to_numpy(dtype=int)
    folds = frame.fold.to_numpy(dtype=int)
    matrix = np.column_stack([
        logit(np.clip(frame[column].to_numpy(float), 1e-5, 1.0 - 1e-5))
        for column in columns
    ])
    prediction = np.full(len(frame), np.nan)
    coefficients = []
    for held_out in range(1, 9):
        train, validation = folds != held_out, folds == held_out
        model = LogisticRegression(C=1.0, max_iter=2000)
        model.fit(matrix[train], labels[train])
        prediction[validation] = model.predict_proba(matrix[validation])[:, 1]
        coefficients.append(model.coef_[0].tolist())
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete diagnostic stack")
    return prediction, coefficients


def main() -> None:
    parser = argparse.ArgumentParser(description="AQUIRE-Med feature routing audit")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--quantum-oof", type=Path, required=True)
    parser.add_argument("--transformer-oof", type=Path, required=True)
    parser.add_argument("--tabular-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    predictions = _load_quantum(args.quantum_oof)
    predictions = predictions.merge(_load_transformer(args.transformer_oof), on="ecg_id", validate="one_to_one")
    predictions = predictions.merge(_load_tabular(args.tabular_oof), on="ecg_id", validate="one_to_one")
    if not predictions.ecg_id.is_unique or set(predictions.fold.unique()) != set(range(1, 9)):
        raise ValueError("Expected one OOF prediction per ECG across folds 1-8")

    feature_frame = pd.read_csv(args.features)
    merged = predictions.merge(feature_frame, on="ecg_id", how="left", validate="one_to_one")
    manifest = json.loads(args.manifest.read_text())
    evidence = pd.read_csv(args.evidence)
    feature_columns = [name for name in manifest["approved_features"] if name in merged]

    routing = route_feature_hypotheses(
        merged[feature_columns],
        merged.label,
        merged.fold,
        merged.transformer,
        merged.q4_vqc,
        approved_features=manifest["approved_features"],
        excluded_features=manifest.get("excluded_features", {}),
        statistical_evidence=evidence,
    )
    routing.to_csv(args.output / "feature_routing_hypotheses.csv", index=False)

    labels = merged.label.to_numpy(dtype=int)
    model_metrics = []
    for name in ("clinical_histgb", "transformer", "q4_vqc"):
        model_metrics.append({"model": name, **_metrics(labels, merged[name])})
    stack_specs = {
        "transformer_plus_clinical": ["transformer", "clinical_histgb"],
        "transformer_plus_quantum": ["transformer", "q4_vqc"],
        "clinical_plus_quantum": ["clinical_histgb", "q4_vqc"],
        "all_three": ["transformer", "clinical_histgb", "q4_vqc"],
    }
    stack_audit = {}
    stack_predictions = {}
    for name, columns in stack_specs.items():
        probability, coefficients = _diagnostic_stack(merged, columns)
        stack_predictions[name] = probability
        model_metrics.append({"model": name, **_metrics(labels, probability)})
        stack_audit[name] = {
            "columns": columns,
            "coefficient_mean": np.mean(coefficients, axis=0).tolist(),
            "coefficient_min": np.min(coefficients, axis=0).tolist(),
            "coefficient_max": np.max(coefficients, axis=0).tolist(),
        }
    merged = pd.concat(
        [merged, pd.DataFrame(stack_predictions, index=merged.index)], axis=1
    )
    pd.DataFrame(model_metrics).sort_values("auprc", ascending=False).to_csv(
        args.output / "junction_metrics.csv", index=False
    )

    score_columns = ["clinical_histgb", "transformer", "q4_vqc"]
    correlations = merged[score_columns].corr(method="spearman")
    correlations.to_csv(args.output / "score_spearman_correlations.csv")
    loss_rows = []
    for left in score_columns:
        for right in score_columns:
            if left >= right:
                continue
            left_loss = binary_log_loss_per_record(labels, merged[left])
            right_loss = binary_log_loss_per_record(labels, merged[right])
            loss_rows.append({
                "model_a": left,
                "model_b": right,
                "loss_spearman_rho": float(spearmanr(left_loss, right_loss).statistic),
                "mean_loss_a_minus_b": float(np.mean(left_loss - right_loss)),
            })
    pd.DataFrame(loss_rows).to_csv(args.output / "loss_complementarity.csv", index=False)
    merged[["ecg_id", "patient_id", "fold", "label", *score_columns, *stack_specs]].to_csv(
        args.output / "diagnostic_predictions.csv", index=False
    )

    route_counts = routing.groupby(["route", "family"], dropna=False).size().rename("features").reset_index()
    route_counts.to_csv(args.output / "routing_summary.csv", index=False)
    summary = {
        "status": "DIAGNOSTIC_ONLY_REQUIRES_NESTED_ROUTING_EXPERIMENT",
        "records": int(len(merged)),
        "patients": int(merged.patient_id.nunique()),
        "folds": sorted(merged.fold.unique().astype(int).tolist()),
        "score_spearman": correlations.to_dict(),
        "stacking": stack_audit,
        "routing_counts": route_counts.to_dict("records"),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
        "warning": "Pooled OOF routing associations formulate hypotheses only; all routing and fusion must be refit inside each outer training fold.",
    }
    (args.output / "feature_routing_audit.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(model_metrics).sort_values("auprc", ascending=False).to_string(index=False))
    print(route_counts.to_string(index=False))


if __name__ == "__main__":
    main()
