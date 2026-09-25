"""Trace predictive and geometric losses through the current ECG/QML pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.manifest import guard_fold_access
from run_advanced_quantum_fusion_screen import _fit_q4
from run_divergence_distillation_screen import _validate_representations
from run_quantum_baselines import _paired_patient_bootstrap


def _metrics(y, p, prevalence, ceiling):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1.0 - 1e-7)
    auprc = float(average_precision_score(y, p))
    return {
        "auprc": auprc,
        "auroc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
        "normalized_discrimination_0_100": float(
            100.0 * (auprc - prevalence) / (ceiling - prevalence)
        ),
    }


def _effective_rank(values: np.ndarray) -> float:
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False) ** 2
    probability = singular / singular.sum()
    entropy = -np.sum(probability * np.log(probability.clip(1e-12)))
    return float(np.exp(entropy))


def _geometry(h: np.ndarray, q: np.ndarray, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    take = rng.choice(len(h), size=min(1200, len(h)), replace=False)
    h_sub, q_sub = h[take], q[take]
    # Rank dimensions before distance calculations so no coordinate dominates.
    h_scale = np.std(h_sub, axis=0).clip(1e-6)
    q_scale = np.std(q_sub, axis=0).clip(1e-6)
    h_standard = (h_sub - np.mean(h_sub, axis=0)) / h_scale
    q_standard = (q_sub - np.mean(q_sub, axis=0)) / q_scale
    distance_rho = float(spearmanr(pdist(h_standard), pdist(q_standard)).statistic)
    neighbors = min(16, len(h_sub))
    h_nn = NearestNeighbors(n_neighbors=neighbors).fit(h_standard).kneighbors(return_distance=False)[:, 1:]
    q_nn = NearestNeighbors(n_neighbors=neighbors).fit(q_standard).kneighbors(return_distance=False)[:, 1:]
    overlap = np.mean(
        [len(set(left).intersection(right)) / (neighbors - 1) for left, right in zip(h_nn, q_nn)]
    )
    return {
        "sample_records": int(len(h_sub)),
        "h128_effective_rank": _effective_rank(h_sub),
        "q4_effective_rank": _effective_rank(q_sub),
        "pairwise_distance_spearman": distance_rho,
        "knn15_overlap": float(overlap),
    }


def _read_transformer(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=False) as data:
        return pd.DataFrame(
            {
                "ecg_id": data["record_ids"].astype(int),
                "transformer": data["raw_probability"].astype(float),
            }
        ).set_index("ecg_id")


def run(args) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    joined = metadata[
        metadata.strat_fold.isin(DEV_FOLDS) & metadata.eligibility.eq("PRIMARY")
    ].copy()
    folds = joined.strat_fold.to_numpy(int)
    guard_fold_access(folds, purpose="tuning")
    labels = joined.mi_label.to_numpy(int)
    patients = joined.patient_id.to_numpy(int)
    record_ids = joined.index.to_numpy(int)
    row_for_id = {int(ecg_id): row for row, ecg_id in enumerate(record_ids)}
    paths = _validate_representations(args.representations, joined)

    q4 = np.full((len(joined), 4), np.nan, dtype=np.float32)
    h128 = np.full((len(joined), 128), np.nan, dtype=np.float32)
    h128_probe = np.full(len(joined), np.nan)
    q4_probe = np.full(len(joined), np.nan)
    geometry = []
    for held_out in sorted(paths):
        with np.load(paths[held_out], allow_pickle=False) as data:
            train_ids = data["train_record_ids"].astype(int)
            val_ids = data["val_record_ids"].astype(int)
            train_h = data["train_embeddings"].astype(np.float32)
            val_h = data["val_embeddings"].astype(np.float32)
            train_y = data["train_labels"].astype(int)
        train_rows = np.asarray([row_for_id[int(value)] for value in train_ids])
        val_rows = np.asarray([row_for_id[int(value)] for value in val_ids])
        train_q, val_q, _ = _fit_q4(train_h, train_y, val_h, args.seed + held_out)
        h128[val_rows], q4[val_rows] = val_h, val_q
        h_model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(
            train_h, train_y
        )
        q_model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(
            train_q, train_y
        )
        h128_probe[val_rows] = h_model.predict_proba(val_h)[:, 1]
        q4_probe[val_rows] = q_model.predict_proba(val_q)[:, 1]
        geometry.append({"fold": held_out, **_geometry(val_h, val_q, args.seed + held_out)})
    if not (np.isfinite(h128).all() and np.isfinite(q4).all()):
        raise RuntimeError("Incomplete fold-coherent representations")

    transformer = _read_transformer(args.transformer_oof)
    difficulty = pd.read_csv(args.difficulty_oof).set_index("ecg_id")
    divergence = pd.read_csv(args.divergence_oof).set_index("ecg_id")
    aligned = pd.DataFrame(index=joined.index)
    aligned["y_true"] = labels
    aligned["patient_id"] = patients
    aligned["strat_fold"] = folds
    aligned["transformer"] = transformer.loc[aligned.index, "transformer"]
    aligned["h128_logistic_full"] = h128_probe
    aligned["q4_logistic_full"] = q4_probe
    aligned["q4_mlp_full"] = difficulty.loc[aligned.index, "q4_assistant_full"]
    aligned["q4_mlp_sample"] = difficulty.loc[aligned.index, "q4_mlp"]
    aligned["vqc_hard"] = difficulty.loc[aligned.index, "vqc_hard"]
    aligned["vqc_curriculum"] = difficulty.loc[aligned.index, "vqc_curriculum_difficulty_js"]
    aligned["vqc_js"] = divergence.loc[aligned.index, "vqc_js_t2"]
    aligned["fusion_vqc_curriculum"] = difficulty.loc[
        aligned.index, "fusion_curriculum_difficulty_js"
    ]
    aligned["fusion_classical"] = difficulty.loc[aligned.index, "fusion_q4_mlp"]
    if aligned.isna().any().any():
        raise RuntimeError("Trace predictions are incomplete after ECG-ID alignment")

    prevalence = float(labels.mean())
    ceiling = float(average_precision_score(labels, aligned["fusion_classical"]))
    stage_columns = [column for column in aligned.columns if column not in {"y_true", "patient_id", "strat_fold"}]
    stage_metrics = {
        column: _metrics(labels, aligned[column], prevalence, ceiling)
        for column in stage_columns
    }
    pairs = [
        ("h128_logistic_full", "q4_logistic_full"),
        ("q4_mlp_sample", "vqc_js"),
        ("vqc_hard", "vqc_curriculum"),
        ("fusion_classical", "fusion_vqc_curriculum"),
    ]
    comparisons = {}
    for better, other in pairs:
        key = f"{better}_minus_{other}"
        comparisons[key] = _paired_patient_bootstrap(
            labels, patients, aligned[better], aligned[other],
            iterations=args.bootstrap_iterations, seed=args.seed, comparison=key,
        )
    correlation = aligned[stage_columns].corr(method="spearman")
    summary = {
        "records": int(len(aligned)),
        "patients": int(len(np.unique(patients))),
        "prevalence": prevalence,
        "normalization": (
            "0=class prevalence AUPRC; 100=current all-classical fusion AUPRC. "
            "This is a relative discrimination index, not literal information retained."
        ),
        "ceiling_auprc": ceiling,
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    aligned.to_csv(args.output / "stage_predictions.csv")
    pd.DataFrame(stage_metrics).T.to_csv(args.output / "stage_metrics.csv")
    pd.DataFrame(geometry).to_csv(args.output / "fold_geometry.csv", index=False)
    correlation.to_csv(args.output / "stage_score_spearman.csv")
    (args.output / "bootstrap.json").write_text(json.dumps(comparisons, indent=2))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "metrics": stage_metrics}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--transformer-oof", type=Path, required=True)
    parser.add_argument("--difficulty-oof", type=Path, required=True)
    parser.add_argument("--divergence-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
