"""Patient-safe classical controls for the frozen Transformer ECG embeddings.

The q4 arm uses the exact patient sample and fold-local PLS/quantile mapping of
the Transformer VQC screen. The h128 arm uses that same sample without the
four-coordinate bottleneck. Scores are exploratory and uncalibrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from run_quantum_core_screen import _atomic_npz, _patient_unique_sample
from run_waveform_quantum_core_screen import _embedding_to_angles


MODEL_NAMES = (
    "q4_logreg", "q4_rbf_svc", "q4_laplacian_svc", "q4_histgb",
    "h128_logreg", "h128_rbf_svc", "h128_histgb",
)


def _predict(train_h: np.ndarray, val_h: np.ndarray, train_q: np.ndarray,
             val_q: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    q_lr = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
    q_lr.fit(train_q, y)
    q_rbf = SVC(C=1.0, gamma="scale", class_weight="balanced")
    q_rbf.fit(train_q, y)
    distances = manhattan_distances(train_q)
    upper = distances[np.triu_indices(len(train_q), 1)]
    upper = upper[upper > 1e-12]
    gamma = 1.0 / float(np.median(upper)) if len(upper) else 1.0
    q_lap = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
    q_lap.fit(np.exp(-gamma * distances), y)
    q_hist = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=7, learning_rate=0.05,
        l2_regularization=1.0, random_state=20260922,
    )
    q_hist.fit(train_q, y)
    h_lr = make_pipeline(RobustScaler(), LogisticRegression(
        C=0.1, class_weight="balanced", max_iter=2000,
    ))
    h_lr.fit(train_h, y)
    h_rbf = make_pipeline(RobustScaler(), SVC(
        C=1.0, gamma="scale", class_weight="balanced",
    ))
    h_rbf.fit(train_h, y)
    h_hist = HistGradientBoostingClassifier(
        max_iter=150, max_leaf_nodes=7, learning_rate=0.05,
        l2_regularization=1.0, random_state=20260922,
    )
    h_hist.fit(train_h, y)
    return {
        "q4_logreg": q_lr.predict_proba(val_q)[:, 1],
        "q4_rbf_svc": expit(q_rbf.decision_function(val_q)),
        "q4_laplacian_svc": expit(q_lap.decision_function(
            np.exp(-gamma * manhattan_distances(val_q, train_q)))),
        "q4_histgb": q_hist.predict_proba(val_q)[:, 1],
        "h128_logreg": h_lr.predict_proba(val_h)[:, 1],
        "h128_rbf_svc": expit(h_rbf.decision_function(val_h)),
        "h128_histgb": h_hist.predict_proba(val_h)[:, 1],
    }


def run(representations: Path, metadata_path: Path, output: Path,
        folds: tuple[int, ...] = tuple(range(1, 9)),
        per_class: int = 500, seed: int = 20260922) -> None:
    if not folds or not set(folds) <= set(range(1, 9)):
        raise ValueError("Only development folds 1-8 may be accessed")
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path).set_index("ecg_id", verify_integrity=True)
    keys = ("ecg_id", "patient_id", "fold", "label", "hard_negative", *MODEL_NAMES)
    collected: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    audits = []
    for held_out in folds:
        checkpoint = output / f"fold_{held_out}.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                fold_out = {key: saved[key] for key in keys}
                audit = json.loads(str(saved["audit"].item()))
            print(f"fold {held_out}: resumed", flush=True)
        else:
            source = representations / f"outer_fold_{held_out}_representations.npz"
            with np.load(source, allow_pickle=False) as data:
                train_id = data["train_record_ids"].astype(int)
                val_id = data["val_record_ids"].astype(int)
                train_pat = data["train_patient_ids"].astype(int)
                val_pat = data["val_patient_ids"].astype(int)
                train_y = data["train_labels"].astype(int)
                val_y = data["val_labels"].astype(int)
                train_h = data["train_embeddings"].astype(np.float32)
                val_h = data["val_embeddings"].astype(np.float32)
            if len(set(train_pat) & set(val_pat)):
                raise ValueError(f"Patient leakage in fold {held_out}")
            if len(np.unique(val_id)) != len(val_id):
                raise ValueError("Duplicate validation ECG")
            train_meta, val_meta = metadata.loc[train_id], metadata.loc[val_id]
            if not np.array_equal(train_meta.mi_label.to_numpy(dtype=int), train_y) or not np.array_equal(val_meta.mi_label.to_numpy(dtype=int), val_y):
                raise ValueError("Label mismatch")
            if not np.array_equal(train_meta.patient_id.to_numpy(dtype=int), train_pat) or not np.array_equal(val_meta.patient_id.to_numpy(dtype=int), val_pat):
                raise ValueError("Patient mismatch")
            if not np.all(np.isin(train_meta.strat_fold.to_numpy(dtype=int), range(1, 9))) or not np.all(train_meta.strat_fold.to_numpy(dtype=int) != held_out) or not val_meta.strat_fold.eq(held_out).all():
                raise ValueError("Fold boundary mismatch")
            if not train_meta.eligibility.eq("PRIMARY").all() or not val_meta.eligibility.eq("PRIMARY").all():
                raise ValueError("Non-primary ECG")
            if not np.isfinite(train_h).all() or not np.isfinite(val_h).all():
                raise ValueError("Non-finite Transformer embedding")
            fit = _patient_unique_sample(train_y, train_meta.hard_negative.to_numpy(dtype=bool), train_pat, per_class, seed + held_out)
            q_train, q_val, representation_audit = _embedding_to_angles(
                train_h, train_y, fit, val_h, 4, seed + held_out)
            scores = _predict(train_h[fit], val_h, q_train, q_val, train_y[fit])
            fold_out = {
                "ecg_id": val_id, "patient_id": val_pat,
                "fold": np.full(len(val_y), held_out, dtype=int),
                "label": val_y,
                "hard_negative": val_meta.hard_negative.to_numpy(dtype=bool),
                **scores,
            }
            if any(len(fold_out[name]) != len(val_y) or not np.isfinite(fold_out[name]).all() for name in MODEL_NAMES):
                raise ValueError("Incomplete or non-finite predictions")
            audit = {
                "fold": held_out, "source": source.name,
                "training_records": int(len(fit)),
                "training_patients": int(len(np.unique(train_pat[fit]))),
                "training_ecg_sha256": hashlib.sha256(np.sort(train_id[fit]).tobytes()).hexdigest(),
                "validation_ecg_sha256": hashlib.sha256(np.sort(val_id).tobytes()).hexdigest(),
                "representation": representation_audit,
                "source_encoder_supervised": True,
                "scores_calibrated": False,
                "model_hyperparameters": "prespecified; no outer-fold tuning",
            }
            _atomic_npz(checkpoint, **fold_out, audit=json.dumps(audit, sort_keys=True))
            print(f"fold {held_out}: {len(fit)} train, {len(val_y)} validation", flush=True)
        for key in keys:
            collected[key].append(fold_out[key])
        audits.append(audit)
    all_out = {key: np.concatenate(value) for key, value in collected.items()}
    if len(np.unique(all_out["ecg_id"])) != len(all_out["ecg_id"]):
        raise ValueError("Duplicate OOF ECG")
    metrics = pd.DataFrame([
        {"model": name, "input_dim": 4 if name.startswith("q4") else 128,
         "training_records_per_fold": per_class * 2,
         "auprc": average_precision_score(all_out["label"], all_out[name]),
         "auroc": roc_auc_score(all_out["label"], all_out[name]),
         "scores_calibrated": False}
        for name in MODEL_NAMES
    ]).sort_values("auprc", ascending=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(all_out).to_csv(output / "predictions_wide.csv", index=False)
    (output / "fold_audits.json").write_text(json.dumps(audits, indent=2))
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    run(args.representations, args.metadata, args.output, tuple(args.folds), args.per_class, args.seed)


if __name__ == "__main__":
    main()
