"""Matched q4/q8 heads on frozen label-free ECG-JEPA representations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.svm import SVC

from aquire_preprocessing.models_quantum import (
    DirectQuantumClassifier,
    ReuploadingQuantumClassifier,
)
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import (
    _atomic_npz,
    _json_default,
    _patient_unique_sample,
    _seed_torch,
    _train_torch_head,
)


def embedding_to_pca_angles(
    train_embeddings: np.ndarray,
    fit_relative: np.ndarray,
    val_embeddings: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train-only, label-free h128 → qd transformation."""
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    fit_scaled = scaler.fit_transform(train_embeddings[fit_relative])
    val_scaled = scaler.transform(val_embeddings)
    pca = PCA(n_components=n_components, whiten=True, random_state=seed)
    fit_pca = pca.fit_transform(fit_scaled)
    val_pca = pca.transform(val_scaled)
    quantile = QuantileTransformer(
        n_quantiles=min(256, len(fit_pca)),
        output_distribution="uniform",
        random_state=seed,
    )
    q_fit = (2.0 * quantile.fit_transform(fit_pca) - 1.0) * np.pi
    q_val = (2.0 * quantile.transform(val_pca) - 1.0) * np.pi
    return q_fit.astype(np.float32), q_val.astype(np.float32), {
        "method": "train-only robust scale + whitened PCA + quantile angles; label-free",
        "source_dim": int(train_embeddings.shape[1]),
        "target_dim": int(n_components),
        "fit_records": int(len(fit_relative)),
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "labels_used_by_transform": False,
    }


def _matched_mlp(n_features: int):
    import torch

    hidden = 4 if n_features == 4 else 5
    return torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
        torch.nn.Flatten(0),
    )


def _heads(q_train, y_train, q_val, dimension: int, epochs: int, seed: int):
    if dimension == 4:
        quantum = DirectQuantumClassifier(n_qubits=4, n_layers=2, topology="ring")
    elif dimension == 8:
        quantum = ReuploadingQuantumClassifier(input_dim=8, n_qubits=4, topology="ring")
    else:
        raise ValueError("Only q4 and q8 are supported")
    _seed_torch(seed)
    quantum_logits, quantum_audit = _train_torch_head(
        quantum, q_train, y_train, q_val, epochs, 64, 0.01, seed
    )
    _seed_torch(seed + 1000)
    mlp_logits, mlp_audit = _train_torch_head(
        _matched_mlp(dimension), q_train, y_train, q_val, epochs, 64, 0.01, seed + 1000
    )
    logistic = LogisticRegression(C=0.1, class_weight="balanced", max_iter=1000)
    logistic.fit(q_train, y_train)
    rbf = SVC(C=1.0, gamma="scale", class_weight="balanced")
    rbf.fit(q_train, y_train)
    prefix = f"q{dimension}"
    return {
        f"{prefix}_vqc": expit(quantum_logits),
        f"{prefix}_logreg": logistic.predict_proba(q_val)[:, 1],
        f"{prefix}_mlp": expit(mlp_logits),
        f"{prefix}_rbf_svc": expit(rbf.decision_function(q_val)),
    }, {"vqc": quantum_audit, "mlp": mlp_audit}


def run_screen(
    representation_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    *,
    per_class: int = 500,
    epochs: int = 20,
    seed: int = 20260922,
    folds: tuple[int, ...] = tuple(range(1, 9)),
) -> None:
    if not folds or not set(folds) <= set(range(1, 9)):
        raise ValueError("Only development folds 1-8 may be used")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path).set_index("ecg_id", verify_integrity=True)
    models = tuple(f"q{d}_{head}" for d in (4, 8) for head in ("vqc", "logreg", "mlp", "rbf_svc"))
    keys = ("ecg_id", "patient_id", "fold", "label", "hard_negative", *models)
    collected = {key: [] for key in keys}
    audits = []
    for held_out in folds:
        checkpoint = output_dir / f"fold_{held_out}.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                fold_out = {key: saved[key] for key in keys}
                audit = json.loads(str(saved["audit"].item()))
            print(f"fold {held_out}: resumed", flush=True)
        else:
            source = representation_dir / f"outer_fold_{held_out}_representations.npz"
            with np.load(source, allow_pickle=False) as data:
                train_ids = data["train_record_ids"].astype(int)
                val_ids = data["val_record_ids"].astype(int)
                train_patients = data["train_patient_ids"].astype(int)
                val_patients = data["val_patient_ids"].astype(int)
                train_y = data["train_labels"].astype(int)
                val_y = data["val_labels"].astype(int)
                train_h = data["train_embeddings"].astype(np.float32)
                val_h = data["val_embeddings"].astype(np.float32)
            if len(set(train_patients) & set(val_patients)):
                raise ValueError(f"Patient leakage in fold {held_out}")
            train_meta, val_meta = metadata.loc[train_ids], metadata.loc[val_ids]
            if not np.array_equal(train_meta.mi_label.to_numpy(dtype=int), train_y) or not np.array_equal(val_meta.mi_label.to_numpy(dtype=int), val_y):
                raise ValueError("Label mismatch against immutable metadata")
            if not np.array_equal(train_meta.patient_id.to_numpy(dtype=int), train_patients) or not np.array_equal(val_meta.patient_id.to_numpy(dtype=int), val_patients):
                raise ValueError("Patient mismatch")
            if not np.all(np.isin(train_meta.strat_fold.to_numpy(dtype=int), range(1, 9))) or not np.all(train_meta.strat_fold.to_numpy(dtype=int) != held_out) or not val_meta.strat_fold.eq(held_out).all():
                raise ValueError("Fold boundary mismatch")
            if not train_meta.eligibility.eq("PRIMARY").all() or not val_meta.eligibility.eq("PRIMARY").all():
                raise ValueError("Non-primary ECG entered the screen")
            if not np.isfinite(train_h).all() or not np.isfinite(val_h).all():
                raise ValueError("Non-finite JEPA embedding")
            fit = _patient_unique_sample(
                train_y, train_meta.hard_negative.to_numpy(dtype=bool),
                train_patients, per_class, seed + held_out,
            )
            fold_scores, representations, head_audits = {}, {}, {}
            for dimension in (4, 8):
                q_train, q_val, rep_audit = embedding_to_pca_angles(
                    train_h, fit, val_h, dimension, seed + held_out
                )
                scores, head_audit = _heads(
                    q_train, train_y[fit], q_val, dimension, epochs,
                    seed + held_out + dimension * 100,
                )
                fold_scores.update(scores)
                representations[f"q{dimension}"] = rep_audit
                head_audits[f"q{dimension}"] = head_audit
            fold_out = {
                "ecg_id": val_ids, "patient_id": val_patients,
                "fold": np.full(len(val_y), held_out, dtype=int),
                "label": val_y,
                "hard_negative": val_meta.hard_negative.to_numpy(dtype=bool),
                **fold_scores,
            }
            if any(len(fold_out[name]) != len(val_y) or not np.isfinite(fold_out[name]).all() for name in models):
                raise ValueError("Incomplete or non-finite predictions")
            audit = {
                "held_out_fold": held_out,
                "source": source.name,
                "training_records": int(len(fit)),
                "training_patients": int(len(np.unique(train_patients[fit]))),
                "training_ecg_sha256": hashlib.sha256(np.sort(train_ids[fit]).tobytes()).hexdigest(),
                "validation_ecg_sha256": hashlib.sha256(np.sort(val_ids).tobytes()).hexdigest(),
                "representations": representations,
                "heads": head_audits,
                "source_encoder_supervision": "label-free masked latent prediction",
                "scores_calibrated": False,
            }
            _atomic_npz(
                checkpoint, **fold_out,
                audit=json.dumps(audit, default=_json_default, sort_keys=True),
            )
            print(f"fold {held_out}: q4/q8 complete", flush=True)
        for key in keys:
            collected[key].append(fold_out[key])
        audits.append(audit)
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    if len(np.unique(arrays["ecg_id"])) != len(arrays["ecg_id"]):
        raise ValueError("Duplicate OOF ECG")
    metric_rows = []
    prediction_rows = []
    for name in models:
        metric_rows.append({
            "model": name,
            "input_dim": int(name[1]),
            "auprc": float(average_precision_score(arrays["label"], arrays[name])),
            "auroc": float(roc_auc_score(arrays["label"], arrays[name])),
            "scores_calibrated": False,
        })
        for index, score in enumerate(arrays[name]):
            prediction_rows.append({
                "ecg_id": int(arrays["ecg_id"][index]),
                "patient_id": int(arrays["patient_id"][index]),
                "fold": int(arrays["fold"][index]),
                "label": int(arrays["label"][index]),
                "hard_negative": bool(arrays["hard_negative"][index]),
                "model": name, "score": float(score),
            })
    metrics = pd.DataFrame(metric_rows).sort_values("auprc", ascending=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "fold_audits.json").write_text(
        json.dumps(audits, indent=2, default=_json_default)
    )
    comparisons = (
        ("q4_vqc", "q4_logreg"), ("q4_vqc", "q4_mlp"),
        ("q8_vqc", "q8_logreg"), ("q8_vqc", "q8_mlp"),
        ("q8_vqc", "q4_vqc"),
    )
    for left, right in comparisons:
        report = _paired_patient_bootstrap(
            arrays["label"], arrays["patient_id"], arrays[left], arrays[right],
            iterations=2000, seed=seed, comparison=f"{left}_minus_{right}",
        )
        (output_dir / f"paired_{left}_vs_{right}.json").write_text(
            json.dumps(report, indent=2, default=_json_default)
        )
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    args = parser.parse_args()
    run_screen(
        args.representations, args.metadata, args.output,
        per_class=args.per_class, epochs=args.epochs, seed=args.seed,
        folds=tuple(args.folds),
    )


if __name__ == "__main__":
    main()
