"""Test the best q4 representation on a wider sixteen-qubit VQC core."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _atomic_npz, _json_default, _patient_unique_sample, _seed_torch
from run_supervised_qubit_scaling_screen import (
    _matched_mlp,
    _sensitivity_at_specificity,
    _train_torch_head_device,
)
from run_waveform_quantum_core_screen import _embedding_to_angles


MODEL_NAMES = (
    "q4_on_16q_vqc",
    "q4_logreg",
    "q4_matched_mlp_177",
    "q4_rbf_svc",
    "h128_logreg",
)


def _metric_row(name, labels, scores, hard_negative):
    hard_mask = (labels == 1) | hard_negative
    return {
        "model": name,
        "input_dim": 128 if name == "h128_logreg" else 4,
        "qubits": 16 if name == "q4_on_16q_vqc" else 0,
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "sensitivity_at_90_specificity": _sensitivity_at_specificity(labels, scores),
        "hard_negative_auprc": float(average_precision_score(labels[hard_mask], scores[hard_mask])),
        "hard_negative_auroc": float(roc_auc_score(labels[hard_mask], scores[hard_mask])),
        "scores_calibrated": False,
    }


def run_screen(
    representations: Path,
    metadata_path: Path,
    output: Path,
    *,
    per_class: int = 500,
    epochs: int = 20,
    batch_size: int = 16,
    seed: int = 20260922,
    folds: tuple[int, ...] = tuple(range(1, 9)),
    device: str = "cuda",
) -> None:
    if not folds or not set(folds) <= set(range(1, 9)):
        raise ValueError("Only development folds 1-8 may be used")
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    if not metadata.ecg_id.is_unique:
        raise ValueError("Metadata ecg_id must be unique")
    metadata = metadata.set_index("ecg_id")
    keys = ("ecg_id", "patient_id", "fold", "label", "hard_negative", *MODEL_NAMES)
    collected = {key: [] for key in keys}
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
                raise ValueError("Label mismatch")
            if not np.array_equal(train_meta.patient_id.to_numpy(dtype=int), train_patients) or not np.array_equal(val_meta.patient_id.to_numpy(dtype=int), val_patients):
                raise ValueError("Patient mismatch")
            if not np.all(np.isin(train_meta.strat_fold.to_numpy(dtype=int), range(1, 9))) or not np.all(train_meta.strat_fold.to_numpy(dtype=int) != held_out) or not val_meta.strat_fold.eq(held_out).all():
                raise ValueError("Fold boundary mismatch")
            if not train_meta.eligibility.eq("PRIMARY").all() or not val_meta.eligibility.eq("PRIMARY").all():
                raise ValueError("Non-primary ECG")
            if not np.isfinite(train_h).all() or not np.isfinite(val_h).all():
                raise ValueError("Non-finite Transformer embedding")

            fit = _patient_unique_sample(
                train_y, train_meta.hard_negative.to_numpy(dtype=bool),
                train_patients, per_class, seed + held_out,
            )
            q_train, q_val, representation_audit = _embedding_to_angles(
                train_h, train_y, fit, val_h, 4, seed + held_out
            )
            y_fit = train_y[fit]
            _seed_torch(seed + held_out + 1600)
            vqc = TorchStatevectorQuantumClassifier(
                n_qubits=16, input_dim=4, n_layers=2, topology="ring"
            )
            vqc_parameters = sum(parameter.numel() for parameter in vqc.parameters())
            mlp, hidden = _matched_mlp(4, vqc_parameters)
            print(
                f"fold {held_out}: q4 on 16 qubits, train={len(fit)}, val={len(val_y)}, "
                f"VQC params={vqc_parameters}, matched MLP hidden={hidden}", flush=True,
            )
            vqc_logits, vqc_audit = _train_torch_head_device(
                vqc, q_train, y_fit, q_val,
                epochs=epochs, batch_size=batch_size, learning_rate=0.01,
                seed=seed + held_out + 1600, device_name=device,
            )
            mlp_logits, mlp_audit = _train_torch_head_device(
                mlp, q_train, y_fit, q_val,
                epochs=epochs, batch_size=128, learning_rate=0.01,
                seed=seed + held_out + 2600, device_name=device,
            )
            logistic = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
            logistic.fit(q_train, y_fit)
            rbf = SVC(C=1.0, gamma="scale", class_weight="balanced")
            rbf.fit(q_train, y_fit)
            h128_logistic = make_pipeline(
                RobustScaler(),
                LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
            )
            h128_logistic.fit(train_h[fit], y_fit)
            fold_out = {
                "ecg_id": val_ids,
                "patient_id": val_patients,
                "fold": np.full(len(val_y), held_out, dtype=int),
                "label": val_y,
                "hard_negative": val_meta.hard_negative.to_numpy(dtype=bool),
                "q4_on_16q_vqc": expit(vqc_logits),
                "q4_logreg": logistic.predict_proba(q_val)[:, 1],
                "q4_matched_mlp_177": expit(mlp_logits),
                "q4_rbf_svc": expit(rbf.decision_function(q_val)),
                "h128_logreg": h128_logistic.predict_proba(val_h)[:, 1],
            }
            if any(len(fold_out[name]) != len(val_y) or not np.isfinite(fold_out[name]).all() for name in MODEL_NAMES):
                raise ValueError("Incomplete or non-finite prediction")
            audit = {
                "held_out_fold": held_out,
                "source": source.name,
                "training_records": int(len(fit)),
                "training_patients": int(len(np.unique(train_patients[fit]))),
                "training_ecg_sha256": hashlib.sha256(np.sort(train_ids[fit]).tobytes()).hexdigest(),
                "validation_ecg_sha256": hashlib.sha256(np.sort(val_ids).tobytes()).hexdigest(),
                "representation": representation_audit,
                "vqc": vqc_audit,
                "matched_mlp": {**mlp_audit, "hidden_units": hidden},
                "wire_feature_indices": vqc.wire_feature_indices.detach().cpu().tolist(),
                "statevector_amplitudes": 1 << 16,
                "scores_calibrated": False,
                "folds_9_and_10_accessed": False,
            }
            _atomic_npz(
                checkpoint, **fold_out,
                audit=json.dumps(audit, default=_json_default, sort_keys=True),
            )
        for key in keys:
            collected[key].append(fold_out[key])
        audits.append(audit)

    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    if len(np.unique(arrays["ecg_id"])) != len(arrays["ecg_id"]):
        raise ValueError("Duplicate OOF ECG")
    metrics = pd.DataFrame([
        _metric_row(name, arrays["label"], arrays[name], arrays["hard_negative"])
        for name in MODEL_NAMES
    ]).sort_values("auprc", ascending=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(arrays).to_csv(output / "predictions_wide.csv", index=False)
    (output / "fold_audits.json").write_text(json.dumps(audits, indent=2, default=_json_default))
    for control in ("q4_logreg", "q4_matched_mlp_177", "q4_rbf_svc"):
        report = _paired_patient_bootstrap(
            arrays["label"], arrays["patient_id"], arrays["q4_on_16q_vqc"], arrays[control],
            iterations=2000, seed=seed, comparison=f"q4_on_16q_vqc_minus_{control}",
        )
        (output / f"paired_q4_on_16q_vqc_vs_{control}.json").write_text(
            json.dumps(report, indent=2, default=_json_default)
        )
    (output / "artifact_manifest.json").write_text(json.dumps({
        "experiment": "q4_representation_on_16_qubit_vqc",
        "input_dim": 4,
        "qubits": 16,
        "folds": list(folds),
        "records": int(len(arrays["label"])),
        "patients": int(len(np.unique(arrays["patient_id"]))),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }, indent=2))
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run_screen(
        args.representations, args.metadata, args.output,
        per_class=args.per_class, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, folds=tuple(args.folds), device=args.device,
    )


if __name__ == "__main__":
    main()
