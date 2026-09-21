"""Bounded patient-safe QSVC or compact HQNN screen on development folds 1-8.

This is an exploratory, one-seed experiment.  No probability calibration or
clinical/quantum-advantage claim follows from its uncalibrated scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.svm import SVC

from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.models_quantum import CompactFusionHQNN, QuantumKernelEstimator
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import (
    _atomic_npz,
    _fit_representation,
    _json_default,
    _patient_unique_sample,
    _seed_torch,
    _train_torch_head,
)
from run_waveform_quantum_core_screen import _embedding_to_angles


def _rbf_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    d2 = euclidean_distances(left, right, squared=True)
    return np.exp(-gamma * np.maximum(d2, 0.0))


def _laplacian_kernel(left: np.ndarray, right: np.ndarray, gamma: float) -> np.ndarray:
    return np.exp(-gamma * manhattan_distances(left, right))


def _median_gamma(train: np.ndarray, metric: str) -> float:
    subset = train[: min(len(train), 500)]
    distances = (
        euclidean_distances(subset, subset, squared=True)
        if metric == "sqeuclidean"
        else manhattan_distances(subset, subset)
    )
    positive = distances[np.triu_indices(len(subset), 1)]
    positive = positive[positive > 1e-12]
    return float(1.0 / np.median(positive)) if len(positive) else 1.0


def _kernel_scores(q_train: np.ndarray, q_val: np.ndarray, labels: np.ndarray) -> tuple[dict, dict]:
    estimator = QuantumKernelEstimator(n_qubits=4, n_layers=2)
    train_states = estimator.get_statevectors(q_train)
    val_states = estimator.get_statevectors(q_val)
    quantum_train = np.abs(train_states @ train_states.conj().T) ** 2
    quantum_train = (quantum_train + quantum_train.T) / 2.0
    np.fill_diagonal(quantum_train, 1.0)
    quantum_val = np.abs(val_states @ train_states.conj().T) ** 2
    # A bounded PSD audit avoids an unnecessary eigendecomposition of every
    # 1000-by-1000 Gram matrix.  State fidelity is PSD by construction.
    sampled = quantum_train[: min(128, len(quantum_train)), : min(128, len(quantum_train))]
    diagnostics = estimator.diagnostics(sampled)
    if diagnostics["negative_eigenvalue_count"] or not np.isfinite(quantum_val).all():
        raise ValueError(f"Invalid IQP kernel: {diagnostics}")
    gamma_rbf = _median_gamma(q_train, "sqeuclidean")
    gamma_lap = _median_gamma(q_train, "manhattan")
    kernels = {
        "iqp_qsvc": (quantum_train, quantum_val),
        "rbf_svc": (_rbf_kernel(q_train, q_train, gamma_rbf), _rbf_kernel(q_val, q_train, gamma_rbf)),
        "laplacian_svc": (_laplacian_kernel(q_train, q_train, gamma_lap), _laplacian_kernel(q_val, q_train, gamma_lap)),
    }
    scores = {}
    support = {}
    for name, (train_kernel, val_kernel) in kernels.items():
        svm = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
        svm.fit(train_kernel, labels)
        scores[name] = expit(svm.decision_function(val_kernel))
        support[name] = int(len(svm.support_))
    return scores, {"kernel_diagnostics_128": diagnostics, "support_vectors": support,
                    "rbf_gamma": gamma_rbf, "laplacian_gamma": gamma_lap}


def _fusion_mlp():
    import torch

    class MatchedFusionMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = torch.nn.Linear(4, 4)
            with torch.no_grad():
                self.mixer.weight.copy_(torch.eye(4))
                self.mixer.bias.zero_()
            self.head = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))

        def forward(self, x):
            return self.head(torch.pi * torch.tanh(self.mixer(x) / torch.pi)).squeeze(-1)

    return MatchedFusionMLP()


def _joined_hqnn():
    import torch

    class JoinedHQNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = CompactFusionHQNN(n_qubits=4, n_layers=2)

        def forward(self, x):
            return self.head(x[:, :2], x[:, 2:])

    return JoinedHQNN()


def _load_features(features_path: Path, manifest_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(features_path).set_index("ecg_id", verify_integrity=True)
    approved = load_feature_manifest(manifest_path, frame.columns)["approved_features"]
    return frame[approved].apply(pd.to_numeric, errors="coerce")


def run_screen(
    task: str,
    representation_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    features_path: Path | None = None,
    manifest_path: Path | None = None,
    per_class: int = 500,
    epochs: int = 20,
    seed: int = 20260922,
    folds: tuple[int, ...] = tuple(range(1, 9)),
) -> None:
    if task not in {"qsvc", "hqnn"}:
        raise ValueError("task must be qsvc or hqnn")
    if not set(folds) <= set(range(1, 9)):
        raise ValueError("Only development folds 1-8 may be used")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path).set_index("ecg_id", verify_integrity=True)
    if task == "hqnn":
        if features_path is None or manifest_path is None:
            raise ValueError("HQNN requires approved deployable features and manifest")
        features = _load_features(features_path, manifest_path)
        names = ("fusion_hqnn", "fusion_mlp", "waveform_rbf_svc")
    else:
        features = None
        names = ("iqp_qsvc", "rbf_svc", "laplacian_svc")
    collected = {key: [] for key in ("ecg_id", "patient_id", "fold", "label", "hard_negative", *names)}
    audits = []
    for held_out in folds:
        source = representation_dir / f"outer_fold_{held_out}_representations.npz"
        checkpoint = output_dir / f"fold_{held_out}.npz"
        if checkpoint.exists():
            saved = np.load(checkpoint, allow_pickle=False)
            fold_output = {key: saved[key] for key in collected}
            audit = json.loads(str(saved["audit"].item()))
            print(f"fold {held_out}: resumed", flush=True)
        else:
            data = np.load(source, allow_pickle=False)
            train_ids = data["train_record_ids"].astype(int)
            val_ids = data["val_record_ids"].astype(int)
            train_y = data["train_labels"].astype(int)
            val_y = data["val_labels"].astype(int)
            train_patient = data["train_patient_ids"].astype(int)
            val_patient = data["val_patient_ids"].astype(int)
            if len(set(train_patient) & set(val_patient)):
                raise ValueError(f"Patient leakage in fold {held_out}")
            if len(np.unique(val_ids)) != len(val_ids):
                raise ValueError("Duplicate validation ECG")
            train_meta = metadata.loc[train_ids]
            val_meta = metadata.loc[val_ids]
            if not np.array_equal(train_meta.mi_label.to_numpy(dtype=int), train_y) or not np.array_equal(val_meta.mi_label.to_numpy(dtype=int), val_y):
                raise ValueError("Label mismatch against immutable metadata")
            train_folds = train_meta.strat_fold.to_numpy(dtype=int)
            val_folds = val_meta.strat_fold.to_numpy(dtype=int)
            if not np.all(np.isin(train_folds, range(1, 9))) or not np.all(train_folds != held_out) or not np.all(val_folds == held_out):
                raise ValueError("Outer-fold boundary mismatch")
            if not train_meta.eligibility.eq("PRIMARY").all() or not val_meta.eligibility.eq("PRIMARY").all():
                raise ValueError("Non-primary ECG entered the screen")
            if not np.array_equal(train_meta.patient_id.to_numpy(dtype=int), train_patient) or not np.array_equal(val_meta.patient_id.to_numpy(dtype=int), val_patient):
                raise ValueError("Patient ID mismatch against immutable metadata")
            fit = _patient_unique_sample(train_y, train_meta.hard_negative.to_numpy(dtype=bool), train_patient, per_class, seed + held_out)
            waveform_train = data["train_embeddings"].astype(np.float32)
            waveform_val = data["val_embeddings"].astype(np.float32)
            if not np.isfinite(waveform_train).all() or not np.isfinite(waveform_val).all():
                raise ValueError("Non-finite waveform embedding")
            dim = 4 if task == "qsvc" else 2
            q_wave_train, q_wave_val, wave_audit = _embedding_to_angles(waveform_train, train_y, fit, waveform_val, dim, seed + held_out)
            if task == "qsvc":
                scores, model_audit = _kernel_scores(q_wave_train, q_wave_val, train_y[fit])
            else:
                assert features is not None
                clinical_train = features.loc[train_ids].to_numpy(dtype=float)
                clinical_val = features.loc[val_ids].to_numpy(dtype=float)
                all_clinical = np.concatenate([clinical_train, clinical_val])
                all_labels = np.concatenate([train_y, val_y])
                val_relative = np.arange(len(train_y), len(all_labels))
                q_clin_train, q_clin_val, clinical_audit = _fit_representation(all_clinical, all_labels, fit, val_relative, 2, seed + held_out)
                joined_train = np.concatenate([q_wave_train, q_clin_train], axis=1)
                joined_val = np.concatenate([q_wave_val, q_clin_val], axis=1)
                _seed_torch(seed + held_out)
                hqnn = _joined_hqnn()
                hqnn_logits, hqnn_audit = _train_torch_head(hqnn, joined_train, train_y[fit], joined_val, epochs, 64, 0.01, seed + held_out)
                _seed_torch(seed + 1000 + held_out)
                mlp_logits, mlp_audit = _train_torch_head(_fusion_mlp(), joined_train, train_y[fit], joined_val, epochs, 64, 0.01, seed + held_out)
                rbf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
                rbf.fit(q_wave_train, train_y[fit])
                scores = {"fusion_hqnn": expit(hqnn_logits), "fusion_mlp": expit(mlp_logits),
                          "waveform_rbf_svc": expit(rbf.decision_function(q_wave_val))}
                model_audit = {"hqnn": hqnn_audit, "mlp": mlp_audit, "clinical": clinical_audit}
            fold_output = {"ecg_id": val_ids, "patient_id": val_patient,
                           "fold": np.full(len(val_y), held_out, dtype=int),
                           "label": val_y, "hard_negative": val_meta.hard_negative.to_numpy(dtype=bool),
                           **scores}
            if any(len(fold_output[name]) != len(val_y) or not np.isfinite(fold_output[name]).all() for name in names):
                raise ValueError("Non-finite or incomplete fold predictions")
            audit = {"held_out_fold": held_out, "source": source.name,
                     "training_records": int(len(fit)), "training_patients": int(len(np.unique(train_patient[fit]))),
                     "training_ecg_sha256": hashlib.sha256(np.sort(train_ids[fit]).tobytes()).hexdigest(),
                     "validation_ecg_sha256": hashlib.sha256(np.sort(val_ids).tobytes()).hexdigest(),
                     "waveform_representation": wave_audit, "model": model_audit,
                     "source_encoder_supervised": True, "scores_calibrated": False}
            _atomic_npz(checkpoint, **fold_output, audit=json.dumps(audit, default=_json_default, sort_keys=True))
            print(f"fold {held_out}: {task} complete; {len(fit)} train, {len(val_y)} validation", flush=True)
        for key in collected:
            collected[key].append(fold_output[key])
        audits.append(audit)
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    if len(np.unique(arrays["ecg_id"])) != len(arrays["ecg_id"]):
        raise ValueError("Duplicate OOF ECG")
    rows = []
    metrics = []
    for name in names:
        score = arrays[name]
        metrics.append({"model": name, "auprc": float(average_precision_score(arrays["label"], score)),
                        "auroc": float(roc_auc_score(arrays["label"], score)), "scores_calibrated": False})
        for i in range(len(score)):
            rows.append({"ecg_id": int(arrays["ecg_id"][i]), "patient_id": int(arrays["patient_id"][i]),
                         "fold": int(arrays["fold"][i]), "label": int(arrays["label"][i]),
                         "hard_negative": bool(arrays["hard_negative"][i]), "model": name, "score": float(score[i])})
    pd.DataFrame(metrics).sort_values("auprc", ascending=False).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "fold_audits.json").write_text(json.dumps(audits, indent=2, default=_json_default))
    for control in names[1:]:
        report = _paired_patient_bootstrap(arrays["label"], arrays["patient_id"], arrays[names[0]], arrays[control], iterations=2000, seed=seed)
        report["comparison"] = f"{names[0]}_minus_{control}"
        (output_dir / f"paired_{names[0]}_vs_{control}.json").write_text(json.dumps(report, indent=2, default=_json_default))
    print(pd.DataFrame(metrics).sort_values("auprc", ascending=False).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("qsvc", "hqnn"), required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    run_screen(args.task, args.representations, args.metadata, args.output,
               features_path=args.features, manifest_path=args.manifest,
               per_class=args.per_class, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
