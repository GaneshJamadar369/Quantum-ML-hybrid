"""Staged, patient-safe Phase 6Q development benchmark.

The default fast gate compares an IQP fidelity-kernel QSVM with an RBF-SVC on
the identical fold-local PCA z8, balanced training sample, outer folds, and
train-only calibration design.  VQC/HQNN remain explicit opt-in experiments.
Fold 9 and Fold 10 are rejected by the development access guard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from aquire_preprocessing.baselines import (
    evaluate_probabilities,
)
from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_quantum import (
    QSVMClassifier,
    VariationalQuantumClassifier,
    HybridQuantumNeuralNetwork,
)


SUPPORTED_MODELS = (
    "qsvm",
    "rbf_svc_z8",
    "rbf_svc_angle_z8",
    "polynomial_svc_angle_z8",
    "laplacian_svc_angle_z8",
    "product_cosine_svc_angle_z8",
    "vqc",
    "hqnn",
)

ANGLE_KERNEL_CONTROLS = {
    "rbf_svc_angle_z8": "rbf",
    "polynomial_svc_angle_z8": "poly",
    "laplacian_svc_angle_z8": "laplacian",
    "product_cosine_svc_angle_z8": "product_cosine",
}


def _fold_local_z8(
    features: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit an unsupervised eight-dimensional representation on one outer train fold."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    x_train = scaler.fit_transform(imputer.fit_transform(features[train_indices]))
    x_val = scaler.transform(imputer.transform(features[val_indices]))
    n_components = min(8, x_train.shape[1], x_train.shape[0] - 1)
    if n_components != 8:
        raise ValueError(f"z8 requires at least eight usable dimensions; got {n_components}")
    pca = PCA(n_components=8, whiten=True, random_state=seed)
    z_train = pca.fit_transform(x_train).astype(np.float32)
    z_val = pca.transform(x_val).astype(np.float32)
    return z_train, z_val, {
        "method": "training-fold median imputation + robust scaling + PCA whitening",
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_total": float(pca.explained_variance_ratio_.sum()),
    }


def _balanced_training_positions(
    labels: np.ndarray, limit_per_class: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    n = min(len(positive), len(negative), int(limit_per_class))
    if n < 2:
        raise ValueError("Balanced QML benchmark needs at least two samples per class")
    selected = np.concatenate(
        [rng.choice(positive, n, replace=False), rng.choice(negative, n, replace=False)]
    )
    return rng.permutation(selected)


def _calibrated_classical_svc(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    kernel,
    seed: int,
) -> np.ndarray:
    """Train-only cross-fitted sigmoid calibration for a matched SVC control."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC

    oof_score = np.full(len(y_train), np.nan, dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for inner_train, inner_cal in cv.split(x_train, y_train):
        model = SVC(kernel=kernel, C=1.0, class_weight="balanced")
        model.fit(x_train[inner_train], y_train[inner_train])
        oof_score[inner_cal] = model.decision_function(x_train[inner_cal])
    calibrator = LogisticRegression(solver="lbfgs", max_iter=500)
    calibrator.fit(oof_score.reshape(-1, 1), y_train)
    model = SVC(kernel=kernel, C=1.0, class_weight="balanced")
    model.fit(x_train, y_train)
    score = model.decision_function(x_val)
    return calibrator.predict_proba(score.reshape(-1, 1))[:, 1]


def _quantum_angle_coordinates(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the QSVM's train-fitted coordinate map for fair controls."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    train = np.tanh(scaler.fit_transform(x_train)) * np.pi
    val = np.tanh(scaler.transform(x_val)) * np.pi
    return train, val


def _classical_kernel(name: str):
    """Return a fixed classical kernel evaluated in the IQP angle coordinates."""
    if name in {"rbf", "poly"}:
        return name
    if name == "laplacian":
        from sklearn.metrics.pairwise import laplacian_kernel

        return lambda left, right: laplacian_kernel(
            left, right, gamma=1.0 / left.shape[1]
        )
    if name == "product_cosine":
        def product_cosine(left, right):
            delta = left[:, None, :] - right[None, :, :]
            return np.prod(np.cos(delta / 2.0) ** 2, axis=2)

        return product_cosine
    raise ValueError(f"Unknown classical kernel: {name}")


def _fingerprint(
    record_ids: np.ndarray,
    model_type: str,
    held_out_fold: int,
    qsvm_per_class: int,
) -> str:
    payload = (
        np.asarray(record_ids, dtype=np.int64).tobytes()
        + model_type.encode()
        + str(held_out_fold).encode()
        + str(qsvm_per_class).encode()
        + b"phase6q-a-schema-v2"
    )
    return hashlib.sha256(payload).hexdigest()


def _paired_patient_bootstrap(
    labels: np.ndarray,
    patient_ids: np.ndarray,
    quantum_probability: np.ndarray,
    classical_probability: np.ndarray,
    iterations: int,
    seed: int,
    comparison: str = "qsvm_minus_rbf_svc_z8",
) -> dict:
    """Paired uncertainty for QML minus classical control, clustered by patient."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    labels = np.asarray(labels, dtype=int)
    patient_ids = np.asarray(patient_ids)
    q = np.asarray(quantum_probability, dtype=float)
    c = np.asarray(classical_probability, dtype=float)
    unique_patients, patient_inverse = np.unique(patient_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    delta_auprc = []
    delta_auroc = []
    delta_brier = []
    for _ in range(int(iterations)):
        multiplicity = np.bincount(
            rng.integers(0, len(unique_patients), size=len(unique_patients)),
            minlength=len(unique_patients),
        )
        sample_weight = multiplicity[patient_inverse]
        present = sample_weight > 0
        if np.unique(labels[present]).size < 2:
            continue
        delta_auprc.append(
            average_precision_score(labels, q, sample_weight=sample_weight)
            - average_precision_score(labels, c, sample_weight=sample_weight)
        )
        delta_auroc.append(
            roc_auc_score(labels, q, sample_weight=sample_weight)
            - roc_auc_score(labels, c, sample_weight=sample_weight)
        )
        delta_brier.append(
            brier_score_loss(labels, q, sample_weight=sample_weight)
            - brier_score_loss(labels, c, sample_weight=sample_weight)
        )

    def summary(values: list[float]) -> dict:
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "ci95_low": float(np.quantile(array, 0.025)),
            "ci95_high": float(np.quantile(array, 0.975)),
        }

    auprc = summary(delta_auprc)
    return {
        "comparison": comparison,
        "bootstrap_unit": "patient",
        "iterations_requested": int(iterations),
        "iterations_valid": len(delta_auprc),
        "delta_auprc": auprc,
        "delta_auroc": summary(delta_auroc),
        "delta_brier": summary(delta_brier),
        "matched_kernel_accuracy_gate": (
            "PASS_MATCHED_KERNEL_ACCURACY_DELTA"
            if auprc["ci95_low"] > 0.0
            else "NO_MATCHED_KERNEL_ACCURACY_DELTA"
        ),
        "gate_rule": "95% patient-bootstrap CI for delta AUPRC must be entirely above zero",
        "claim_boundary": (
            "This gate compares predictive accuracy against one matched RBF-SVC on a "
            "classical simulator. It does not establish computational quantum advantage."
        ),
    }


class FocalLoss:
    """Asymmetric Focal Loss for imbalanced clinical classification."""
    def __init__(self, alpha: float = 0.65, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets, weights=None):
        import torch.nn.functional as F
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_factor * ((1.0 - p_t) ** self.gamma)
        loss = focal_weight * bce
        if weights is not None:
            loss = loss * weights
        return loss.mean()


def train_and_eval_quantum_fold(
    model_type: str,
    signals_arr: np.ndarray | None,
    metadata_df: pd.DataFrame,
    features_arr: np.ndarray,
    held_out_fold: int,
    device,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    qsvm_per_class: int = 500,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    folds = metadata_df.strat_fold.to_numpy(dtype=int)
    labels = metadata_df.mi_label.to_numpy(dtype=float)
    hard_neg = metadata_df.hard_negative.to_numpy(dtype=bool) if "hard_negative" in metadata_df else np.zeros(len(labels), dtype=bool)

    train_mask = folds != held_out_fold
    val_mask = folds == held_out_fold

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    X_train_tab_norm, X_val_tab_norm, z8_audit = _fold_local_z8(
        features_arr, train_indices, val_indices, seed + held_out_fold
    )

    start_time = time.perf_counter()

    if model_type == "qsvm":
        qsvm = QSVMClassifier(n_qubits=8, n_layers=2, C=1.0, seed=seed)
        sub_train = _balanced_training_positions(
            labels[train_indices], qsvm_per_class, seed + held_out_fold
        )
        qsvm.fit(X_train_tab_norm[sub_train], labels[train_indices][sub_train])
        p_val = qsvm.predict_proba(X_val_tab_norm)[:, 1]
        eval_time = (time.perf_counter() - start_time) * 1000.0 / len(val_indices)
        return val_indices, p_val, eval_time, {
            "z8": z8_audit,
            "training_records": int(len(sub_train)),
            "kernel": qsvm.kernel_diagnostics_,
            "simulator": qsvm.qke.dev.name,
        }

    elif model_type == "rbf_svc_z8":
        sub_train = _balanced_training_positions(
            labels[train_indices], qsvm_per_class, seed + held_out_fold
        )
        p_val = _calibrated_classical_svc(
            X_train_tab_norm[sub_train], labels[train_indices][sub_train],
            X_val_tab_norm, kernel="rbf", seed=seed + held_out_fold,
        )
        eval_time = (time.perf_counter() - start_time) * 1000.0 / len(val_indices)
        return val_indices, p_val, eval_time, {
            "z8": z8_audit,
            "training_records": int(len(sub_train)),
            "kernel": "rbf",
        }

    elif model_type in ANGLE_KERNEL_CONTROLS:
        sub_train = _balanced_training_positions(
            labels[train_indices], qsvm_per_class, seed + held_out_fold
        )
        angle_train, angle_val = _quantum_angle_coordinates(
            X_train_tab_norm[sub_train], X_val_tab_norm
        )
        kernel_name = ANGLE_KERNEL_CONTROLS[model_type]
        p_val = _calibrated_classical_svc(
            angle_train,
            labels[train_indices][sub_train],
            angle_val,
            kernel=_classical_kernel(kernel_name),
            seed=seed + held_out_fold,
        )
        eval_time = (time.perf_counter() - start_time) * 1000.0 / len(val_indices)
        return val_indices, p_val, eval_time, {
            "z8": z8_audit,
            "training_records": int(len(sub_train)),
            "kernel": kernel_name,
            "coordinate_map": "training-only StandardScaler -> tanh -> pi",
        }

    elif model_type == "vqc":
        X_train_tab_t = torch.tensor(X_train_tab_norm, dtype=torch.float32)
        X_val_tab_t = torch.tensor(X_val_tab_norm, dtype=torch.float32)
        y_train_t = torch.tensor(labels[train_indices], dtype=torch.float32)
        w_train = np.ones(len(train_indices), dtype=np.float32)
        w_train[(labels[train_indices] == 0) & hard_neg[train_indices]] = 1.5
        w_train_t = torch.tensor(w_train, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_tab_t, y_train_t, w_train_t),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(TensorDataset(X_val_tab_t), batch_size=batch_size, shuffle=False)

        # Analytic PennyLane simulation is CPU execution.  Keeping the entire
        # small z8 VQC on CPU avoids CUDA tensors crossing into a CPU QNode.
        device = torch.device("cpu")
        model = VariationalQuantumClassifier(
            in_features=8, n_qubits=8, n_layers=3
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = FocalLoss(alpha=0.65, gamma=2.0)

        model.train()
        for epoch in range(epochs):
            for b_tab, b_y, b_w in train_loader:
                b_tab, b_y, b_w = b_tab.to(device), b_y.to(device), b_w.to(device)
                optimizer.zero_grad()
                logits = model(b_tab)
                loss = criterion(logits, b_y, b_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for (b_tab,) in val_loader:
                b_tab = b_tab.to(device)
                logits = model(b_tab)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
        eval_time = (time.perf_counter() - start_time) * 1000.0 / len(val_indices)
        p_val = np.concatenate(val_preds, axis=0)
        return val_indices, p_val, eval_time, {
            "z8": z8_audit, "training_records": int(len(train_indices))
        }

    elif model_type == "hqnn":
        if signals_arr is None:
            raise ValueError("HQNN requires --h5-path")
        X_train_sig_t = torch.tensor(signals_arr[train_indices], dtype=torch.float32)
        X_val_sig_t = torch.tensor(signals_arr[val_indices], dtype=torch.float32)
        X_train_tab_t = torch.tensor(X_train_tab_norm, dtype=torch.float32)
        X_val_tab_t = torch.tensor(X_val_tab_norm, dtype=torch.float32)
        y_train_t = torch.tensor(labels[train_indices], dtype=torch.float32)
        w_train = np.ones(len(train_indices), dtype=np.float32)
        w_train[(labels[train_indices] == 0) & hard_neg[train_indices]] = 1.5
        w_train_t = torch.tensor(w_train, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_sig_t, X_train_tab_t, y_train_t, w_train_t),
            batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
            TensorDataset(X_val_sig_t, X_val_tab_t),
            batch_size=batch_size, shuffle=False
        )

        device = torch.device("cpu")
        model = HybridQuantumNeuralNetwork(
            tabular_dim=8,
            raw_channels=12,
            n_qubits=8,
            n_quantum_layers=3,
            resnet_base_filters=32,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = FocalLoss(alpha=0.65, gamma=2.0)

        model.train()
        for epoch in range(epochs):
            for b_sig, b_tab, b_y, b_w in train_loader:
                b_sig, b_tab, b_y, b_w = b_sig.to(device), b_tab.to(device), b_y.to(device), b_w.to(device)
                optimizer.zero_grad()
                logits = model(b_sig, b_tab)
                loss = criterion(logits, b_y, b_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        model.eval()
        val_preds = []
        with torch.no_grad():
            for b_sig, b_tab in val_loader:
                b_sig, b_tab = b_sig.to(device), b_tab.to(device)
                logits = model(b_sig, b_tab)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
        eval_time = (time.perf_counter() - start_time) * 1000.0 / len(val_indices)
        p_val = np.concatenate(val_preds, axis=0)
        return val_indices, p_val, eval_time, {
            "z8": z8_audit, "training_records": int(len(train_indices)),
            "warning": "analytic CPU simulation; HQNN remains experimental",
        }

    else:
        raise ValueError(f"Unknown quantum model type: {model_type}")


def run_quantum_baselines(
    h5_path: Path | None,
    metadata_path: Path,
    features_csv: Path,
    manifest_path: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    models: tuple[str, ...] = ("qsvm", "rbf_svc_z8"),
    qsvm_per_class: int = 500,
    resume: bool = True,
    bootstrap_iterations: int = 2000,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Torch accelerator available: {device}; quantum kernels use an analytic CPU simulator",
        flush=True,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata and approved features
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")
    features_raw = pd.read_csv(features_csv).set_index("ecg_id")

    manifest = load_feature_manifest(manifest_path, features_raw.columns)
    approved = manifest["approved_features"]
    features_raw = features_raw[approved]

    joined = metadata.join(features_raw, how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")]

    labels = joined.mi_label.to_numpy(dtype=int)
    folds = joined.strat_fold.to_numpy(dtype=int)
    patient_ids = joined.patient_id.to_numpy()
    record_ids = joined.index.to_numpy()
    hard_neg = (
        joined.hard_negative.to_numpy(dtype=bool)
        if "hard_negative" in joined.columns
        else np.zeros(len(joined), dtype=bool)
    )

    guard_fold_access(folds, purpose="tuning")
    feature_frame = joined[approved].select_dtypes(include=[np.number]).replace(
        [np.inf, -np.inf], np.nan
    )
    features_mat = feature_frame.to_numpy(dtype=np.float32)

    raw_signals = None
    if "hqnn" in models:
        if h5_path is None:
            raise ValueError("--h5-path is required when hqnn is selected")
        import h5py
        print("Pre-loading 12-lead raw signals for HQNN ...", flush=True)
        with h5py.File(h5_path, "r") as h5:
            h5_ids = np.asarray(h5["ecg_id"])
            id_to_idx = {int(eid): idx for idx, eid in enumerate(h5_ids)}
            ordered_indices = np.array([id_to_idx[eid] for eid in record_ids], dtype=int)
            raw_signals = h5["accepted_signal"][ordered_indices].astype(np.float32)
            if raw_signals.shape[1] == 1000 and raw_signals.shape[2] == 12:
                raw_signals = np.transpose(raw_signals, (0, 2, 1))
            print(f"Loaded raw signals shape: {raw_signals.shape}", flush=True)

    model_types = list(models)
    unknown = sorted(set(model_types) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}; choose from {SUPPORTED_MODELS}")
    all_predictions = []
    all_metrics = []
    probabilities_by_model: dict[str, np.ndarray] = {}

    for model_type in model_types:
        print(f"\n==========================================", flush=True)
        print(f"Training 8-Fold OOF for: {model_type.upper()}", flush=True)
        print(f"==========================================", flush=True)

        prob_oof = np.full(len(joined), np.nan, dtype=float)
        total_latency = 0.0
        fold_audits = []

        for held_out in sorted(np.unique(folds)):
            print(f"  [Fold {held_out}/8] Training on folds {[f for f in DEV_FOLDS if f != held_out]} ...", flush=True)
            checkpoint_dir = output_dir / "checkpoints" / model_type
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = checkpoint_dir / f"fold_{int(held_out)}.npz"
            expected_fingerprint = _fingerprint(
                record_ids, model_type, int(held_out), qsvm_per_class
            )
            loaded_checkpoint = False
            if resume and checkpoint.exists():
                saved = np.load(checkpoint, allow_pickle=False)
                if str(saved["fingerprint"].item()) == expected_fingerprint:
                    val_idx = saved["val_idx"].astype(int)
                    p_val = saved["probability"].astype(float)
                    latency = float(saved["latency_ms"].item())
                    audit = json.loads(str(saved["audit_json"].item()))
                    loaded_checkpoint = True
                    print(f"    resumed {checkpoint.name}", flush=True)
                else:
                    print(
                        f"    ignored stale {checkpoint.name}; configuration changed",
                        flush=True,
                    )
            if not loaded_checkpoint:
                val_idx, p_val, latency, audit = train_and_eval_quantum_fold(
                    model_type=model_type,
                    signals_arr=raw_signals,
                    metadata_df=joined,
                    features_arr=features_mat,
                    held_out_fold=int(held_out),
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    seed=seed,
                    qsvm_per_class=qsvm_per_class,
                )
                temporary = checkpoint.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle, fingerprint=expected_fingerprint, val_idx=val_idx,
                        probability=p_val, latency_ms=latency,
                        audit_json=json.dumps(audit, sort_keys=True),
                    )
                temporary.replace(checkpoint)
            prob_oof[val_idx] = p_val
            total_latency += latency
            fold_audits.append({"held_out_fold": int(held_out), **audit})

        avg_latency = total_latency / 8.0

        if not np.isfinite(prob_oof).all():
            raise RuntimeError(f"{model_type} produced incomplete OOF predictions")
        # QSVM and its matched RBF control are calibrated using only the outer
        # training data inside each fold.  Neural QML probabilities are kept as
        # raw development probabilities until a nested calibrator is added.
        calibrated_prob = prob_oof
        probabilities_by_model[model_type] = calibrated_prob.copy()
        (output_dir / f"{model_type}_fold_diagnostics.json").write_text(
            json.dumps(fold_audits, indent=2)
        )

        metrics = evaluate_probabilities(
            labels, calibrated_prob, model_type, avg_latency, hard_negative=hard_neg,
        )

        metrics_dict = asdict(metrics)
        metrics_dict["model_family"] = (
            f"quantum_{model_type}" if model_type in {"qsvm", "vqc", "hqnn"}
            else f"matched_classical_{model_type}"
        )
        metrics_dict["feature_subset"] = "fold_local_pca_z8"
        metrics_dict["calibration"] = (
            "inner_5fold_platt" if model_type == "qsvm" or model_type.endswith("_z8")
            else "none_development_only"
        )
        metrics_dict["mean_latency_ms"] = round(avg_latency, 4)
        all_metrics.append(metrics_dict)

        print(f"[{model_type.upper()} OOF RESULTS]")
        print(f"  AUPRC: {metrics.auprc:.4f}")
        print(f"  AUROC: {metrics.auroc:.4f}")
        print(f"  Sensitivity @ 90% Spec: {metrics.sensitivity_at_90_specificity:.4f}")
        print(f"  F1 Score @ 0.5: {metrics.f1_at_05:.4f}")
        print(f"  ECE: {metrics.calibration_error:.4f}")
        print(f"  MI vs hard-negative AUROC: {metrics.mi_vs_hard_neg_auroc:.4f}")
        print(f"  Latency: {avg_latency:.3f} ms/rec")

        for idx, (rec_id, pat_id, fold, raw_p, cal_p, y, hn) in enumerate(
            zip(record_ids, patient_ids, folds, prob_oof, calibrated_prob, labels, hard_neg)
        ):
            all_predictions.append({
                "ecg_id": int(rec_id),
                "patient_id": int(pat_id),
                "strat_fold": int(fold),
                "model_family": (
                    f"quantum_{model_type}"
                    if model_type in {"qsvm", "vqc", "hqnn"}
                    else f"matched_classical_{model_type}"
                ),
                "feature_subset": "fold_local_pca_z8",
                "y_true": int(y),
                "raw_probability": float(raw_p),
                "calibrated_probability": float(cal_p),
                "hard_negative": bool(hn),
            })

    # Save benchmark outputs
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = output_dir / "quantum_benchmark_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved metrics summary to: {metrics_csv}")

    preds_df = pd.DataFrame(all_predictions)
    preds_csv = output_dir / "quantum_oof_predictions.csv"
    preds_df.to_csv(preds_csv, index=False)
    print(f"Saved OOF predictions to: {preds_csv}")

    if "qsvm" in probabilities_by_model:
        for control_name, control_probability in probabilities_by_model.items():
            if control_name == "qsvm" or control_name in {"vqc", "hqnn"}:
                continue
            comparison = _paired_patient_bootstrap(
                labels=labels,
                patient_ids=patient_ids,
                quantum_probability=probabilities_by_model["qsvm"],
                classical_probability=control_probability,
                iterations=bootstrap_iterations,
                seed=seed,
                comparison=f"qsvm_minus_{control_name}",
            )
            comparison_path = output_dir / f"paired_qsvm_vs_{control_name}_bootstrap.json"
            comparison_path.write_text(json.dumps(comparison, indent=2))
            print(
                "Paired matched-kernel accuracy gate: "
                f"{comparison['matched_kernel_accuracy_gate']} -> {comparison_path}",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 6Q Quantum ML Baselines")
    parser.add_argument("--h5-path", type=Path)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("quantum_benchmark_outputs"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models", nargs="+", choices=SUPPORTED_MODELS,
        default=["qsvm", "rbf_svc_z8"],
    )
    parser.add_argument("--qsvm-per-class", type=int, default=500)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    run_quantum_baselines(
        h5_path=args.h5_path,
        metadata_path=args.metadata_path,
        features_csv=args.features_csv,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        models=tuple(args.models),
        qsvm_per_class=args.qsvm_per_class,
        resume=not args.no_resume,
        bootstrap_iterations=args.bootstrap_iterations,
    )
