"""
8-Fold OOF Quantum Machine Learning Training and Evaluation Engine (Phase 6Q).
Trains and evaluates:
1. QSVM (Quantum Support Vector Machine with PennyLane Quantum Kernel)
2. VQC (Variational Quantum Classifier with StronglyEntanglingLayers)
3. HQNN (Hybrid Quantum Neural Network with 1D-ResNet + MLP + Quantum Bottleneck)
Performs out-of-fold Platt calibration, Hard-Negative sensitivity analysis, and generates comparative metrics.
"""

from __future__ import annotations

import argparse
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
    FoldLocalPlattCalibrator,
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
    signals_arr: np.ndarray,
    metadata_df: pd.DataFrame,
    features_arr: np.ndarray,
    held_out_fold: int,
    device,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    folds = metadata_df.strat_fold.to_numpy(dtype=int)
    labels = metadata_df.mi_label.to_numpy(dtype=float)
    hard_neg = metadata_df.hard_negative.to_numpy(dtype=bool) if "hard_negative" in metadata_df else np.zeros(len(labels), dtype=bool)

    train_mask = folds != held_out_fold
    val_mask = folds == held_out_fold

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    # Tabular features standardization
    X_train_tab_raw = features_arr[train_indices]
    X_val_tab_raw = features_arr[val_indices]

    mean_tab = np.mean(X_train_tab_raw, axis=0, keepdims=True)
    std_tab = np.std(X_train_tab_raw, axis=0, keepdims=True)
    std_tab[std_tab < 1e-5] = 1.0

    X_train_tab_norm = (X_train_tab_raw - mean_tab) / std_tab
    X_val_tab_norm = (X_val_tab_raw - mean_tab) / std_tab

    start_time = time.perf_counter()

    if model_type == "qsvm":
        qsvm = QSVMClassifier(n_qubits=8, n_layers=2, C=1.0)
        
        pos_idx = np.where(labels[train_indices] == 1)[0]
        neg_idx = np.where(labels[train_indices] == 0)[0]
        n_sub = min(len(pos_idx), 500)
        sub_pos = np.random.choice(pos_idx, n_sub, replace=False)
        sub_neg = np.random.choice(neg_idx, n_sub, replace=False)
        sub_train = np.concatenate([sub_pos, sub_neg])
        
        qsvm.fit(X_train_tab_norm[sub_train], labels[train_indices[sub_train]])
        p_val = qsvm.predict_proba(X_val_tab_norm)[:, 1]
        eval_time = (time.perf_counter() - start_time) * 1000.0 / len(val_indices)
        return val_indices, p_val, eval_time

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

        model = VariationalQuantumClassifier(
            in_features=features_arr.shape[1], n_qubits=8, n_layers=3
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
        return val_indices, p_val, eval_time

    elif model_type == "hqnn":
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

        model = HybridQuantumNeuralNetwork(
            tabular_dim=features_arr.shape[1],
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
        return val_indices, p_val, eval_time

    else:
        raise ValueError(f"Unknown quantum model type: {model_type}")


def run_quantum_baselines(
    h5_path: Path,
    metadata_path: Path,
    features_csv: Path,
    manifest_path: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Quantum ML execution device: {device}", flush=True)

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
    features_mat = joined[approved].select_dtypes(include=[np.number]).fillna(0.0).to_numpy(dtype=np.float32)

    # Pre-load 12-lead signals
    import h5py
    print("Pre-loading 12-lead raw signals from HDF5 ...", flush=True)
    with h5py.File(h5_path, "r") as h5:
        h5_ids = np.asarray(h5["ecg_id"])
        id_to_idx = {int(eid): idx for idx, eid in enumerate(h5_ids)}
        ordered_indices = np.array([id_to_idx[eid] for eid in record_ids], dtype=int)
        raw_signals = h5["accepted_signal"][ordered_indices].astype(np.float32)
        if raw_signals.shape[1] == 1000 and raw_signals.shape[2] == 12:
            raw_signals = np.transpose(raw_signals, (0, 2, 1))
        print(f"Loaded raw signals shape: {raw_signals.shape} ({raw_signals.nbytes / 1024 / 1024:.1f} MB)", flush=True)

    model_types = ["qsvm", "vqc", "hqnn"]
    all_predictions = []
    all_metrics = []

    for model_type in model_types:
        print(f"\n==========================================", flush=True)
        print(f"Training 8-Fold OOF for: {model_type.upper()}", flush=True)
        print(f"==========================================", flush=True)

        prob_oof = np.full(len(joined), np.nan, dtype=float)
        total_latency = 0.0

        for held_out in sorted(np.unique(folds)):
            print(f"  [Fold {held_out}/8] Training on folds {[f for f in DEV_FOLDS if f != held_out]} ...", flush=True)
            val_idx, p_val, latency = train_and_eval_quantum_fold(
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
            )
            prob_oof[val_idx] = p_val
            total_latency += latency

        avg_latency = total_latency / 8.0

        # Leave-One-Fold-Out Platt Sigmoid Calibration
        logits_oof = np.log(np.clip(prob_oof, 1e-7, 1 - 1e-7) / np.clip(1 - prob_oof, 1e-7, 1))
        calibrator = FoldLocalPlattCalibrator()
        calibrator.fit_from_oof_logits(logits_oof, labels, folds)
        calibrated_prob = calibrator.transform(logits_oof, folds)

        metrics = evaluate_probabilities(
            labels, calibrated_prob, model_type, avg_latency, hard_negative=hard_neg,
        )

        metrics_dict = asdict(metrics)
        metrics_dict["model_family"] = f"quantum_{model_type}"
        metrics_dict["feature_subset"] = "quantum_top8_multimodal"
        metrics_dict["mean_latency_ms"] = round(avg_latency, 4)
        all_metrics.append(metrics_dict)

        print(f"[{model_type.upper()} OOF RESULTS]")
        print(f"  AUPRC: {metrics.auprc:.4f}")
        print(f"  AUROC: {metrics.auroc:.4f}")
        print(f"  Sensitivity @ 90% Spec: {metrics.sens_at_90_spec:.4f}")
        print(f"  F1 Score: {metrics.f1_optimal:.4f}")
        print(f"  ECE: {metrics.expected_calibration_error:.4f}")
        print(f"  Hard-Neg AUROC: {metrics.hard_neg_auroc:.4f}")
        print(f"  Latency: {avg_latency:.3f} ms/rec")

        for idx, (rec_id, pat_id, fold, raw_p, cal_p, y, hn) in enumerate(
            zip(record_ids, patient_ids, folds, prob_oof, calibrated_prob, labels, hard_neg)
        ):
            all_predictions.append({
                "ecg_id": int(rec_id),
                "patient_id": int(pat_id),
                "strat_fold": int(fold),
                "model_family": f"quantum_{model_type}",
                "feature_subset": "quantum_top8_multimodal",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 6Q Quantum ML Baselines")
    parser.add_argument("--h5-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("quantum_benchmark_outputs"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

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
    )
