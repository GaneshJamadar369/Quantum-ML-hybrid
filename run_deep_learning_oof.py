"""8-Fold Patient-Safe OOF Deep Learning and Multimodal Hybrid Training Engine."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from aquire_preprocessing.baselines import (
    FoldLocalPlattCalibrator,
    evaluate_probabilities,
)
from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.dataset import HDF5ECGDataset
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access


class FocalLoss:
    """Asymmetric Focal Loss for imbalanced clinical classification."""

    def __init__(self, alpha: float = 0.65, gamma: float = 2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets, weights=None):
        import torch
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


def train_and_eval_fold(
    model_type: str,
    signals_arr: np.ndarray,
    metadata_df: pd.DataFrame,
    features_arr: np.ndarray,
    held_out_fold: int,
    device,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 5e-4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from aquire_preprocessing.models_1d import ECGResNet1D
    from aquire_preprocessing.models_hybrid import ECGMultimodalHybrid

    torch.manual_seed(seed)
    np.random.seed(seed)

    folds = metadata_df.strat_fold.to_numpy(dtype=int)
    labels = metadata_df.mi_label.to_numpy(dtype=float)
    hard_neg = metadata_df.hard_negative.to_numpy(dtype=bool) if "hard_negative" in metadata_df else np.zeros(len(labels), dtype=bool)

    train_mask = folds != held_out_fold
    val_mask = folds == held_out_fold

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    # Signal tensors: guaranteed (N, 12, 1000)
    X_train_sig = torch.tensor(signals_arr[train_indices], dtype=torch.float32)
    X_val_sig = torch.tensor(signals_arr[val_indices], dtype=torch.float32)

    y_train = torch.tensor(labels[train_indices], dtype=torch.float32)
    y_val = torch.tensor(labels[val_indices], dtype=torch.float32)

    w_train = np.ones(len(train_indices), dtype=np.float32)
    w_train[(labels[train_indices] == 0) & hard_neg[train_indices]] = 1.5
    w_train = torch.tensor(w_train, dtype=torch.float32)

    X_train_tab = torch.tensor(features_arr[train_indices], dtype=torch.float32)
    X_val_tab = torch.tensor(features_arr[val_indices], dtype=torch.float32)

    # Standardize tabular features with training-fold statistics
    mean_tab = X_train_tab.mean(dim=0, keepdim=True)
    std_tab = X_train_tab.std(dim=0, keepdim=True).clamp(min=1e-5)
    X_train_tab = (X_train_tab - mean_tab) / std_tab
    X_val_tab = (X_val_tab - mean_tab) / std_tab

    if model_type == "hybrid":
        train_loader = DataLoader(
            TensorDataset(X_train_sig, X_train_tab, y_train, w_train),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )
        model = ECGMultimodalHybrid(
            num_tabular_features=features_arr.shape[1],
            tabular_hidden=64,
            waveform_channels=12,
            waveform_embedding_dim=128,
            fused_dim=128,
        ).to(device)
    else:
        train_loader = DataLoader(
            TensorDataset(X_train_sig, y_train, w_train),
            batch_size=batch_size, shuffle=True, drop_last=True,
        )
        model = ECGResNet1D(
            in_channels=12, base_filters=32, embedding_dim=128,
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=0.65, gamma=2.0)

    # Training loop
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            if model_type == "hybrid":
                b_sig, b_tab, b_y, b_w = [t.to(device) for t in batch]
                logits, _ = model(b_sig, b_tab)
            else:
                b_sig, b_y, b_w = [t.to(device) for t in batch]
                logits, _ = model(b_sig)
            loss = criterion(logits, b_y, b_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

    # Validation evaluation
    model.eval()
    val_preds = []
    val_loader = DataLoader(
        TensorDataset(X_val_sig, X_val_tab) if model_type == "hybrid" else TensorDataset(X_val_sig),
        batch_size=batch_size, shuffle=False,
    )
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in val_loader:
            if model_type == "hybrid":
                b_sig, b_tab = [t.to(device) for t in batch]
                logits, _ = model(b_sig, b_tab)
            else:
                b_sig = batch[0].to(device)
                logits, _ = model(b_sig)
            val_preds.append(torch.sigmoid(logits).cpu().numpy())
    total_val_time = (time.perf_counter() - start_time) * 1000.0 / len(val_indices)

    p_val = np.concatenate(val_preds, axis=0)
    return val_indices, p_val, total_val_time


def run_deep_learning_baselines(
    h5_path: Path,
    metadata_path: Path,
    features_csv: Path,
    manifest_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 5e-4,
    seed: int = 42,
) -> None:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deep learning execution device: {device}", flush=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and align datasets
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

    # Pre-load 12-lead signals aligned by record_id
    import h5py
    print("Pre-loading 12-lead raw signals from HDF5 ...", flush=True)
    with h5py.File(h5_path, "r") as h5:
        h5_ids = np.asarray(h5["ecg_id"])
        id_to_idx = {int(eid): idx for idx, eid in enumerate(h5_ids)}
        ordered_indices = np.array([id_to_idx[eid] for eid in record_ids], dtype=int)
        raw_signals = h5["accepted_signal"][ordered_indices].astype(np.float32)
        # Transpose if shape is (N, 1000, 12) -> (N, 12, 1000)
        if raw_signals.shape[1] == 1000 and raw_signals.shape[2] == 12:
            raw_signals = np.transpose(raw_signals, (0, 2, 1))
        print(f"Loaded raw signals shape: {raw_signals.shape} ({raw_signals.nbytes / 1024 / 1024:.1f} MB)", flush=True)

    model_types = ["resnet1d", "hybrid"]
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
            val_idx, p_val, latency = train_and_eval_fold(
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
        all_metrics.append(asdict(metrics))

        all_predictions.append(pd.DataFrame({
            "ecg_id": record_ids,
            "patient_id": patient_ids,
            "fold": folds,
            "label": labels,
            "model": model_type,
            "raw_probability": prob_oof,
            "calibrated_probability": calibrated_prob,
            "hard_negative": hard_neg,
        }))

    metrics_df = pd.DataFrame(all_metrics).sort_values(["auprc", "brier"], ascending=[False, True])
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    metrics_df.to_csv(output_dir / "deep_learning_oof_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "deep_learning_oof_predictions.csv", index=False)

    champion = metrics_df.iloc[0].to_dict()
    (output_dir / "provisional_deep_learning_champion.json").write_text(json.dumps(champion, indent=2))

    print("\nDeep Learning Benchmark Summary:", flush=True)
    print(metrics_df[["model", "auprc", "auroc", "sensitivity_at_90_specificity", "hard_neg_fpr", "brier"]].to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6B & 6C Deep Learning and Multimodal Hybrid Trainer")
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--approved-feature-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/g6_deep_learning"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_deep_learning_baselines(
        h5_path=args.hdf5,
        metadata_path=args.metadata,
        features_csv=args.features,
        manifest_path=args.approved_feature_manifest,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
