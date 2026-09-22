"""Nested-epoch, patient-safe ECG patch-Transformer representation export.

For each outer development fold, an inner patient split chooses an epoch count.
A fresh model then trains on all seven outer-training folds for exactly that
many epochs.  The same frozen encoder exports its train and held-out vectors.
Outer labels are used only for the final OOF evaluation, never stopping.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_transformer import ECGPatchTransformer
from run_waveform_representation_export import (
    _atomic_npz,
    _embed,
    _load_normalizer,
    _model_checksum,
    _normalise,
    _seed_everything,
    patient_balanced_weights,
)


def stratified_patient_inner_split(
    metadata: pd.DataFrame, outer_train: np.ndarray, seed: int, fraction: float = 0.10
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint 90/10 patient split, stratified by patient-ever-MI status."""
    if not 0 < fraction < 0.5:
        raise ValueError("Inner validation fraction must be between 0 and 0.5")
    patients = metadata.iloc[outer_train].groupby("patient_id").mi_label.max()
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        ids = patients.index[patients.eq(label)].to_numpy()
        if len(ids) < 2:
            raise ValueError(f"Too few patient groups for inner class {label}")
        count = max(1, min(len(ids) - 1, int(round(fraction * len(ids)))))
        selected.extend(rng.permutation(ids)[:count].tolist())
    inner_mask = metadata.iloc[outer_train].patient_id.isin(selected).to_numpy()
    fit_idx, val_idx = outer_train[~inner_mask], outer_train[inner_mask]
    if set(metadata.iloc[fit_idx].patient_id) & set(metadata.iloc[val_idx].patient_id):
        raise RuntimeError("Inner patient split overlaps")
    if metadata.iloc[val_idx].mi_label.nunique() != 2:
        raise RuntimeError("Inner validation is single-class")
    return fit_idx, val_idx


def _augment_ecg(batch, *, gain_range: float = 0.05, noise_std: float = 0.01):
    """Small voltage perturbations; no beat deletion or time warping."""
    import torch

    gain = torch.empty((len(batch), batch.shape[1], 1), device=batch.device).uniform_(
        1.0 - gain_range, 1.0 + gain_range
    )
    return batch * gain + noise_std * torch.randn_like(batch)


def _fit_epoch_schedule(
    signals: np.ndarray,
    metadata: pd.DataFrame,
    fit_idx: np.ndarray,
    inner_val_idx: np.ndarray | None,
    *,
    max_epochs: int,
    fixed_epochs: int | None,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device,
    patience: int = 4,
    min_epochs: int = 6,
):
    import torch
    from sklearn.metrics import average_precision_score
    from torch.utils.data import DataLoader, TensorDataset

    from run_deep_learning_oof import FocalLoss

    if max_epochs < min_epochs:
        raise ValueError("max_epochs must meet the minimum early-stopping epoch")
    _seed_everything(seed)
    labels = metadata.mi_label.to_numpy(dtype=np.float32)
    patients = metadata.patient_id.to_numpy()
    hard = metadata.hard_negative.to_numpy(dtype=bool)
    weights = patient_balanced_weights(patients[fit_idx], labels[fit_idx], hard[fit_idx])
    dataset = TensorDataset(
        torch.from_numpy(signals[fit_idx]),
        torch.from_numpy(labels[fit_idx]),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    model = ECGPatchTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    total_epochs = int(fixed_epochs or max_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=learning_rate * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = FocalLoss(alpha=0.65, gamma=2.0)
    history = []
    best_ap = -np.inf
    best_epoch = None
    stalled = 0
    for epoch in range(1, total_epochs + 1):
        model.train()
        loss_sum, seen, last_grad = 0.0, 0, 0.0
        for signal, target, weight in loader:
            signal = signal.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            signal = _augment_ecg(signal)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, _ = model(signal)
                loss = criterion(logits, target, weight)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Transformer training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            last_grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not np.isfinite(last_grad):
                raise RuntimeError("Non-finite Transformer gradient")
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(signal)
            seen += len(signal)
        scheduler.step()
        train_loss = loss_sum / max(seen, 1)
        inner_ap = None
        if inner_val_idx is not None:
            inner_logits = _embed(model, signals[inner_val_idx], device, batch_size)[1]
            if not np.isfinite(inner_logits).all():
                raise RuntimeError("Non-finite inner validation logits")
            inner_ap = float(average_precision_score(labels[inner_val_idx], inner_logits))
            if epoch >= min_epochs:
                if inner_ap > best_ap + 1e-4:
                    best_ap, best_epoch, stalled = inner_ap, epoch, 0
                else:
                    stalled += 1
        record = {
            "epoch": epoch,
            "training_loss": float(train_loss),
            "inner_auprc": inner_ap,
            "gradient_norm_last_batch": last_grad,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"    epoch {epoch:02d}/{total_epochs}: loss={train_loss:.5f}, "
            f"inner_ap={inner_ap if inner_ap is not None else 'retrain'}",
            flush=True,
        )
        if inner_val_idx is not None and epoch >= min_epochs and stalled >= patience:
            break
    return model, history, (int(best_epoch) if best_epoch is not None else total_epochs)


def export_representations(
    hdf5_path: Path,
    metadata_path: Path,
    normalizer_dir: Path,
    output_dir: Path,
    *,
    max_epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 3e-4,
    seed: int = 20260922,
) -> None:
    import h5py
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata.strat_fold.isin(DEV_FOLDS) & metadata.eligibility.eq("PRIMARY")].copy()
    metadata = metadata.set_index("ecg_id", drop=False)
    folds = metadata.strat_fold.to_numpy(dtype=int)
    guard_fold_access(folds, purpose="tuning")
    if metadata.groupby("patient_id").strat_fold.nunique().max() != 1:
        raise ValueError("Patient crosses development folds")
    with h5py.File(hdf5_path, "r") as handle:
        h5_ids = np.asarray(handle["ecg_id"], dtype=int)
        id_to_index = {int(ecg_id): index for index, ecg_id in enumerate(h5_ids)}
        ordered = np.asarray([id_to_index[int(ecg_id)] for ecg_id in metadata.ecg_id])
        signals = handle["accepted_signal"][ordered].astype(np.float32)
    if signals.shape[1:] == (1000, 12):
        signals = np.transpose(signals, (0, 2, 1))
    if signals.shape[1:] != (12, 1000) or not np.isfinite(signals).all():
        raise ValueError(f"Invalid waveform tensor {signals.shape}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Transformer representation device: {device}; records={len(signals)}", flush=True)
    labels = metadata.mi_label.to_numpy(dtype=int)
    oof_embeddings = np.full((len(metadata), 128), np.nan, dtype=np.float32)
    oof_logits = np.full(len(metadata), np.nan, dtype=np.float32)
    audits = []
    for held_out in sorted(np.unique(folds)):
        artifact = output_dir / f"outer_fold_{held_out}_representations.npz"
        audit_path = output_dir / f"outer_fold_{held_out}_audit.json"
        checkpoint = output_dir / f"outer_fold_{held_out}_encoder.pt"
        if artifact.exists() and audit_path.exists() and checkpoint.exists():
            saved = np.load(artifact, allow_pickle=False)
            val_idx = saved["val_indices"].astype(int)
            oof_embeddings[val_idx] = saved["val_embeddings"]
            oof_logits[val_idx] = saved["val_logits"]
            audits.append(json.loads(audit_path.read_text()))
            print(f"outer fold {held_out}: resumed", flush=True)
            continue
        outer_train = np.flatnonzero(folds != held_out)
        outer_val = np.flatnonzero(folds == held_out)
        inner_fit, inner_val = stratified_patient_inner_split(metadata, outer_train, seed + int(held_out))
        normalizer_path = normalizer_dir / f"normalizer_holdout_fold_{held_out}.json"
        medians, iqrs, normalizer_audit = _load_normalizer(normalizer_path)
        expected_folds = sorted(int(x) for x in np.unique(folds[outer_train]))
        if normalizer_audit.get("folds_used") != expected_folds:
            raise ValueError(f"Fold-local normalizer mismatch in outer fold {held_out}")
        fold_signals = _normalise(signals, medians, iqrs)
        print(
            f"outer fold {held_out}: inner_fit={len(inner_fit)}, inner_val={len(inner_val)}, "
            f"outer_train={len(outer_train)}, outer_val={len(outer_val)}",
            flush=True,
        )
        _, inner_history, selected_epoch = _fit_epoch_schedule(
            fold_signals, metadata, inner_fit, inner_val, max_epochs=max_epochs,
            fixed_epochs=None, batch_size=batch_size, learning_rate=learning_rate,
            seed=seed + int(held_out), device=device,
        )
        model, retrain_history, _ = _fit_epoch_schedule(
            fold_signals, metadata, outer_train, None, max_epochs=max_epochs,
            fixed_epochs=selected_epoch, batch_size=batch_size, learning_rate=learning_rate,
            seed=seed + 1000 + int(held_out), device=device,
        )
        train_embeddings, train_logits = _embed(model, fold_signals[outer_train], device, batch_size)
        val_embeddings, val_logits = _embed(model, fold_signals[outer_val], device, batch_size)
        if not all(np.isfinite(x).all() for x in (train_embeddings, train_logits, val_embeddings, val_logits)):
            raise RuntimeError("Non-finite Transformer representation")
        oof_embeddings[outer_val] = val_embeddings.astype(np.float32)
        oof_logits[outer_val] = val_logits.astype(np.float32)
        state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        model_hash = _model_checksum(state_dict)
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save({"state_dict": state_dict, "held_out_fold": int(held_out),
                    "selected_epoch": selected_epoch, "model_checksum": model_hash}, temporary)
        temporary.replace(checkpoint)
        _atomic_npz(
            artifact,
            train_indices=outer_train,
            val_indices=outer_val,
            train_record_ids=metadata.ecg_id.to_numpy()[outer_train],
            val_record_ids=metadata.ecg_id.to_numpy()[outer_val],
            train_patient_ids=metadata.patient_id.to_numpy()[outer_train],
            val_patient_ids=metadata.patient_id.to_numpy()[outer_val],
            train_labels=labels[outer_train],
            val_labels=labels[outer_val],
            train_embeddings=train_embeddings.astype(np.float32),
            val_embeddings=val_embeddings.astype(np.float32),
            train_logits=train_logits.astype(np.float32),
            val_logits=val_logits.astype(np.float32),
        )
        audit = {
            "held_out_fold": int(held_out),
            "training_folds": expected_folds,
            "training_records": int(len(outer_train)),
            "validation_records": int(len(outer_val)),
            "training_patients": int(metadata.iloc[outer_train].patient_id.nunique()),
            "validation_patients": int(metadata.iloc[outer_val].patient_id.nunique()),
            "patient_overlap": int(len(set(metadata.iloc[outer_train].patient_id) & set(metadata.iloc[outer_val].patient_id))),
            "inner_fit_patients": int(metadata.iloc[inner_fit].patient_id.nunique()),
            "inner_validation_patients": int(metadata.iloc[inner_val].patient_id.nunique()),
            "inner_patient_overlap": 0,
            "inner_validation_ecg_sha256": hashlib.sha256(np.sort(metadata.ecg_id.to_numpy()[inner_val]).tobytes()).hexdigest(),
            "selected_epoch": selected_epoch,
            "best_inner_auprc": max(x["inner_auprc"] for x in inner_history if x["epoch"] >= 6),
            "normalizer_checksum": normalizer_audit.get("training_patient_checksum"),
            "model_checksum": model_hash,
            "representation_dim": 128,
            "encoder_supervision": "MI-label supervised; quantum-head ablation input",
            "architecture": "ECGPatchTransformer patch10 width96 heads4 layers3 ff192",
            "inner_history": inner_history,
            "retrain_history": retrain_history,
        }
        audit_path.write_text(json.dumps(audit, indent=2))
        audits.append(audit)
        print(f"outer fold {held_out}: selected_epoch={selected_epoch}, checksum={model_hash[:12]}", flush=True)
    if not np.isfinite(oof_embeddings).all() or not np.isfinite(oof_logits).all():
        raise RuntimeError("Incomplete Transformer OOF export")
    raw_probability = 1.0 / (1.0 + np.exp(-np.clip(oof_logits, -30, 30)))
    summary = {
        "records": int(len(metadata)),
        "patients": int(metadata.patient_id.nunique()),
        "folds": sorted(int(x) for x in np.unique(folds)),
        "representation_dim": 128,
        "oof_auprc_raw": float(average_precision_score(labels, raw_probability)),
        "oof_auroc_raw": float(roc_auc_score(labels, raw_probability)),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
        "encoder_supervision": "MI-label supervised",
        "coordinate_contract": "Each fold's training and validation embeddings share one retrained encoder; do not pool training coordinates across folds.",
        "outer_folds": audits,
    }
    _atomic_npz(
        output_dir / "waveform_oof_validation_only.npz",
        record_ids=metadata.ecg_id.to_numpy(),
        patient_ids=metadata.patient_id.to_numpy(),
        folds=folds,
        labels=labels,
        hard_negative=metadata.hard_negative.to_numpy(dtype=bool),
        embeddings=oof_embeddings,
        logits=oof_logits,
        raw_probability=raw_probability.astype(np.float32),
    )
    (output_dir / "representation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "outer_folds"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--normalizers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    export_representations(
        args.hdf5, args.metadata, args.normalizers, args.output,
        max_epochs=args.max_epochs, batch_size=args.batch_size,
        learning_rate=args.lr, seed=args.seed,
    )


if __name__ == "__main__":
    main()
