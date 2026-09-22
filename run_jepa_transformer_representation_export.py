"""Export patient-safe, label-free ECG-JEPA-style Transformer representations.

For every outer development fold, masked latent prediction is trained on the
other seven folds. Diagnostic labels never enter encoder training. One frozen
EMA target encoder then exports both training and held-out h128 vectors.
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
from aquire_preprocessing.models_jepa import ECGMaskedLatentPredictor
from run_waveform_representation_export import (
    _atomic_npz,
    _load_normalizer,
    _model_checksum,
    _normalise,
    _seed_everything,
)


def random_token_mask(
    batch_size: int, tokens: int, mask_ratio: float, *, device=None
):
    """Mask an exact random fraction of tokens independently for every ECG."""
    import torch

    if batch_size < 1 or tokens < 2 or not 0.0 < mask_ratio < 1.0:
        raise ValueError("Invalid masking configuration")
    count = max(1, min(tokens - 1, int(round(tokens * mask_ratio))))
    order = torch.rand(batch_size, tokens, device=device).argsort(dim=1)
    mask = torch.zeros(batch_size, tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, order[:, :count], True)
    return mask


def _augment_context(batch, gain_range: float = 0.05, noise_std: float = 0.01):
    import torch

    gain = torch.empty((len(batch), batch.shape[1], 1), device=batch.device).uniform_(
        1.0 - gain_range, 1.0 + gain_range
    )
    return batch * gain + noise_std * torch.randn_like(batch)


def _train_label_free(
    signals: np.ndarray,
    train_idx: np.ndarray,
    *,
    device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_ratio: float,
    ema_momentum: float,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _seed_everything(seed)
    dataset = TensorDataset(torch.from_numpy(signals[train_idx]))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    model = ECGMaskedLatentPredictor().to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        model.target_encoder.eval()
        totals = {
            "loss": 0.0, "prediction": 0.0, "global": 0.0,
            "variance": 0.0, "std": 0.0,
        }
        seen, last_grad = 0, np.nan
        for (target_signal,) in loader:
            target_signal = target_signal.to(device, non_blocking=True)
            context_signal = _augment_context(target_signal)
            mask = random_token_mask(
                len(target_signal), model.online_encoder.tokens, mask_ratio, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss, components = model(context_signal, target_signal, mask)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite JEPA loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            last_grad = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            if not np.isfinite(last_grad):
                raise RuntimeError("Non-finite JEPA gradient")
            scaler.step(optimizer)
            scaler.update()
            model.update_target(ema_momentum)
            size = len(target_signal)
            totals["loss"] += float(loss.detach()) * size
            totals["prediction"] += float(components["prediction_loss"]) * size
            totals["global"] += float(components["global_loss"]) * size
            totals["variance"] += float(components["variance_loss"]) * size
            totals["std"] += float(components["mean_feature_std"]) * size
            seen += size
        scheduler.step()
        record = {
            "epoch": epoch,
            "training_loss": totals["loss"] / seen,
            "prediction_loss": totals["prediction"] / seen,
            "global_loss": totals["global"] / seen,
            "variance_loss": totals["variance"] / seen,
            "mean_feature_std": totals["std"] / seen,
            "gradient_norm_last_batch": last_grad,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"    epoch {epoch:02d}/{epochs}: loss={record['training_loss']:.5f}, "
            f"std={record['mean_feature_std']:.4f}",
            flush=True,
        )
    return model, history


def _encode(model, signals: np.ndarray, device, batch_size: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    output = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(signals)), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for (batch,) in loader:
            output.append(model.encode(batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def export_representations(
    hdf5_path: Path,
    metadata_path: Path,
    normalizer_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 12,
    batch_size: int = 96,
    learning_rate: float = 3e-4,
    mask_ratio: float = 0.60,
    ema_momentum: float = 0.996,
    seed: int = 20260922,
) -> None:
    import h5py
    import torch

    if epochs < 1:
        raise ValueError("epochs must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[
        metadata.strat_fold.isin(DEV_FOLDS) & metadata.eligibility.eq("PRIMARY")
    ].copy().set_index("ecg_id", drop=False)
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
    print(f"JEPA representation device: {device}; records={len(signals)}", flush=True)
    labels = metadata.mi_label.to_numpy(dtype=int)  # exported only; never passed to training
    oof_embeddings = np.full((len(metadata), 128), np.nan, dtype=np.float32)
    audits = []
    for held_out in sorted(np.unique(folds)):
        artifact = output_dir / f"outer_fold_{held_out}_representations.npz"
        audit_path = output_dir / f"outer_fold_{held_out}_audit.json"
        checkpoint = output_dir / f"outer_fold_{held_out}_target_encoder.pt"
        if artifact.exists() and audit_path.exists() and checkpoint.exists():
            with np.load(artifact, allow_pickle=False) as saved:
                val_idx = saved["val_indices"].astype(int)
                oof_embeddings[val_idx] = saved["val_embeddings"]
            audits.append(json.loads(audit_path.read_text()))
            print(f"outer fold {held_out}: resumed", flush=True)
            continue
        train_idx = np.flatnonzero(folds != held_out)
        val_idx = np.flatnonzero(folds == held_out)
        normalizer_path = normalizer_dir / f"normalizer_holdout_fold_{held_out}.json"
        medians, iqrs, normalizer_audit = _load_normalizer(normalizer_path)
        expected_folds = sorted(int(x) for x in np.unique(folds[train_idx]))
        if normalizer_audit.get("folds_used") != expected_folds:
            raise ValueError(f"Fold-local normalizer mismatch in outer fold {held_out}")
        fold_signals = _normalise(signals, medians, iqrs)
        print(
            f"outer fold {held_out}: train={len(train_idx)}, val={len(val_idx)}, labels_hidden=True",
            flush=True,
        )
        model, history = _train_label_free(
            fold_signals, train_idx, device=device, epochs=epochs,
            batch_size=batch_size, learning_rate=learning_rate,
            mask_ratio=mask_ratio, ema_momentum=ema_momentum,
            seed=seed + int(held_out),
        )
        train_embeddings = _encode(model, fold_signals[train_idx], device, batch_size)
        val_embeddings = _encode(model, fold_signals[val_idx], device, batch_size)
        if not np.isfinite(train_embeddings).all() or not np.isfinite(val_embeddings).all():
            raise RuntimeError("Non-finite JEPA representation")
        oof_embeddings[val_idx] = val_embeddings
        state = {
            key: value.detach().cpu()
            for key, value in model.target_encoder.state_dict().items()
        }
        checksum = _model_checksum(state)
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(
            {"state_dict": state, "held_out_fold": int(held_out),
             "model_checksum": checksum, "encoder_supervision": "label-free"},
            temporary,
        )
        temporary.replace(checkpoint)
        _atomic_npz(
            artifact,
            train_indices=train_idx,
            val_indices=val_idx,
            train_record_ids=metadata.ecg_id.to_numpy()[train_idx],
            val_record_ids=metadata.ecg_id.to_numpy()[val_idx],
            train_patient_ids=metadata.patient_id.to_numpy()[train_idx],
            val_patient_ids=metadata.patient_id.to_numpy()[val_idx],
            train_labels=labels[train_idx],
            val_labels=labels[val_idx],
            train_embeddings=train_embeddings,
            val_embeddings=val_embeddings,
        )
        audit = {
            "held_out_fold": int(held_out),
            "training_folds": expected_folds,
            "training_records": int(len(train_idx)),
            "validation_records": int(len(val_idx)),
            "training_patients": int(metadata.iloc[train_idx].patient_id.nunique()),
            "validation_patients": int(metadata.iloc[val_idx].patient_id.nunique()),
            "patient_overlap": int(len(set(metadata.iloc[train_idx].patient_id) & set(metadata.iloc[val_idx].patient_id))),
            "normalizer_checksum": normalizer_audit.get("training_patient_checksum"),
            "model_checksum": checksum,
            "representation_dim": 128,
            "encoder_supervision": "label-free masked latent prediction; mi_label not passed to training",
            "architecture": "ECGPatchTransformer online encoder + EMA target + token predictor",
            "mask_ratio": mask_ratio,
            "ema_momentum": ema_momentum,
            "epochs": epochs,
            "history": history,
        }
        audit_path.write_text(json.dumps(audit, indent=2))
        audits.append(audit)
        print(f"outer fold {held_out}: checksum={checksum[:12]}", flush=True)
        del model, fold_signals, train_embeddings, val_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(oof_embeddings).all():
        raise RuntimeError("Incomplete JEPA OOF export")
    summary = {
        "records": int(len(metadata)),
        "patients": int(metadata.patient_id.nunique()),
        "folds": sorted(int(x) for x in np.unique(folds)),
        "representation_dim": 128,
        "encoder_supervision": "label-free masked latent prediction",
        "fold_9_accessed": False,
        "fold_10_accessed": False,
        "coordinate_contract": "Each fold's training and validation embeddings share one EMA target encoder.",
        "outer_folds": audits,
    }
    _atomic_npz(
        output_dir / "jepa_oof_validation_only.npz",
        record_ids=metadata.ecg_id.to_numpy(),
        patient_ids=metadata.patient_id.to_numpy(),
        folds=folds,
        labels=labels,
        hard_negative=metadata.hard_negative.to_numpy(dtype=bool),
        embeddings=oof_embeddings,
    )
    (output_dir / "representation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "outer_folds"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--normalizers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask-ratio", type=float, default=0.60)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    export_representations(
        args.hdf5, args.metadata, args.normalizers, args.output,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.lr, mask_ratio=args.mask_ratio,
        ema_momentum=args.ema_momentum, seed=args.seed,
    )


if __name__ == "__main__":
    main()
