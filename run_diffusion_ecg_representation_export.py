"""Export outer-fold, label-free diffusion-pretrained ECG representations."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_diffusion import (
    ECGDiffusionEncoder,
    cosine_alpha_bar,
    min_snr_noise_loss,
    q_sample,
)
from run_waveform_representation_export import (
    _atomic_npz,
    _load_normalizer,
    _model_checksum,
    _normalise,
    _seed_everything,
)


def _update_ema(ema_model, model, decay: float) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    with __import__("torch").no_grad():
        for ema, current in zip(ema_model.parameters(), model.parameters(), strict=True):
            ema.lerp_(current.detach(), 1.0 - decay)
        for ema, current in zip(ema_model.buffers(), model.buffers(), strict=True):
            ema.copy_(current)


def _train_diffusion(
    signals: np.ndarray,
    train_idx: np.ndarray,
    *,
    device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    diffusion_steps: int,
    ema_decay: float,
    min_snr_gamma: float,
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
    model = ECGDiffusionEncoder().to(device)
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.1
    )
    alpha_bar = cosine_alpha_bar(diffusion_steps).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, seen, last_gradient = 0.0, 0, np.nan
        for (clean,) in loader:
            clean = clean.to(device, non_blocking=True)
            timestep = torch.randint(
                0, diffusion_steps, (len(clean),), device=device, dtype=torch.long
            )
            noise = torch.randn_like(clean)
            noisy = q_sample(clean, timestep, alpha_bar, noise)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                prediction = model(noisy, timestep)
                loss = min_snr_noise_loss(
                    prediction, noise, timestep, alpha_bar, gamma=min_snr_gamma
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite diffusion loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            last_gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not np.isfinite(last_gradient):
                raise RuntimeError("Non-finite diffusion gradient")
            scaler.step(optimizer)
            scaler.update()
            _update_ema(ema_model, model, ema_decay)
            loss_sum += float(loss.detach()) * len(clean)
            seen += len(clean)
        scheduler.step()
        record = {
            "epoch": int(epoch),
            "training_min_snr_noise_loss": loss_sum / max(seen, 1),
            "gradient_norm_last_batch": last_gradient,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"    epoch {epoch:02d}/{epochs}: diffusion_loss="
            f"{record['training_min_snr_noise_loss']:.6f}",
            flush=True,
        )
    return ema_model, history


def _encode(model, signals: np.ndarray, device, batch_size: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    outputs = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(signals)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    with torch.inference_mode():
        for (batch,) in loader:
            outputs.append(model.encode(batch.to(device, non_blocking=True)).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def _representation_audit(values: np.ndarray) -> dict:
    centered = values - values.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = np.square(singular)
    probability = energy / max(float(energy.sum()), 1e-12)
    positive = probability[probability > 0]
    return {
        "dimension": int(values.shape[1]),
        "finite": bool(np.isfinite(values).all()),
        "feature_std_mean": float(values.std(axis=0).mean()),
        "feature_std_min": float(values.std(axis=0).min()),
        "effective_rank": float(np.exp(-(positive * np.log(positive)).sum())),
    }


def export_representations(
    hdf5_path: Path,
    metadata_path: Path,
    normalizer_dir: Path,
    output_dir: Path,
    *,
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 2e-4,
    diffusion_steps: int = 1000,
    ema_decay: float = 0.999,
    min_snr_gamma: float = 5.0,
    seed: int = 20260927,
) -> None:
    import h5py
    import torch

    if epochs < 1 or batch_size < 2:
        raise ValueError("invalid training budget")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[
        metadata.strat_fold.isin(DEV_FOLDS) & metadata.eligibility.eq("PRIMARY")
    ].copy().set_index("ecg_id", drop=False)
    folds = metadata.strat_fold.to_numpy(int)
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
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Diffusion representation device={device}; records={len(signals)}", flush=True)
    labels = metadata.mi_label.to_numpy(int)  # exported only, never used in pretraining
    oof_embeddings = np.full((len(metadata), 128), np.nan, dtype=np.float32)
    audits = []
    for held_out in sorted(np.unique(folds)):
        artifact = output_dir / f"outer_fold_{held_out}_representations.npz"
        audit_path = output_dir / f"outer_fold_{held_out}_audit.json"
        checkpoint = output_dir / f"outer_fold_{held_out}_ema_encoder.pt"
        if artifact.exists() and audit_path.exists() and checkpoint.exists():
            with np.load(artifact, allow_pickle=False) as saved:
                oof_embeddings[saved["val_indices"].astype(int)] = saved["val_embeddings"]
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
        model, history = _train_diffusion(
            fold_signals,
            train_idx,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            diffusion_steps=diffusion_steps,
            ema_decay=ema_decay,
            min_snr_gamma=min_snr_gamma,
            seed=seed + int(held_out),
        )
        train_embeddings = _encode(model, fold_signals[train_idx], device, batch_size)
        val_embeddings = _encode(model, fold_signals[val_idx], device, batch_size)
        if not np.isfinite(train_embeddings).all() or not np.isfinite(val_embeddings).all():
            raise RuntimeError("Non-finite diffusion representation")
        oof_embeddings[val_idx] = val_embeddings
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        checksum = _model_checksum(state)
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(
            {
                "state_dict": state,
                "held_out_fold": int(held_out),
                "model_checksum": checksum,
                "encoder_supervision": "label-free DDPM noise prediction",
            },
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
            "patient_overlap": int(
                len(
                    set(metadata.iloc[train_idx].patient_id)
                    & set(metadata.iloc[val_idx].patient_id)
                )
            ),
            "normalizer_checksum": normalizer_audit.get("training_patient_checksum"),
            "model_checksum": checksum,
            "representation_dim": 128,
            "encoder_supervision": "label-free DDPM epsilon prediction; labels not passed",
            "architecture": "forced-bottleneck temporal diffusion encoder-decoder",
            "diffusion_steps": int(diffusion_steps),
            "min_snr_gamma": float(min_snr_gamma),
            "ema_decay": float(ema_decay),
            "epochs": int(epochs),
            "train_representation": _representation_audit(train_embeddings),
            "validation_representation": _representation_audit(val_embeddings),
            "history": history,
        }
        audit_path.write_text(json.dumps(audit, indent=2))
        audits.append(audit)
        print(f"outer fold {held_out}: checksum={checksum[:12]}", flush=True)
        del model, fold_signals, train_embeddings, val_embeddings
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not np.isfinite(oof_embeddings).all():
        raise RuntimeError("Incomplete diffusion OOF export")
    summary = {
        "records": int(len(metadata)),
        "patients": int(metadata.patient_id.nunique()),
        "folds": sorted(int(x) for x in np.unique(folds)),
        "representation_dim": 128,
        "encoder_supervision": "label-free DDPM epsilon prediction",
        "representation_audit": _representation_audit(oof_embeddings),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
        "coordinate_contract": (
            "Every fold uses one EMA encoder fit only on its outer-training ECGs; "
            "training and validation coordinates share that encoder."
        ),
        "outer_folds": audits,
    }
    _atomic_npz(
        output_dir / "diffusion_oof_validation_only.npz",
        record_ids=metadata.ecg_id.to_numpy(),
        patient_ids=metadata.patient_id.to_numpy(),
        folds=folds,
        labels=labels,
        hard_negative=metadata.hard_negative.to_numpy(bool),
        embeddings=oof_embeddings,
    )
    (output_dir / "representation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "outer_folds"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--normalizers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--min-snr-gamma", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    export_representations(
        args.hdf5,
        args.metadata,
        args.normalizers,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        diffusion_steps=args.diffusion_steps,
        ema_decay=args.ema_decay,
        min_snr_gamma=args.min_snr_gamma,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
