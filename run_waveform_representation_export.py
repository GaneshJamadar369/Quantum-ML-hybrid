"""Patient-safe outer-fold waveform representation export.

For every development fold, one ECG encoder is trained on the other seven
folds, frozen, and then used to embed *both* its training records and held-out
records.  Each fold artifact therefore has a single coherent 128-dimensional
coordinate system suitable for downstream classical/quantum head comparisons.

Folds 9 and 10 are rejected.  The script is intentionally limited to the
waveform encoder; it does not select a quantum model or inspect held-out labels
during training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import pandas as pd

from aquire_preprocessing.config import CANONICAL_LEAD_ORDER, DEV_FOLDS
from aquire_preprocessing.manifest import guard_fold_access


def patient_balanced_weights(
    patient_ids: np.ndarray,
    labels: np.ndarray,
    hard_negative: np.ndarray,
    hard_negative_multiplier: float = 1.5,
) -> np.ndarray:
    """Give every patient equal total base weight, then upweight hard negatives."""
    patient_ids = np.asarray(patient_ids)
    labels = np.asarray(labels)
    hard_negative = np.asarray(hard_negative, dtype=bool)
    _, inverse, counts = np.unique(patient_ids, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= weights.mean()
    weights[(labels == 0) & hard_negative] *= float(hard_negative_multiplier)
    # Keep the average loss scale comparable across folds.
    weights /= weights.mean()
    return weights.astype(np.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_normalizer(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    payload = json.loads(path.read_text())
    medians = np.asarray(
        [payload["lead_medians"][lead] for lead in CANONICAL_LEAD_ORDER], dtype=np.float32
    )
    iqrs = np.asarray(
        [payload["lead_iqrs"][lead] for lead in CANONICAL_LEAD_ORDER], dtype=np.float32
    )
    if medians.shape != (12,) or iqrs.shape != (12,) or not np.isfinite(medians).all():
        raise ValueError(f"Invalid normalizer: {path}")
    if not np.isfinite(iqrs).all() or np.any(iqrs <= 0):
        raise ValueError(f"Invalid normalizer IQR: {path}")
    return medians, iqrs, payload


def _normalise(signals: np.ndarray, medians: np.ndarray, iqrs: np.ndarray) -> np.ndarray:
    return ((signals - medians[None, :, None]) / iqrs[None, :, None]).astype(np.float32)


def _atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _model_checksum(state_dict: dict) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def _embed(model, signals: np.ndarray, device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model.eval()
    embeddings: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(torch.from_numpy(signals)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for (batch,) in loader:
            batch_logits, batch_embeddings = model(batch.to(device, non_blocking=True))
            logits.append(batch_logits.cpu().numpy())
            embeddings.append(batch_embeddings.cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(logits)


def _train_fold(
    signals: np.ndarray,
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from aquire_preprocessing.models_1d import ECGResNet1D
    from run_deep_learning_oof import FocalLoss

    _seed_everything(seed)
    labels = metadata.mi_label.to_numpy(dtype=np.float32)
    hard_negative = metadata.hard_negative.to_numpy(dtype=bool)
    patient_ids = metadata.patient_id.to_numpy()
    weights = patient_balanced_weights(
        patient_ids[train_idx], labels[train_idx], hard_negative[train_idx]
    )
    train_dataset = TensorDataset(
        torch.from_numpy(signals[train_idx]),
        torch.from_numpy(labels[train_idx]),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    model = ECGResNet1D(in_channels=12, base_filters=32, embedding_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = FocalLoss(alpha=0.65, gamma=2.0)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        count = 0
        for batch_signal, batch_label, batch_weight in loader:
            batch_signal = batch_signal.to(device, non_blocking=True)
            batch_label = batch_label.to(device, non_blocking=True)
            batch_weight = batch_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch_signal)
            loss = criterion(logits, batch_label, batch_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_signal)
            count += len(batch_signal)
        scheduler.step()
        mean_loss = total_loss / max(count, 1)
        history.append({"epoch": epoch + 1, "training_loss": mean_loss})
        print(f"    epoch {epoch + 1:02d}/{epochs}: loss={mean_loss:.6f}", flush=True)
    return model, history


def export_representations(
    hdf5_path: Path,
    metadata_path: Path,
    normalizer_dir: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 5e-4,
    seed: int = 42,
) -> None:
    import h5py
    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    metadata = metadata[
        metadata.strat_fold.isin(DEV_FOLDS) & metadata.eligibility.eq("PRIMARY")
    ].copy()
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
        raise ValueError(f"Unexpected waveform tensor {signals.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"representation device: {device}; records={len(signals)}", flush=True)
    labels = metadata.mi_label.to_numpy(dtype=int)
    oof_embeddings = np.full((len(metadata), 128), np.nan, dtype=np.float32)
    oof_logits = np.full(len(metadata), np.nan, dtype=np.float32)
    audits = []

    for held_out in sorted(np.unique(folds)):
        artifact = output_dir / f"outer_fold_{held_out}_representations.npz"
        audit_path = output_dir / f"outer_fold_{held_out}_audit.json"
        checkpoint_path = output_dir / f"outer_fold_{held_out}_encoder.pt"
        if artifact.exists() and audit_path.exists() and checkpoint_path.exists():
            saved = np.load(artifact, allow_pickle=False)
            val_idx = saved["val_indices"].astype(int)
            oof_embeddings[val_idx] = saved["val_embeddings"]
            oof_logits[val_idx] = saved["val_logits"]
            audits.append(json.loads(audit_path.read_text()))
            print(f"outer fold {held_out}: resumed", flush=True)
            continue

        train_idx = np.flatnonzero(folds != held_out)
        val_idx = np.flatnonzero(folds == held_out)
        normalizer_path = normalizer_dir / f"normalizer_holdout_fold_{held_out}.json"
        medians, iqrs, normalizer_audit = _load_normalizer(normalizer_path)
        expected_folds = sorted(int(value) for value in np.unique(folds[train_idx]))
        if normalizer_audit.get("folds_used") != expected_folds:
            raise ValueError(
                f"Normalizer {normalizer_path} used {normalizer_audit.get('folds_used')}, "
                f"expected {expected_folds}"
            )
        fold_signals = _normalise(signals, medians, iqrs)
        print(
            f"outer fold {held_out}: train={len(train_idx)}, val={len(val_idx)}, "
            f"seed={seed + held_out}",
            flush=True,
        )
        model, history = _train_fold(
            fold_signals,
            metadata,
            train_idx,
            device,
            epochs,
            batch_size,
            learning_rate,
            seed + int(held_out),
        )
        # One frozen encoder creates both sides of this outer-fold experiment.
        train_embeddings, train_logits = _embed(
            model, fold_signals[train_idx], device, batch_size
        )
        val_embeddings, val_logits = _embed(
            model, fold_signals[val_idx], device, batch_size
        )
        if not all(
            np.isfinite(value).all()
            for value in (train_embeddings, train_logits, val_embeddings, val_logits)
        ):
            raise RuntimeError(f"Non-finite representation in outer fold {held_out}")
        oof_embeddings[val_idx] = val_embeddings.astype(np.float32)
        oof_logits[val_idx] = val_logits.astype(np.float32)
        state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
        checksum = _model_checksum(state_dict)
        temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "state_dict": state_dict,
                "held_out_fold": int(held_out),
                "seed": int(seed + held_out),
                "model_checksum": checksum,
            },
            temporary_checkpoint,
        )
        temporary_checkpoint.replace(checkpoint_path)
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
            train_embeddings=train_embeddings.astype(np.float32),
            val_embeddings=val_embeddings.astype(np.float32),
            train_logits=train_logits.astype(np.float32),
            val_logits=val_logits.astype(np.float32),
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
            "model_checksum": checksum,
            "normalizer_checksum": normalizer_audit.get("training_patient_checksum"),
            "representation_dim": 128,
            "encoder_supervision": "MI-label supervised; quantum-core ablation input",
            "patient_balanced_loss": True,
            "history": history,
        }
        audit_path.write_text(json.dumps(audit, indent=2))
        audits.append(audit)
        print(f"outer fold {held_out}: exported checksum={checksum[:12]}", flush=True)

    if not np.isfinite(oof_embeddings).all() or not np.isfinite(oof_logits).all():
        raise RuntimeError("Incomplete OOF representation export")
    raw_probability = 1.0 / (1.0 + np.exp(-oof_logits))
    summary = {
        "records": int(len(metadata)),
        "patients": int(metadata.patient_id.nunique()),
        "folds": sorted(int(value) for value in np.unique(folds)),
        "representation_dim": 128,
        "oof_auprc_raw": float(average_precision_score(labels, raw_probability)),
        "oof_auroc_raw": float(roc_auc_score(labels, raw_probability)),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
        "coordinate_contract": (
            "Each outer-fold artifact uses one frozen encoder for its train and validation rows; "
            "coordinates must not be pooled across fold artifacts."
        ),
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    export_representations(
        args.hdf5,
        args.metadata,
        args.normalizers,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
