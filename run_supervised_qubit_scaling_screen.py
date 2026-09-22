"""Fold-safe 8/12/16-qubit screen on supervised Transformer ECG vectors.

Each job evaluates one width so an expensive larger circuit cannot erase the
completed output of a smaller circuit.  The only changing factors are the
fold-local PLS width and the number of directly encoded qubits.  The patient
sample, Transformer h128 source, outer folds and training budget remain fixed.
"""

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
from run_quantum_core_screen import (
    _atomic_npz,
    _json_default,
    _patient_unique_sample,
    _seed_torch,
)
from run_waveform_quantum_core_screen import _embedding_to_angles


ALLOWED_QUBITS = (8, 12, 16)


def _matched_mlp(input_dim: int, target_parameters: int):
    import torch

    # Linear(d,h) + Linear(h,1) has h*(d+2)+1 parameters.
    hidden = max(1, round((target_parameters - 1) / (input_dim + 2)))
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
        torch.nn.Flatten(0),
    )
    return model, hidden


def _train_torch_head_device(
    model,
    q_train: np.ndarray,
    y_train: np.ndarray,
    q_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> tuple[np.ndarray, dict]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _seed_torch(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    model = model.to(device)
    dataset = TensorDataset(
        torch.from_numpy(q_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    losses, gradient_norms = [], []
    for epoch in range(epochs):
        model.train()
        total, count, epoch_gradients = 0.0, 0, []
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device, non_blocking=True))
            loss = criterion(logits, batch_y.to(device, non_blocking=True))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite quantum-head loss")
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            optimizer.step()
            epoch_gradients.append(gradient)
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        losses.append(total / max(count, 1))
        gradient_norms.append(float(np.median(epoch_gradients)))
        print(
            f"epoch {epoch + 1:02d}/{epochs}: loss={losses[-1]:.6f}, "
            f"gradient={gradient_norms[-1]:.6f}",
            flush=True,
        )
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(q_val), batch_size):
            batch = torch.from_numpy(q_val[start : start + batch_size]).to(
                device, non_blocking=True
            )
            outputs.append(model(batch).detach().cpu().numpy())
    logits = np.concatenate(outputs)
    if not np.isfinite(logits).all():
        raise RuntimeError("Non-finite validation output")
    audit = {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "gradient_norm_min": float(np.min(gradient_norms)),
        "gradient_norm_max": float(np.max(gradient_norms)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "device": str(device),
        "simulator": "exact PyTorch complex statevector",
    }
    return logits, audit


def _sensitivity_at_specificity(labels: np.ndarray, scores: np.ndarray, specificity: float = 0.9) -> float:
    negative = scores[labels == 0]
    positive = scores[labels == 1]
    threshold = float(np.quantile(negative, specificity, method="higher"))
    return float(np.mean(positive >= threshold))


def _metric_row(
    name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    hard_negative: np.ndarray,
    n_qubits: int,
) -> dict:
    hard_mask = (labels == 1) | hard_negative
    return {
        "model": name,
        "input_dim": int(n_qubits if not name.startswith("h128") else 128),
        "qubits": int(n_qubits if name.startswith(f"q{n_qubits}_vqc") else 0),
        "auprc": float(average_precision_score(labels, scores)),
        "auroc": float(roc_auc_score(labels, scores)),
        "sensitivity_at_90_specificity": _sensitivity_at_specificity(labels, scores),
        "hard_negative_auprc": float(average_precision_score(labels[hard_mask], scores[hard_mask])),
        "hard_negative_auroc": float(roc_auc_score(labels[hard_mask], scores[hard_mask])),
        "scores_calibrated": False,
    }


def run_screen(
    representation_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    *,
    n_qubits: int,
    per_class: int = 500,
    epochs: int = 20,
    batch_size: int = 32,
    seed: int = 20260922,
    folds: tuple[int, ...] = tuple(range(1, 9)),
    device: str = "cuda",
) -> None:
    if n_qubits not in ALLOWED_QUBITS:
        raise ValueError(f"n_qubits must be one of {ALLOWED_QUBITS}")
    if not folds or not set(folds) <= set(range(1, 9)):
        raise ValueError("Only development folds 1-8 may be used")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path)
    if not metadata.ecg_id.is_unique:
        raise ValueError("Metadata ecg_id must be unique")
    metadata = metadata.set_index("ecg_id")
    prefix = f"q{n_qubits}"
    model_names = (
        f"{prefix}_vqc",
        f"{prefix}_logreg",
        f"{prefix}_matched_mlp",
        f"{prefix}_rbf_svc",
        "h128_logreg",
    )
    keys = ("ecg_id", "patient_id", "fold", "label", "hard_negative", *model_names)
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
                raise ValueError("Non-finite supervised Transformer embedding")

            fit = _patient_unique_sample(
                train_y,
                train_meta.hard_negative.to_numpy(dtype=bool),
                train_patients,
                per_class,
                seed + held_out,
            )
            q_train, q_val, representation_audit = _embedding_to_angles(
                train_h, train_y, fit, val_h, n_qubits, seed + held_out
            )
            y_fit = train_y[fit]
            _seed_torch(seed + held_out + n_qubits * 100)
            vqc = TorchStatevectorQuantumClassifier(
                n_qubits=n_qubits, n_layers=2, topology="ring"
            )
            vqc_parameters = sum(parameter.numel() for parameter in vqc.parameters())
            mlp, hidden = _matched_mlp(n_qubits, vqc_parameters)
            print(
                f"fold {held_out}: q{n_qubits}, train={len(fit)}, val={len(val_y)}, "
                f"VQC params={vqc_parameters}, matched MLP hidden={hidden}",
                flush=True,
            )
            vqc_logits, vqc_audit = _train_torch_head_device(
                vqc, q_train, y_fit, q_val,
                epochs=epochs, batch_size=batch_size, learning_rate=0.01,
                seed=seed + held_out + n_qubits * 100, device_name=device,
            )
            mlp_logits, mlp_audit = _train_torch_head_device(
                mlp, q_train, y_fit, q_val,
                epochs=epochs, batch_size=min(128, max(32, batch_size)), learning_rate=0.01,
                seed=seed + held_out + n_qubits * 100 + 1000, device_name=device,
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
                f"{prefix}_vqc": expit(vqc_logits),
                f"{prefix}_logreg": logistic.predict_proba(q_val)[:, 1],
                f"{prefix}_matched_mlp": expit(mlp_logits),
                f"{prefix}_rbf_svc": expit(rbf.decision_function(q_val)),
                "h128_logreg": h128_logistic.predict_proba(val_h)[:, 1],
            }
            if any(len(fold_out[name]) != len(val_y) or not np.isfinite(fold_out[name]).all() for name in model_names):
                raise ValueError("Incomplete or non-finite prediction")
            audit = {
                "held_out_fold": held_out,
                "source": source.name,
                "source_encoder_supervised": True,
                "training_records": int(len(fit)),
                "training_patients": int(len(np.unique(train_patients[fit]))),
                "training_ecg_sha256": hashlib.sha256(np.sort(train_ids[fit]).tobytes()).hexdigest(),
                "validation_ecg_sha256": hashlib.sha256(np.sort(val_ids).tobytes()).hexdigest(),
                "representation": representation_audit,
                "vqc": vqc_audit,
                "matched_mlp": {**mlp_audit, "hidden_units": hidden},
                "statevector_amplitudes": int(1 << n_qubits),
                "scores_calibrated": False,
                "folds_9_and_10_accessed": False,
            }
            _atomic_npz(
                checkpoint,
                **fold_out,
                audit=json.dumps(audit, default=_json_default, sort_keys=True),
            )
        for key in keys:
            collected[key].append(fold_out[key])
        audits.append(audit)

    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    if len(np.unique(arrays["ecg_id"])) != len(arrays["ecg_id"]):
        raise ValueError("Duplicate OOF ECG")
    metrics = pd.DataFrame([
        _metric_row(name, arrays["label"], arrays[name], arrays["hard_negative"], n_qubits)
        for name in model_names
    ]).sort_values("auprc", ascending=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(arrays).to_csv(output_dir / "predictions_wide.csv", index=False)
    (output_dir / "fold_audits.json").write_text(
        json.dumps(audits, indent=2, default=_json_default)
    )
    for control in (f"{prefix}_logreg", f"{prefix}_matched_mlp", f"{prefix}_rbf_svc"):
        report = _paired_patient_bootstrap(
            arrays["label"], arrays["patient_id"], arrays[f"{prefix}_vqc"], arrays[control],
            iterations=2000, seed=seed,
            comparison=f"{prefix}_vqc_minus_{control}",
        )
        (output_dir / f"paired_{prefix}_vqc_vs_{control}.json").write_text(
            json.dumps(report, indent=2, default=_json_default)
        )
    manifest = {
        "experiment": "supervised_transformer_direct_qubit_scaling",
        "qubits": n_qubits,
        "folds": list(folds),
        "records": int(len(arrays["label"])),
        "patients": int(len(np.unique(arrays["patient_id"]))),
        "source_dim": 128,
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--qubits", required=True, type=int, choices=ALLOWED_QUBITS)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run_screen(
        args.representations, args.metadata, args.output,
        n_qubits=args.qubits, per_class=args.per_class, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed, folds=tuple(args.folds),
        device=args.device,
    )


if __name__ == "__main__":
    main()
