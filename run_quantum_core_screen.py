"""Patient-safe direct quantum-core screen on deployable ECG features.

The classical front-end is restricted to fold-local imputation, robust scaling,
supervised PLS compression and a quantile-to-angle map.  The VQC consumes that
``q_d`` vector directly; it has no high-capacity classical encoder and only a
linear readout.  Matched RBF and small-MLP heads receive the identical vectors.

This is a development-fold screening experiment.  Folds 9 and 10 are rejected.
Probabilities are deliberately marked uncalibrated; calibration belongs after
one quantum champion is frozen.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.svm import SVC

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_quantum import DirectQuantumClassifier
from run_quantum_baselines import _paired_patient_bootstrap


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _patient_unique_sample(
    labels: np.ndarray,
    hard_negative: np.ndarray,
    patient_ids: np.ndarray,
    per_class: int,
    seed: int,
) -> np.ndarray:
    """Balanced MI/hard-negative/other sample with at most one record per patient."""
    frame = pd.DataFrame(
        {
            "row": np.arange(len(labels)),
            "label": np.asarray(labels, dtype=int),
            "hard": np.asarray(hard_negative, dtype=bool),
            "patient": np.asarray(patient_ids),
        }
    ).sample(frac=1.0, random_state=seed)
    positive = frame[frame.label.eq(1)].drop_duplicates("patient")
    n_positive = min(per_class, len(positive))
    positive = positive.iloc[:n_positive]
    used = set(positive.patient)
    negative = frame[frame.label.eq(0) & ~frame.patient.isin(used)]
    hard = negative[negative.hard].drop_duplicates("patient")
    n_hard = min(n_positive // 2, len(hard))
    hard = hard.iloc[:n_hard]
    used.update(hard.patient)
    other = negative[~negative.hard & ~negative.patient.isin(used)].drop_duplicates("patient")
    other = other.iloc[: max(n_positive - n_hard, 0)]
    negative_rows = pd.concat([hard, other], ignore_index=True)
    n = min(len(positive), len(negative_rows))
    selected = np.r_[positive.row.to_numpy()[:n], negative_rows.row.to_numpy()[:n]]
    rng = np.random.default_rng(seed)
    selected = rng.permutation(selected.astype(int))
    selected_patients = patient_ids[selected]
    if len(np.unique(selected_patients)) != len(selected_patients):
        raise RuntimeError("Patient-unique sample contains duplicate patients")
    return selected


def _fit_representation(
    features: np.ndarray,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    transform_idx: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    x_fit = scaler.fit_transform(imputer.fit_transform(features[fit_idx]))
    x_transform = scaler.transform(imputer.transform(features[transform_idx]))
    pls = PLSRegression(n_components=n_components, scale=False, max_iter=1000)
    q_fit = pls.fit_transform(x_fit, labels[fit_idx])[0]
    q_transform = pls.transform(x_transform)
    quantile = QuantileTransformer(
        n_quantiles=min(256, len(q_fit)),
        output_distribution="uniform",
        random_state=seed,
    )
    q_fit = (2.0 * quantile.fit_transform(q_fit) - 1.0) * np.pi
    q_transform = (2.0 * quantile.transform(q_transform) - 1.0) * np.pi
    audit = {
        "method": "patient-unique train-only median + robust scale + supervised PLS + quantile angles",
        "fit_records": int(len(fit_idx)),
        "n_components": int(n_components),
        "minimum": float(q_fit.min()),
        "maximum": float(q_fit.max()),
    }
    return q_fit.astype(np.float32), q_transform.astype(np.float32), audit


def _seed_torch(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _train_torch_head(
    model,
    q_train: np.ndarray,
    y_train: np.ndarray,
    q_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _seed_torch(seed)
    device = torch.device("cpu")
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
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss()
    losses = []
    gradient_norms = []
    for _ in range(epochs):
        model.train()
        total, count, last_gradient = 0.0, 0, np.nan
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device))
            loss = criterion(logits, batch_y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite quantum-head loss")
            loss.backward()
            last_gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
        losses.append(total / max(count, 1))
        gradient_norms.append(last_gradient)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(q_val), batch_size):
            outputs.append(model(torch.from_numpy(q_val[start:start + batch_size])).numpy())
    logits = np.concatenate(outputs)
    if not np.isfinite(logits).all():
        raise RuntimeError("Non-finite quantum-head validation output")
    audit = {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "gradient_norm_min": float(np.nanmin(gradient_norms)),
        "gradient_norm_max": float(np.nanmax(gradient_norms)),
        "epochs": int(epochs),
    }
    return logits, audit


def _matched_mlp(n_features: int):
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(n_features, n_features),
        torch.nn.Tanh(),
        torch.nn.Linear(n_features, 1),
        torch.nn.Flatten(0),
    )


def run_screen(
    metadata_path: Path,
    features_path: Path,
    manifest_path: Path,
    output_dir: Path,
    n_qubits: int = 4,
    per_class: int = 500,
    epochs: int = 20,
    batch_size: int = 64,
    seed: int = 20260922,
) -> None:
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")
    feature_frame = pd.read_csv(features_path).set_index("ecg_id")
    approved = load_feature_manifest(manifest_path, feature_frame.columns)["approved_features"]
    joined = metadata.join(feature_frame[approved], how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")]
    folds = joined.strat_fold.to_numpy(dtype=int)
    guard_fold_access(folds, purpose="tuning")
    if joined.groupby("patient_id").strat_fold.nunique().max() != 1:
        raise ValueError("Patient crosses development folds")
    labels = joined.mi_label.to_numpy(dtype=int)
    patient_ids = joined.patient_id.to_numpy()
    hard_negative = joined.hard_negative.to_numpy(dtype=bool)
    record_ids = joined.index.to_numpy(dtype=int)
    features = (
        joined[approved]
        .select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=float)
    )
    model_names = ["direct_vqc", "matched_mlp", "rbf_svc"]
    oof = {name: np.full(len(joined), np.nan, dtype=float) for name in model_names}
    audits = []

    for held_out in sorted(np.unique(folds)):
        checkpoint = output_dir / f"fold_{held_out}.npz"
        if checkpoint.exists():
            saved = np.load(checkpoint, allow_pickle=False)
            val_idx = saved["val_idx"].astype(int)
            for name in model_names:
                oof[name][val_idx] = saved[name]
            audits.append(json.loads(str(saved["audit"].item())))
            print(f"fold {held_out}: resumed", flush=True)
            continue
        outer_train = np.flatnonzero(folds != held_out)
        val_idx = np.flatnonzero(folds == held_out)
        relative = _patient_unique_sample(
            labels[outer_train],
            hard_negative[outer_train],
            patient_ids[outer_train],
            per_class,
            seed + int(held_out),
        )
        fit_idx = outer_train[relative]
        q_train, q_val, representation_audit = _fit_representation(
            features,
            labels,
            fit_idx,
            val_idx,
            n_qubits,
            seed + int(held_out),
        )
        y_train = labels[fit_idx]
        print(
            f"fold {held_out}: q{n_qubits}, train={len(fit_idx)}, val={len(val_idx)}",
            flush=True,
        )

        vqc = DirectQuantumClassifier(n_qubits=n_qubits, n_layers=2, topology="ring")
        vqc_logits, vqc_audit = _train_torch_head(
            vqc,
            q_train,
            y_train,
            q_val,
            epochs,
            batch_size,
            learning_rate=0.01,
            seed=seed + int(held_out),
        )
        mlp = _matched_mlp(n_qubits)
        mlp_logits, mlp_audit = _train_torch_head(
            mlp,
            q_train,
            y_train,
            q_val,
            epochs,
            batch_size,
            learning_rate=0.01,
            seed=seed + int(held_out),
        )
        rbf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
        rbf.fit(q_train, y_train)
        rbf_logits = rbf.decision_function(q_val)
        oof["direct_vqc"][val_idx] = expit(vqc_logits)
        oof["matched_mlp"][val_idx] = expit(mlp_logits)
        oof["rbf_svc"][val_idx] = expit(rbf_logits)
        audit = {
            "held_out_fold": int(held_out),
            "training_records": int(len(fit_idx)),
            "training_patients": int(len(np.unique(patient_ids[fit_idx]))),
            "representation": representation_audit,
            "vqc": vqc_audit,
            "matched_mlp": mlp_audit,
            "probability_status": "uncalibrated_monotonic_sigmoid_of_margin",
        }
        audits.append(audit)
        _atomic_npz(
            checkpoint,
            val_idx=val_idx,
            direct_vqc=oof["direct_vqc"][val_idx],
            matched_mlp=oof["matched_mlp"][val_idx],
            rbf_svc=oof["rbf_svc"][val_idx],
            audit=json.dumps(audit, default=_json_default, sort_keys=True),
        )
        print(
            f"fold {held_out}: VQC loss {vqc_audit['initial_loss']:.4f} -> "
            f"{vqc_audit['final_loss']:.4f}",
            flush=True,
        )

    metrics = []
    prediction_rows = []
    for name, probability in oof.items():
        if not np.isfinite(probability).all():
            raise RuntimeError(f"Incomplete OOF predictions: {name}")
        metrics.append(
            {
                "model": name,
                "representation": f"clinical_pls_q{n_qubits}",
                "auprc": float(average_precision_score(labels, probability)),
                "auroc": float(roc_auc_score(labels, probability)),
                "probability_status": "uncalibrated_monotonic_sigmoid_of_margin",
            }
        )
        for record, patient, fold, label, hard, value in zip(
            record_ids, patient_ids, folds, labels, hard_negative, probability
        ):
            prediction_rows.append(
                {
                    "ecg_id": int(record),
                    "patient_id": int(patient),
                    "fold": int(fold),
                    "label": int(label),
                    "hard_negative": bool(hard),
                    "model": name,
                    "score": float(value),
                }
            )
    metric_frame = pd.DataFrame(metrics).sort_values("auprc", ascending=False)
    metric_frame.to_csv(output_dir / "quantum_core_screen_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "quantum_core_screen_predictions.csv", index=False
    )
    (output_dir / "quantum_core_fold_audits.json").write_text(
        json.dumps(audits, indent=2, default=_json_default)
    )
    for control in ("matched_mlp", "rbf_svc"):
        report = _paired_patient_bootstrap(
            labels,
            patient_ids,
            oof["direct_vqc"],
            oof[control],
            iterations=2000,
            seed=seed,
        )
        report["comparison"] = f"direct_vqc_minus_{control}"
        (output_dir / f"paired_direct_vqc_vs_{control}.json").write_text(
            json.dumps(report, indent=2, default=_json_default)
        )
    print(metric_frame.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    run_screen(
        args.metadata,
        args.features,
        args.manifest,
        args.output,
        n_qubits=args.qubits,
        per_class=args.per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
