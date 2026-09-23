"""Nested outer-fold screen for evidence-routed classical/quantum fusion.

The classical expert combines a fold-coherent Transformer representation and
the frozen deployable clinical-feature manifest.  Inner official folds create
honest base-model OOF predictions inside each outer training set.  Their
stacked probability defines a residual target.  Four residual PLS coordinates
are then built from three candidate sources and passed to a q4 VQC and a
parameter-count-matched MLP.

Fusion weights are fixed before observing outer-fold results.  This is a
screening experiment; a learned fusion weight requires a further nested OOF
quantum loop if this fixed-weight gate is positive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _atomic_npz, _json_default, _patient_unique_sample, _seed_torch


REPRESENTATIONS = ("h128_residual", "clinical_residual", "combined_residual")
FUSION_WEIGHTS = (0.10, 0.25, 0.50)


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(probability, dtype=float), 1e-5, 1.0 - 1e-5))


def _clinical_model(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            random_state=seed,
        ),
    )


def _waveform_model():
    return make_pipeline(
        RobustScaler(quantile_range=(25.0, 75.0)),
        LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
    )


def _classical_expert(
    train_h: np.ndarray,
    val_h: np.ndarray,
    train_c: np.ndarray,
    val_c: np.ndarray,
    train_y: np.ndarray,
    train_folds: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return inner-OOF train and outer-validation classical probabilities."""

    wave_oof = np.full(len(train_y), np.nan)
    clinical_oof = np.full(len(train_y), np.nan)
    inner_folds = sorted(np.unique(train_folds).astype(int).tolist())
    if len(inner_folds) != 7 or not set(inner_folds) <= set(range(1, 9)):
        raise ValueError("Each outer training set must contain seven official development folds")
    for inner_fold in inner_folds:
        inner_train = train_folds != inner_fold
        inner_validation = train_folds == inner_fold
        wave = _waveform_model().fit(train_h[inner_train], train_y[inner_train])
        clinical = _clinical_model(seed + inner_fold).fit(
            train_c[inner_train], train_y[inner_train]
        )
        wave_oof[inner_validation] = wave.predict_proba(train_h[inner_validation])[:, 1]
        clinical_oof[inner_validation] = clinical.predict_proba(train_c[inner_validation])[:, 1]
    if not np.isfinite(wave_oof).all() or not np.isfinite(clinical_oof).all():
        raise RuntimeError("Incomplete inner OOF classical predictions")

    meta_x = np.column_stack([_safe_logit(wave_oof), _safe_logit(clinical_oof)])
    meta = LogisticRegression(C=1.0, max_iter=2000).fit(meta_x, train_y)
    train_probability = meta.predict_proba(meta_x)[:, 1]

    wave = _waveform_model().fit(train_h, train_y)
    clinical = _clinical_model(seed + 100).fit(train_c, train_y)
    val_wave = wave.predict_proba(val_h)[:, 1]
    val_clinical = clinical.predict_proba(val_c)[:, 1]
    val_probability = meta.predict_proba(
        np.column_stack([_safe_logit(val_wave), _safe_logit(val_clinical)])
    )[:, 1]
    audit = {
        "inner_folds": inner_folds,
        "meta_coefficients": meta.coef_[0].tolist(),
        "meta_intercept": float(meta.intercept_[0]),
        "train_oof_auprc": float(average_precision_score(train_y, train_probability)),
        "train_oof_base_wave_auprc": float(average_precision_score(train_y, wave_oof)),
        "train_oof_base_clinical_auprc": float(average_precision_score(train_y, clinical_oof)),
    }
    return train_probability, val_probability, audit


def _source_matrix(
    name: str,
    h: np.ndarray,
    clinical: np.ndarray,
) -> np.ndarray:
    if name == "h128_residual":
        return h
    if name == "clinical_residual":
        return clinical
    if name == "combined_residual":
        return np.concatenate([h, clinical], axis=1)
    raise ValueError(name)


def _residual_angles(
    train_source: np.ndarray,
    val_source: np.ndarray,
    residual: np.ndarray,
    fit: np.ndarray,
    seed: int,
    n_components: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    fit_scaled = scaler.fit_transform(imputer.fit_transform(train_source[fit]))
    val_scaled = scaler.transform(imputer.transform(val_source))
    pls = PLSRegression(n_components=n_components, scale=False, max_iter=1000)
    q_fit = pls.fit_transform(fit_scaled, residual[fit])[0]
    q_val = pls.transform(val_scaled)
    quantile = QuantileTransformer(
        n_quantiles=min(256, len(fit)), output_distribution="uniform", random_state=seed
    )
    q_fit = (2.0 * quantile.fit_transform(q_fit) - 1.0) * np.pi
    q_val = (2.0 * quantile.transform(q_val) - 1.0) * np.pi
    return q_fit.astype(np.float32), q_val.astype(np.float32), {
        "source_dim": int(train_source.shape[1]),
        "target_dim": n_components,
        "method": "fit-sample median + robust scale + PLS(y-p_classical_oof) + quantile angles",
        "pls_y_variance": float(np.var(residual[fit])),
        "angle_min": float(q_fit.min()),
        "angle_max": float(q_fit.max()),
    }


def _matched_mlp(input_dim: int, target_parameters: int):
    import torch

    hidden = max(1, round((target_parameters - 1) / (input_dim + 2)))
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 1),
        torch.nn.Flatten(0),
    ), hidden


def _train_weighted_head(
    model,
    q_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    q_val: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _seed_torch(seed)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    model = model.to(device)
    weights = np.asarray(sample_weight, dtype=np.float32)
    weights = weights / max(float(weights.mean()), 1e-6)
    dataset = TensorDataset(
        torch.from_numpy(q_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    losses, gradients = [], []
    for _ in range(epochs):
        model.train()
        total, count, epoch_gradient = 0.0, 0, []
        for batch_x, batch_y, batch_weight in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x.to(device, non_blocking=True))
            loss = (
                criterion(logits, batch_y.to(device, non_blocking=True))
                * batch_weight.to(device, non_blocking=True)
            ).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite residual-head loss")
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            optimizer.step()
            total += float(loss.detach()) * len(batch_x)
            count += len(batch_x)
            epoch_gradient.append(gradient)
        losses.append(total / max(count, 1))
        gradients.append(float(np.median(epoch_gradient)))

    def predict(array: np.ndarray) -> np.ndarray:
        output = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(array), batch_size):
                batch = torch.from_numpy(array[start : start + batch_size]).to(
                    device, non_blocking=True
                )
                output.append(model(batch).detach().cpu().numpy())
        return np.concatenate(output)

    train_logits, val_logits = predict(q_train), predict(q_val)
    center = float(np.median(train_logits))
    scale = float(np.subtract(*np.percentile(train_logits, [75, 25])))
    if scale < 1e-6:
        scale = float(np.std(train_logits))
    if scale < 1e-6:
        raise RuntimeError("Residual head produced a constant training score")
    val_standardized = np.clip((val_logits - center) / scale, -8.0, 8.0)
    audit = {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "gradient_norm_min": float(np.min(gradients)),
        "gradient_norm_max": float(np.max(gradients)),
        "train_logit_center": center,
        "train_logit_iqr": scale,
        "epochs": epochs,
        "device": str(device),
    }
    return train_logits, val_standardized, audit


def _metric(name: str, y: np.ndarray, score: np.ndarray, hard: np.ndarray) -> dict:
    probability = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    hard_mask = (y == 1) | hard
    return {
        "model": name,
        "auprc": float(average_precision_score(y, probability)),
        "auroc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "hard_negative_auprc": float(average_precision_score(y[hard_mask], probability[hard_mask])),
        "scores_calibrated": False,
    }


def run_screen(
    representation_dir: Path,
    metadata_path: Path,
    features_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    per_class: int = 500,
    epochs: int = 20,
    batch_size: int = 64,
    seed: int = 20260923,
    folds: tuple[int, ...] = tuple(range(1, 9)),
    device: str = "cuda",
) -> None:
    if not folds or not set(folds) <= set(range(1, 9)):
        raise ValueError("Only development folds 1-8 may be used")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")
    features = pd.read_csv(features_path).set_index("ecg_id")
    manifest = load_feature_manifest(manifest_path, features.columns)
    approved = list(manifest["approved_features"])
    missing = sorted(set(approved) - set(features.columns))
    if missing:
        raise ValueError(f"Approved feature columns missing: {missing[:10]}")

    model_names = ["classical_expert"]
    for representation in REPRESENTATIONS:
        model_names.extend([f"{representation}_vqc", f"{representation}_mlp"])
        for weight in FUSION_WEIGHTS:
            suffix = str(weight).replace(".", "p")
            model_names.extend([
                f"{representation}_vqc_fusion_{suffix}",
                f"{representation}_mlp_fusion_{suffix}",
            ])
    keys = ("ecg_id", "patient_id", "fold", "label", "hard_negative", *model_names)
    collected = {key: [] for key in keys}
    audits = []

    for held_out in folds:
        checkpoint = output_dir / f"fold_{held_out}.npz"
        if checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                fold_output = {key: saved[key] for key in keys}
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
            train_meta, val_meta = metadata.loc[train_ids], metadata.loc[val_ids]
            if len(set(train_patients) & set(val_patients)):
                raise ValueError(f"Patient leakage in fold {held_out}")
            if not train_meta.strat_fold.isin(range(1, 9)).all() or not val_meta.strat_fold.eq(held_out).all():
                raise ValueError("Fold boundary mismatch")
            if not np.array_equal(train_meta.mi_label.to_numpy(int), train_y) or not np.array_equal(val_meta.mi_label.to_numpy(int), val_y):
                raise ValueError("Immutable label mismatch")
            train_c = features.loc[train_ids, approved].to_numpy(dtype=np.float32)
            val_c = features.loc[val_ids, approved].to_numpy(dtype=np.float32)
            train_classical, val_classical, classical_audit = _classical_expert(
                train_h, val_h, train_c, val_c, train_y,
                train_meta.strat_fold.to_numpy(int), seed + held_out,
            )
            residual = train_y.astype(float) - train_classical
            fit = _patient_unique_sample(
                train_y,
                train_meta.hard_negative.to_numpy(bool),
                train_patients,
                per_class,
                seed + held_out,
            )
            y_fit = train_y[fit]
            residual_weight = 1.0 + 2.0 * np.abs(residual[fit])
            fold_output = {
                "ecg_id": val_ids,
                "patient_id": val_patients,
                "fold": np.full(len(val_y), held_out, dtype=int),
                "label": val_y,
                "hard_negative": val_meta.hard_negative.to_numpy(bool),
                "classical_expert": val_classical,
            }
            representation_audits = {}
            for representation_index, representation in enumerate(REPRESENTATIONS):
                train_source = _source_matrix(representation, train_h, train_c)
                val_source = _source_matrix(representation, val_h, val_c)
                q_train, q_val, representation_audit = _residual_angles(
                    train_source, val_source, residual, fit,
                    seed + held_out + representation_index * 100,
                )
                _seed_torch(seed + held_out + representation_index * 1000)
                vqc = TorchStatevectorQuantumClassifier(n_qubits=4, n_layers=2, topology="ring")
                parameter_count = sum(parameter.numel() for parameter in vqc.parameters())
                mlp, hidden = _matched_mlp(4, parameter_count)
                _, vqc_score, vqc_audit = _train_weighted_head(
                    vqc, q_train, y_fit, residual_weight, q_val,
                    epochs=epochs, batch_size=batch_size, learning_rate=0.01,
                    seed=seed + held_out + representation_index * 1000,
                    device_name=device,
                )
                _, mlp_score, mlp_audit = _train_weighted_head(
                    mlp, q_train, y_fit, residual_weight, q_val,
                    epochs=epochs, batch_size=min(128, max(32, batch_size)), learning_rate=0.01,
                    seed=seed + held_out + representation_index * 1000 + 500,
                    device_name=device,
                )
                fold_output[f"{representation}_vqc"] = expit(vqc_score)
                fold_output[f"{representation}_mlp"] = expit(mlp_score)
                classical_logit = _safe_logit(val_classical)
                for weight in FUSION_WEIGHTS:
                    suffix = str(weight).replace(".", "p")
                    fold_output[f"{representation}_vqc_fusion_{suffix}"] = expit(
                        classical_logit + weight * vqc_score
                    )
                    fold_output[f"{representation}_mlp_fusion_{suffix}"] = expit(
                        classical_logit + weight * mlp_score
                    )
                representation_audits[representation] = {
                    "representation": representation_audit,
                    "vqc": vqc_audit,
                    "matched_mlp": {**mlp_audit, "hidden_units": hidden},
                }
            if any(len(fold_output[name]) != len(val_y) or not np.isfinite(fold_output[name]).all() for name in model_names):
                raise RuntimeError("Incomplete or non-finite fold output")
            audit = {
                "held_out_fold": held_out,
                "training_records": int(len(train_y)),
                "training_patients": int(len(np.unique(train_patients))),
                "quantum_fit_records": int(len(fit)),
                "quantum_fit_patients": int(len(np.unique(train_patients[fit]))),
                "training_ecg_sha256": hashlib.sha256(np.sort(train_ids).tobytes()).hexdigest(),
                "validation_ecg_sha256": hashlib.sha256(np.sort(val_ids).tobytes()).hexdigest(),
                "classical_expert": classical_audit,
                "residual_mean": float(residual.mean()),
                "residual_std": float(residual.std()),
                "representations": representation_audits,
                "fusion_weights_prespecified": list(FUSION_WEIGHTS),
                "folds_9_and_10_accessed": False,
            }
            _atomic_npz(
                checkpoint,
                **fold_output,
                audit=json.dumps(audit, default=_json_default, sort_keys=True),
            )
        for key in keys:
            collected[key].append(fold_output[key])
        audits.append(audit)
        print(f"fold {held_out}: complete", flush=True)

    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    if len(np.unique(arrays["ecg_id"])) != len(arrays["ecg_id"]):
        raise ValueError("Duplicate OOF ECG")
    metrics = pd.DataFrame([
        _metric(name, arrays["label"], arrays[name], arrays["hard_negative"])
        for name in model_names
    ]).sort_values("auprc", ascending=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(arrays).to_csv(output_dir / "predictions_wide.csv", index=False)
    (output_dir / "fold_audits.json").write_text(json.dumps(audits, indent=2, default=_json_default))

    primary_suffix = "0p25"
    for representation in REPRESENTATIONS:
        quantum_name = f"{representation}_vqc_fusion_{primary_suffix}"
        mlp_name = f"{representation}_mlp_fusion_{primary_suffix}"
        for comparator in ("classical_expert", mlp_name):
            report = _paired_patient_bootstrap(
                arrays["label"], arrays["patient_id"], arrays[quantum_name], arrays[comparator],
                iterations=2000, seed=seed,
                comparison=f"{quantum_name}_minus_{comparator}",
            )
            filename = f"paired_{quantum_name}_vs_{comparator}.json"
            (output_dir / filename).write_text(json.dumps(report, indent=2, default=_json_default))
    manifest_output = {
        "experiment": "residual_feature_routing_quantum_fusion_screen",
        "representations": list(REPRESENTATIONS),
        "fusion_weights": list(FUSION_WEIGHTS),
        "primary_fusion_weight": 0.25,
        "records": int(len(arrays["label"])),
        "patients": int(len(np.unique(arrays["patient_id"]))),
        "folds": list(folds),
        "feature_manifest_sha256": manifest["manifest_sha256"],
        "fold_9_accessed": False,
        "fold_10_accessed": False,
        "decision_rule": "advance only if q4 fusion beats classical and matched-MLP fusion by patient-bootstrap AUPRC intervals above zero",
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(manifest_output, indent=2))
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual q4 fusion screen")
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--folds", type=int, nargs="*", default=list(range(1, 9)))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    run_screen(
        args.representations, args.metadata, args.features, args.manifest, args.output,
        per_class=args.per_class, epochs=args.epochs, batch_size=args.batch_size,
        seed=args.seed, folds=tuple(args.folds), device=args.device,
    )


if __name__ == "__main__":
    main()
