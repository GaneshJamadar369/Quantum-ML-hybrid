"""Patient-safe advanced multimodal representation screen with a q4 VQC core.

Each outer fold loads its frozen ECG patch Transformer, builds one of three
small classical fusion representations, compresses the learned h32 vector to
the same fold-local PLS-q4 coordinates, and compares a four-qubit VQC with
classical heads on exactly those coordinates.  Folds 9 and 10 remain sealed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.svm import SVC

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_advanced_fusion import (
    build_quantum_input_fusion,
    clinical_feature_groups,
)
from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier
from aquire_preprocessing.models_transformer import ECGPatchTransformer
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _patient_unique_sample
from run_waveform_representation_export import _load_normalizer, _normalise


def _seed(seed: int) -> None:
    import torch

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _read_h5_rows(handle, dataset: str, rows: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows while respecting h5py's sorted-index rule."""
    rows = np.asarray(rows, dtype=int)
    order = np.argsort(rows)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.asarray(handle[dataset][rows[order]])[inverse]


def _encode_tokens(
    checkpoint_path: Path,
    signals: np.ndarray,
    medians: np.ndarray,
    iqrs: np.ndarray,
    device,
    batch_size: int,
) -> np.ndarray:
    import torch

    model = ECGPatchTransformer().to(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    normalized = _normalise(signals.astype(np.float32), medians, iqrs)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
            outputs.append(model.encode_tokens(batch).to(torch.float16).cpu().numpy())
    result = np.concatenate(outputs, axis=0)
    if result.shape[1:] != (100, 96) or not np.isfinite(result).all():
        raise RuntimeError(f"Invalid Transformer token tensor {result.shape}")
    del model, normalized
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


class _FusionProbe:
    """Namespace wrapper built dynamically to keep torch an optional import."""

    @staticmethod
    def build(fusion_name: str, groups, width: int, dropout: float):
        import torch
        from torch import nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fusion = build_quantum_input_fusion(
                    fusion_name, groups, width=width, dropout=dropout
                )
                self.classifier = nn.Linear(width, 1)

            def forward(self, wave, clinical, observed):
                embedding = self.fusion(
                    wave, clinical, observed_mask=observed, return_embedding=True
                )
                return self.classifier(embedding).squeeze(-1), embedding

        return Model()


def _train_fusion(
    fusion_name: str,
    groups,
    train_tokens: np.ndarray,
    train_clinical: np.ndarray,
    train_observed: np.ndarray,
    train_labels: np.ndarray,
    train_hard: np.ndarray,
    val_tokens: np.ndarray,
    val_clinical: np.ndarray,
    val_observed: np.ndarray,
    *,
    width: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _seed(seed)
    model = _FusionProbe.build(fusion_name, groups, width, dropout).to(device)
    weights = np.ones(len(train_labels), dtype=np.float32)
    weights[(train_labels == 0) & train_hard] = 1.25
    dataset = TensorDataset(
        torch.from_numpy(train_tokens),
        torch.from_numpy(train_clinical.astype(np.float32)),
        torch.from_numpy(train_observed.astype(np.float32)),
        torch.from_numpy(train_labels.astype(np.float32)),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=False,
        generator=torch.Generator().manual_seed(seed), pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    for _ in range(epochs):
        model.train()
        running = 0.0
        seen = 0
        for tokens, clinical, observed, target, weight in loader:
            tokens = tokens.to(device, dtype=torch.float32, non_blocking=True)
            clinical = clinical.to(device, non_blocking=True)
            observed = observed.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            # Prespecified regularization: patch and clinical-feature dropout.
            patch_keep = (torch.rand(tokens.shape[:2], device=device) > 0.05).unsqueeze(-1)
            feature_keep = torch.rand(clinical.shape, device=device) > 0.05
            tokens = tokens * patch_keep
            clinical = clinical * feature_keep
            observed = observed * feature_keep
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tokens, clinical, observed)
            raw = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            loss = (raw * weight).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite fusion loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach()) * len(tokens)
            seen += len(tokens)
        scheduler.step()
        losses.append(running / max(seen, 1))

    def infer(tokens: np.ndarray, clinical: np.ndarray, observed: np.ndarray):
        model.eval()
        logits, embeddings = [], []
        with torch.inference_mode():
            for start in range(0, len(tokens), batch_size):
                token_batch = torch.from_numpy(tokens[start : start + batch_size]).to(
                    device, dtype=torch.float32
                )
                clinical_batch = torch.from_numpy(clinical[start : start + batch_size]).to(device)
                observed_batch = torch.from_numpy(observed[start : start + batch_size]).to(device)
                score, representation = model(token_batch, clinical_batch, observed_batch)
                logits.append(score.cpu().numpy())
                embeddings.append(representation.cpu().numpy())
        return np.concatenate(logits), np.concatenate(embeddings)

    train_logits, train_embeddings = infer(train_tokens, train_clinical, train_observed)
    val_logits, val_embeddings = infer(val_tokens, val_clinical, val_observed)
    zero_clinical_logits, _ = infer(
        val_tokens, np.zeros_like(val_clinical), np.zeros_like(val_observed)
    )
    zero_wave_logits, _ = infer(
        np.zeros_like(val_tokens), val_clinical, val_observed
    )
    audit = {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_declined": bool(losses[-1] < losses[0]),
    }
    return (
        train_logits, train_embeddings, val_logits, val_embeddings,
        zero_clinical_logits, zero_wave_logits, audit,
    )


def _fit_q4(train_h: np.ndarray, train_y: np.ndarray, val_h: np.ndarray, seed: int):
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    train_x = scaler.fit_transform(imputer.fit_transform(train_h))
    val_x = scaler.transform(imputer.transform(val_h))
    pls = PLSRegression(n_components=4, scale=False, max_iter=1000)
    train_q = pls.fit_transform(train_x, train_y)[0]
    val_q = pls.transform(val_x)
    signs = np.ones(4, dtype=float)
    correlations = []
    for component in range(4):
        correlation = float(np.corrcoef(train_q[:, component], train_y)[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
        if correlation < 0:
            signs[component] = -1.0
            train_q[:, component] *= -1.0
            val_q[:, component] *= -1.0
            correlation *= -1.0
        correlations.append(correlation)
    quantile = QuantileTransformer(
        n_quantiles=min(256, len(train_q)), output_distribution="uniform", random_state=seed
    )
    train_q = (2 * quantile.fit_transform(train_q) - 1) * (np.pi / 2)
    val_q = (2 * quantile.transform(val_q) - 1) * (np.pi / 2)
    return train_q.astype(np.float32), val_q.astype(np.float32), correlations


def _train_vqc(
    train_q: np.ndarray,
    train_y: np.ndarray,
    val_q: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    device,
    entanglement: bool = True,
):
    import torch

    _seed(seed)
    model = TorchStatevectorQuantumClassifier(4, n_layers=2, topology="ring").to(device)
    if not entanglement:
        with torch.no_grad():
            model.interactions.zero_()
        model.interactions.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    x = torch.from_numpy(train_q).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(train_q))
    for _ in range(epochs):
        rng.shuffle(indices)
        model.train()
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[batch]), y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(val_q).to(device)).cpu().numpy()
    return logits, int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    probability = expit(np.asarray(logits, dtype=float))
    return {
        "auprc": float(average_precision_score(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
    }


def run(args) -> None:
    import h5py
    import torch

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    features_frame = pd.read_csv(args.features).set_index("ecg_id")
    manifest = load_feature_manifest(args.manifest, features_frame.columns)
    approved = manifest["approved_features"]
    joined = metadata.join(features_frame[approved], how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")].copy()
    folds = joined.strat_fold.to_numpy(dtype=int)
    guard_fold_access(folds, purpose="tuning")
    if joined.groupby("patient_id").strat_fold.nunique().max() != 1:
        raise RuntimeError("Patient crosses development folds")
    labels = joined.mi_label.to_numpy(dtype=int)
    patients = joined.patient_id.to_numpy(dtype=int)
    hard = joined.hard_negative.to_numpy(dtype=bool)
    ids = joined.index.to_numpy(dtype=int)
    raw_features = joined[approved].to_numpy(dtype=np.float32)
    observed = np.isfinite(raw_features).astype(np.float32)
    groups = clinical_feature_groups(approved)
    with h5py.File(args.hdf5, "r") as handle:
        h5_ids = np.asarray(handle["ecg_id"], dtype=int)
        h5_map = {int(ecg_id): row for row, ecg_id in enumerate(h5_ids)}
    if not set(ids).issubset(h5_map):
        raise RuntimeError("HDF5 does not cover the aligned cohort")

    predictions = {
        name: np.full(len(joined), np.nan, dtype=float)
        for name in (
            "transformer", "fusion_probe", "fusion_zero_clinical", "fusion_zero_wave",
            "vqc", "vqc_no_entanglement", "q4_logistic", "q4_mlp", "q4_rbf",
        )
    }
    fold_audits = []
    for held_out_value in sorted(np.unique(folds)):
        held_out = int(held_out_value)
        started = time.time()
        train_global = np.flatnonzero(folds != held_out)
        val_global = np.flatnonzero(folds == held_out)
        train_pick_local = _patient_unique_sample(
            labels[train_global], hard[train_global], patients[train_global],
            per_class=args.fusion_per_class, seed=args.seed + held_out,
        )
        train_global = train_global[train_pick_local]
        checkpoint = args.transformer_dir / f"outer_fold_{held_out}_encoder.pt"
        rep_path = args.transformer_dir / f"outer_fold_{held_out}_representations.npz"
        normalizer_path = args.normalizers / f"normalizer_holdout_fold_{held_out}.json"
        for required in (checkpoint, rep_path, normalizer_path):
            if not required.exists():
                raise FileNotFoundError(required)
        with np.load(rep_path, allow_pickle=False) as rep:
            rep_val_ids = rep["val_record_ids"].astype(int)
            if set(rep_val_ids) != set(ids[val_global]):
                raise RuntimeError(f"Transformer validation ID mismatch in fold {held_out}")
            rep_order = {record_id: row for row, record_id in enumerate(rep_val_ids)}
            predictions["transformer"][val_global] = rep["val_logits"][
                [rep_order[int(record_id)] for record_id in ids[val_global]]
            ]
        medians, iqrs, normalizer_audit = _load_normalizer(normalizer_path)
        expected_training_folds = sorted(int(x) for x in np.unique(folds[folds != held_out]))
        if normalizer_audit.get("folds_used") != expected_training_folds:
            raise RuntimeError(f"Normalizer fold mismatch in fold {held_out}")

        selected = np.r_[train_global, val_global]
        rows = np.asarray([h5_map[int(record_id)] for record_id in ids[selected]], dtype=int)
        with h5py.File(args.hdf5, "r") as handle:
            signals = _read_h5_rows(handle, "accepted_signal", rows).astype(np.float32)
        if signals.shape[1:] == (1000, 12):
            signals = np.transpose(signals, (0, 2, 1))
        tokens = _encode_tokens(
            checkpoint, signals, medians, iqrs, device, args.encoder_batch_size
        )
        train_tokens = tokens[: len(train_global)]
        val_tokens = tokens[len(train_global) :]
        del tokens, signals

        # Imputation/scaling uses every outer-training record and no held-out record.
        outer_train_raw = raw_features[folds != held_out]
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        scaler = RobustScaler(quantile_range=(25.0, 75.0))
        scaler.fit(imputer.fit_transform(outer_train_raw))
        train_clinical = scaler.transform(imputer.transform(raw_features[train_global])).astype(np.float32)
        val_clinical = scaler.transform(imputer.transform(raw_features[val_global])).astype(np.float32)

        result = _train_fusion(
            args.fusion, groups, train_tokens, train_clinical, observed[train_global],
            labels[train_global], hard[train_global], val_tokens, val_clinical,
            observed[val_global], width=args.width, dropout=args.dropout,
            epochs=args.fusion_epochs, batch_size=args.batch_size,
            seed=args.seed + 100 * held_out, device=device,
        )
        train_probe, train_h, val_probe, val_h, zero_clinical, zero_wave, fusion_audit = result
        predictions["fusion_probe"][val_global] = val_probe
        predictions["fusion_zero_clinical"][val_global] = zero_clinical
        predictions["fusion_zero_wave"][val_global] = zero_wave

        train_q, val_q, correlations = _fit_q4(
            train_h, labels[train_global], val_h, args.seed + held_out
        )
        quantum_pick = _patient_unique_sample(
            labels[train_global], hard[train_global], patients[train_global],
            per_class=args.quantum_per_class, seed=args.seed + 1000 + held_out,
        )
        q_fit = train_q[quantum_pick]
        y_fit = labels[train_global][quantum_pick]
        vqc, vqc_parameters = _train_vqc(
            q_fit, y_fit, val_q, epochs=args.quantum_epochs,
            batch_size=args.batch_size, seed=args.seed + 2000 + held_out,
            device=device, entanglement=True,
        )
        no_entanglement, no_ent_parameters = _train_vqc(
            q_fit, y_fit, val_q, epochs=args.quantum_epochs,
            batch_size=args.batch_size, seed=args.seed + 2000 + held_out,
            device=device, entanglement=False,
        )
        predictions["vqc"][val_global] = vqc
        predictions["vqc_no_entanglement"][val_global] = no_entanglement
        logistic = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        logistic.fit(q_fit, y_fit)
        predictions["q4_logistic"][val_global] = logistic.decision_function(val_q)
        mlp = MLPClassifier(
            hidden_layer_sizes=(7,), activation="relu", max_iter=500,
            random_state=args.seed + held_out,
        )
        mlp.fit(q_fit, y_fit)
        predictions["q4_mlp"][val_global] = np.log(
            np.clip(mlp.predict_proba(val_q)[:, 1], 1e-5, 1 - 1e-5)
            / np.clip(mlp.predict_proba(val_q)[:, 0], 1e-5, 1 - 1e-5)
        )
        rbf = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")
        rbf.fit(q_fit, y_fit)
        predictions["q4_rbf"][val_global] = rbf.decision_function(val_q)
        fold_audits.append(
            {
                "fold": held_out,
                "fusion_training_records": int(len(train_global)),
                "quantum_training_records": int(len(quantum_pick)),
                "validation_records": int(len(val_global)),
                "fusion": fusion_audit,
                "pls_component_label_correlations": correlations,
                "vqc_parameters": vqc_parameters,
                "no_entanglement_parameters": no_ent_parameters,
                "runtime_seconds": time.time() - started,
            }
        )
        print(
            f"fold {held_out}: probe={average_precision_score(labels[val_global], expit(val_probe)):.4f} "
            f"vqc={average_precision_score(labels[val_global], expit(vqc)):.4f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        del train_tokens, val_tokens, train_h, val_h
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("Incomplete OOF predictions")
    metrics = {name: _metrics(labels, values) for name, values in predictions.items()}
    comparisons = {}
    for control in ("q4_logistic", "q4_mlp", "q4_rbf", "vqc_no_entanglement"):
        comparisons[f"vqc_minus_{control}"] = _paired_patient_bootstrap(
            labels, patients, expit(predictions["vqc"]), expit(predictions[control]),
            iterations=args.bootstrap_iterations, seed=args.seed, comparison=f"vqc_minus_{control}",
        )
    comparisons["fusion_minus_transformer"] = _paired_patient_bootstrap(
        labels, patients, expit(predictions["fusion_probe"]), expit(predictions["transformer"]),
        iterations=args.bootstrap_iterations, seed=args.seed,
        comparison="fusion_minus_transformer",
    )
    best_control = max(("q4_logistic", "q4_mlp", "q4_rbf"), key=lambda name: metrics[name]["auprc"])
    q_delta = metrics["vqc"]["auprc"] - metrics[best_control]["auprc"]
    q_ci = comparisons[f"vqc_minus_{best_control}"]["delta_auprc"]
    representation_ci = comparisons["fusion_minus_transformer"]["delta_auprc"]
    verdict = {
        "fusion": args.fusion,
        "representation_gate": bool(
            metrics["fusion_probe"]["auprc"] - metrics["transformer"]["auprc"] >= 0.005
            and representation_ci["ci95_low"] > 0
            and metrics["fusion_probe"]["auprc"] > metrics["fusion_zero_clinical"]["auprc"]
            and metrics["fusion_probe"]["auprc"] > metrics["fusion_zero_wave"]["auprc"]
        ),
        "quantum_gate": bool(q_delta >= 0.005 and q_ci["ci95_low"] > 0),
        "best_identical_q4_control": best_control,
        "vqc_delta_auprc": q_delta,
        "claim_boundary": "A simulator accuracy comparison cannot establish computational quantum advantage.",
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    pd.DataFrame(
        {
            "ecg_id": ids, "patient_id": patients, "strat_fold": folds, "y_true": labels,
            **{name: expit(values) for name, values in predictions.items()},
        }
    ).to_csv(output / "oof_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "bootstrap.json").write_text(json.dumps(comparisons, indent=2))
    (output / "fold_audits.json").write_text(json.dumps(fold_audits, indent=2))
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"metrics": metrics, "verdict": verdict}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--transformer-dir", type=Path, required=True)
    parser.add_argument("--normalizers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fusion", choices=["film", "lmf", "cross_attention"], required=True)
    parser.add_argument("--fusion-per-class", type=int, default=2000)
    parser.add_argument("--quantum-per-class", type=int, default=500)
    parser.add_argument("--fusion-epochs", type=int, default=10)
    parser.add_argument("--quantum-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--encoder-batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260924)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
