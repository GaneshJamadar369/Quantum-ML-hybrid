"""Independent dual-route classical/quantum fusion screen.

Route A: 106 approved clinical features → HistGB  ∥  Transformer h128 → PLS-q4 → VQC
Route B: 60 QRS/rhythm features → HistGB         ∥  46 ST/T features → PLS-q4 → VQC

Both branches predict MI directly.  No residual target, no classical-error
weighting, no uncertainty gating.  The fusion is a cross-fitted one-neuron
logistic regression on held-out branch logits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.svm import SVC

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier, ReuploadingQuantumClassifier
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import (
    _atomic_npz,
    _json_default,
    _patient_unique_sample,
    _seed_torch,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_route_config(path: Path) -> dict:
    with open(path) as f:
        config = json.load(f)
    assert config["protocol"] == "independent_dual_route_fusion_v1"
    return config


# ---------------------------------------------------------------------------
# Reusable helpers
# ---------------------------------------------------------------------------

def _safe_logit(p: np.ndarray) -> np.ndarray:
    return logit(np.clip(np.asarray(p, dtype=float), 1e-5, 1.0 - 1e-5))


def _classical_expert(features: np.ndarray, labels: np.ndarray, seed: int):
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


def _fit_q4_representation(
    features: np.ndarray,
    labels: np.ndarray,
    fit_idx: np.ndarray,
    transform_idx: np.ndarray,
    n_components: int = 4,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fold-local supervised PLS + quantile angle mapping."""
    seed = int(seed)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    x_fit = scaler.fit_transform(imputer.fit_transform(features[fit_idx]))
    x_transform = scaler.transform(imputer.transform(features[transform_idx]))
    pls = PLSRegression(n_components=n_components, scale=False, max_iter=1000)
    q_fit = pls.fit_transform(x_fit, labels[fit_idx])[0]
    q_transform = pls.transform(x_transform)
    component_correlations = []
    for component in range(n_components):
        correlation = float(np.corrcoef(q_fit[:, component], labels[fit_idx])[0, 1])
        if not np.isfinite(correlation):
            correlation = 0.0
        if correlation < 0:
            q_fit[:, component] *= -1.0
            q_transform[:, component] *= -1.0
            correlation *= -1.0
        component_correlations.append(correlation)
    quantile = QuantileTransformer(
        n_quantiles=min(256, len(q_fit)),
        output_distribution="uniform",
        random_state=seed,
    )
    # The frozen independent-route protocol uses the narrower [-pi/2, pi/2]
    # interval to avoid saturating every input rotation at the extremes.
    q_fit = (2.0 * quantile.fit_transform(q_fit) - 1.0) * (np.pi / 2.0)
    q_transform = (2.0 * quantile.transform(q_transform) - 1.0) * (np.pi / 2.0)
    audit = {
        "method": "train-only median + robust scale + supervised PLS + quantile angles",
        "fit_records": int(len(fit_idx)),
        "n_components": int(n_components),
        "q_fit_range": [float(q_fit.min()), float(q_fit.max())],
        "component_label_correlations": component_correlations,
    }
    return q_fit.astype(np.float32), q_transform.astype(np.float32), audit


def _train_vqc(
    q_train: np.ndarray,
    y_train: np.ndarray,
    y_soft: np.ndarray,
    q_val: np.ndarray,
    n_qubits: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
    alpha: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Train a Reuploading VQC using Knowledge Distillation."""
    import torch

    seed = int(seed)
    _seed_torch(seed)
    model = ReuploadingQuantumClassifier(
        input_dim=8, n_qubits=n_qubits, topology="ring"
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
    ).to(device)
    criterion_hard = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_soft = torch.nn.BCEWithLogitsLoss()

    x_t = torch.tensor(q_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_s = torch.tensor(y_soft, dtype=torch.float32).to(device)

    indices = np.arange(len(q_train))
    rng = np.random.default_rng(seed)
    losses = []
    gradients = []
    for epoch in range(epochs):
        rng.shuffle(indices)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_gradients = []
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x_t[batch])
            loss_hard = criterion_hard(logits, y_t[batch])
            loss_soft = criterion_soft(logits, y_s[batch])
            loss = (1.0 - alpha) * loss_hard + alpha * loss_soft
            loss.backward()
            gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0))
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
            epoch_gradients.append(gradient)
        losses.append(epoch_loss / max(n_batches, 1))
        gradients.append(float(np.median(epoch_gradients)))

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(q_val, dtype=torch.float32).to(device))
        val_logits = val_logits.cpu().numpy()

    audit = {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "initial_loss": losses[0] if losses else float("nan"),
        "final_loss": losses[-1] if losses else float("nan"),
        "loss_declined": bool(len(losses) >= 2 and losses[-1] < losses[0]),
        "gradient_norm_min": float(np.min(gradients)),
        "gradient_norm_max": float(np.max(gradients)),
        "epochs": epochs,
        "simulator": "exact_torch_statevector",
        "device": device,
    }
    return val_logits, audit


def _train_mlp_control(
    q_train: np.ndarray,
    y_train: np.ndarray,
    q_val: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Parameter-count-matched MLP on same q4 coordinates."""
    seed = int(seed)
    mlp = MLPClassifier(
        # q4 two-layer statevector VQC has 45 trainable parameters; seven
        # hidden units give this 4->7->1 MLP 43 parameters.
        hidden_layer_sizes=(7,),
        activation="relu",
        max_iter=500,
        random_state=seed,
    )
    mlp.fit(q_train, y_train)
    return _safe_logit(mlp.predict_proba(q_val)[:, 1])


def _kernel_controls(
    q_train: np.ndarray,
    y_train: np.ndarray,
    q_val: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """RBF and Laplacian SVC controls on the identical q4 coordinates."""
    seed = int(seed)
    rbf = SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed)
    rbf.fit(q_train, y_train)
    rbf_score = np.asarray(rbf.decision_function(q_val), dtype=float)

    distances = pdist(q_train, metric="cityblock")
    positive = distances[distances > 1e-12]
    gamma = 1.0 / (float(np.median(positive)) if len(positive) else 1.0)
    train_kernel = np.exp(-gamma * cdist(q_train, q_train, metric="cityblock"))
    val_kernel = np.exp(-gamma * cdist(q_val, q_train, metric="cityblock"))
    laplacian = SVC(C=1.0, kernel="precomputed", class_weight="balanced", random_state=seed)
    laplacian.fit(train_kernel, y_train)
    laplacian_score = np.asarray(laplacian.decision_function(val_kernel), dtype=float)
    return rbf_score, laplacian_score


def _fit_nonnegative_logistic(x: np.ndarray, y: np.ndarray, c_value: float) -> np.ndarray:
    """Intercept plus nonnegative branch weights with L2 regularization."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    l2 = 1.0 / float(c_value)

    def objective(parameters):
        intercept, weights = parameters[0], parameters[1:]
        logits = intercept + x @ weights
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits)
        loss += 0.5 * l2 * float(weights @ weights) / len(y)
        error = expit(logits) - y
        gradient = np.r_[
            np.mean(error),
            (x.T @ error) / len(y) + (l2 / len(y)) * weights,
        ]
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(x.shape[1] + 1),
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None)] + [(0.0, None)] * x.shape[1],
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Fusion optimization failed: {result.message}")
    return result.x


def _cross_fitted_fusion(
    s_c: np.ndarray,
    s_q: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    """Cross-fitted nonnegative logistic fusion with inner L2 selection."""
    fusion_logits = np.full(len(labels), np.nan)
    coef_records = []
    unique_folds = sorted(np.unique(folds).astype(int))

    for meta_fold in unique_folds:
        meta_val = folds == meta_fold
        meta_train = ~meta_val

        X_train_raw = np.column_stack([s_c[meta_train], s_q[meta_train]])
        X_val_raw = np.column_stack([s_c[meta_val], s_q[meta_val]])
        y_train = labels[meta_train]
        train_folds = folds[meta_train]

        candidate_losses = {}
        for c_value in (0.1, 1.0, 10.0):
            losses = []
            for inner_fold in sorted(np.unique(train_folds).astype(int)):
                inner_val = train_folds == inner_fold
                inner_train = ~inner_val
                inner_mu = X_train_raw[inner_train].mean(axis=0)
                inner_sigma = X_train_raw[inner_train].std(axis=0).clip(1e-6)
                inner_x = (X_train_raw[inner_train] - inner_mu) / inner_sigma
                inner_validation = (X_train_raw[inner_val] - inner_mu) / inner_sigma
                parameters = _fit_nonnegative_logistic(
                    inner_x, y_train[inner_train], c_value
                )
                probability = expit(parameters[0] + inner_validation @ parameters[1:])
                losses.append(log_loss(y_train[inner_val], probability))
            candidate_losses[c_value] = float(np.mean(losses))
        selected_c = min(candidate_losses, key=candidate_losses.get)

        # Standardize from meta-training only
        mu = X_train_raw.mean(axis=0)
        sigma = X_train_raw.std(axis=0).clip(1e-6)
        X_train = (X_train_raw - mu) / sigma
        X_val = (X_val_raw - mu) / sigma
        parameters = _fit_nonnegative_logistic(X_train, y_train, selected_c)
        fusion_logits[meta_val] = parameters[0] + X_val @ parameters[1:]
        coef_records.append({
            "meta_fold": int(meta_fold),
            "selected_c": float(selected_c),
            "candidate_logloss": candidate_losses,
            "intercept": float(parameters[0]),
            "beta_c": float(parameters[1]),
            "beta_q": float(parameters[2]),
        })

    if not np.isfinite(fusion_logits).all():
        raise RuntimeError("Incomplete cross-fitted fusion output")
    return fusion_logits, coef_records


def _metrics(labels: np.ndarray, logits: np.ndarray, name: str) -> dict:
    prob = expit(logits)
    return {
        "model": name,
        "auprc": float(average_precision_score(labels, prob)),
        "auroc": float(roc_auc_score(labels, prob)),
        "brier": float(brier_score_loss(labels, prob)),
        "logloss": float(log_loss(labels, prob)),
    }


# ---------------------------------------------------------------------------
# Main outer-fold loop
# ---------------------------------------------------------------------------

def run_independent_dual_route_screen(
    representations_dir: Path,
    metadata_path: Path,
    features_csv: Path,
    manifest_path: Path,
    route_config_path: Path,
    output_dir: Path,
    per_class: int = 500,
    epochs: int = 20,
    batch_size: int = 128,
    seed: int = 42,
    device: str = "cpu",
    preflight_only: bool = False,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_route_config(route_config_path)

    # Load metadata
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")
    features_raw = pd.read_csv(features_csv).set_index("ecg_id")
    manifest = load_feature_manifest(manifest_path, features_raw.columns)
    approved = manifest["approved_features"]
    if config.get("source_manifest_hash") != manifest["manifest_sha256"]:
        raise ValueError("Independent route config is not bound to this feature manifest")
    features_raw = features_raw[approved]

    joined = metadata.join(features_raw, how="inner", validate="one_to_one")
    joined = joined[
        joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")
    ]

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

    # Route B feature splits
    route_b_classical = config["route_b"]["classical_features"]
    route_b_quantum = config["route_b"]["quantum_features"]
    missing_route_features = sorted(
        (set(route_b_classical) | set(route_b_quantum)) - set(approved)
    )
    if missing_route_features:
        raise ValueError(f"Route B uses unapproved features: {missing_route_features}")
    if set(route_b_classical) & set(route_b_quantum):
        raise ValueError("Route B feature families overlap")
    if set(route_b_classical) | set(route_b_quantum) != set(approved):
        raise ValueError("Route B must partition the complete approved manifest")
    if len(route_b_classical) != 60 or len(route_b_quantum) != 46:
        raise ValueError("Frozen Route B must contain exactly 60 and 46 features")
    all_features_mat = joined[approved].to_numpy(dtype=np.float32)
    route_b_c_idx = [approved.index(f) for f in route_b_classical if f in approved]
    route_b_q_idx = [approved.index(f) for f in route_b_quantum if f in approved]

    # Verify disjointness
    assert set(route_b_c_idx).isdisjoint(set(route_b_q_idx)), "Route B features overlap!"
    print(f"Route B: {len(route_b_c_idx)} classical, {len(route_b_q_idx)} quantum features", flush=True)

    # Validate the complete fold-specific Transformer contract before any
    # expensive training.  Route A must use train and validation embeddings
    # produced by the same outer-fold encoder; a pooled OOF h128 matrix would
    # mix eight different representation spaces inside one PLS fit.
    print("Validating Transformer representation archives...", flush=True)
    record_to_row = {int(ecg_id): row for row, ecg_id in enumerate(record_ids)}
    representation_paths = {}
    seen_validation_ids = []
    required_representation_keys = {
        "train_record_ids", "val_record_ids", "train_patient_ids", "val_patient_ids",
        "train_labels", "val_labels", "train_embeddings", "val_embeddings",
    }
    for fold_id in sorted(np.unique(folds)):
        rep_path = representations_dir / f"outer_fold_{fold_id}_representations.npz"
        if not rep_path.exists():
            raise FileNotFoundError(f"Missing representation: {rep_path}")
        with np.load(rep_path, allow_pickle=False) as rep:
            missing = sorted(required_representation_keys - set(rep.files))
            if missing:
                raise KeyError(
                    f"{rep_path.name} missing {missing}; available keys={rep.files}"
                )
            train_ids = rep["train_record_ids"].astype(int)
            val_ids = rep["val_record_ids"].astype(int)
            train_patients = rep["train_patient_ids"].astype(int)
            val_patients = rep["val_patient_ids"].astype(int)
            if len(set(train_patients) & set(val_patients)):
                raise ValueError(f"Patient leakage in representation fold {fold_id}")
            if set(train_ids) != set(record_ids[folds != fold_id].astype(int)):
                raise ValueError(f"Training ID mismatch in representation fold {fold_id}")
            if set(val_ids) != set(record_ids[folds == fold_id].astype(int)):
                raise ValueError(f"Validation ID mismatch in representation fold {fold_id}")
            if not np.array_equal(joined.loc[train_ids].mi_label.to_numpy(int), rep["train_labels"]):
                raise ValueError(f"Training label mismatch in representation fold {fold_id}")
            if not np.array_equal(joined.loc[val_ids].mi_label.to_numpy(int), rep["val_labels"]):
                raise ValueError(f"Validation label mismatch in representation fold {fold_id}")
            if rep["train_embeddings"].shape != (len(train_ids), 128):
                raise ValueError(f"Unexpected training h128 shape in fold {fold_id}")
            if rep["val_embeddings"].shape != (len(val_ids), 128):
                raise ValueError(f"Unexpected validation h128 shape in fold {fold_id}")
            seen_validation_ids.extend(val_ids.tolist())
        representation_paths[int(fold_id)] = rep_path
    if len(seen_validation_ids) != len(set(seen_validation_ids)) or set(seen_validation_ids) != set(record_ids.astype(int)):
        raise ValueError("Transformer OOF coverage is incomplete or duplicated")
    print("Transformer archives passed schema, label, patient and OOF coverage checks", flush=True)
    if preflight_only:
        report = {
            "records": int(len(record_ids)),
            "patients": int(len(np.unique(patient_ids))),
            "approved_features": int(len(approved)),
            "route_b_classical_features": int(len(route_b_c_idx)),
            "route_b_quantum_features": int(len(route_b_q_idx)),
            "folds": sorted(np.unique(folds).astype(int).tolist()),
            "fold_9_accessed": False,
            "fold_10_accessed": False,
        }
        (output_dir / "preflight.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
        return

    # ===================================================================
    # ROUTE A — representation-disjoint
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ROUTE A: Clinical(106) → HistGB  ∥  h128 → PLS-q4 → VQC", flush=True)
    print("=" * 70, flush=True)

    # A0: Classical reference (all 106 features)
    s_c_a = np.full(len(labels), np.nan)
    for held_out_value in sorted(np.unique(folds)):
        held_out = int(held_out_value)
        train_mask = folds != held_out
        val_mask = folds == held_out
        clf = _classical_expert(all_features_mat[train_mask], labels[train_mask], seed)
        clf.fit(all_features_mat[train_mask], labels[train_mask])
        s_c_a[val_mask] = _safe_logit(clf.predict_proba(all_features_mat[val_mask])[:, 1])
    print(f"A0 (Classical 106): {_metrics(labels, s_c_a, 'A0')}", flush=True)

    # A1: Quantum branch (h128 → PLS-q4 → VQC)
    s_q_a_vqc = np.full(len(labels), np.nan)
    s_q_a_mlp = np.full(len(labels), np.nan)
    s_q_a_logistic = np.full(len(labels), np.nan)
    s_q_a_rbf = np.full(len(labels), np.nan)
    s_q_a_laplacian = np.full(len(labels), np.nan)
    quantum_audits_a = []

    for held_out_value in sorted(np.unique(folds)):
        held_out = int(held_out_value)
        print(f"\n  Route A Fold {held_out}:", flush=True)
        with np.load(representation_paths[int(held_out)], allow_pickle=False) as rep:
            train_ids = rep["train_record_ids"].astype(int)
            val_ids = rep["val_record_ids"].astype(int)
            train_h = rep["train_embeddings"].astype(np.float32)
            val_h = rep["val_embeddings"].astype(np.float32)
            train_y = rep["train_labels"].astype(int)
            train_patients = rep["train_patient_ids"].astype(int)
        train_meta = joined.loc[train_ids]
        val_idx = np.asarray([record_to_row[int(ecg_id)] for ecg_id in val_ids], dtype=int)

        # Subsample for quantum
        sample_idx = _patient_unique_sample(
            train_y, train_meta.hard_negative.to_numpy(bool), train_patients,
            per_class=per_class, seed=seed + held_out,
        )

        # PLS is fit on the complete outer-training representation from this
        # fold's encoder; the VQC and all matched controls use the same
        # patient-unique balanced subset of those resulting coordinates.
        combined_h = np.vstack([train_h, val_h])
        # Validation labels are deliberately absent from the reducer fit; the
        # placeholder tail is never indexed by ``train_rows``.
        combined_y = np.r_[train_y, np.zeros(len(val_h), dtype=int)]
        train_rows = np.arange(len(train_h))
        val_rows = np.arange(len(train_h), len(combined_h))
        q_train_all, q_val, rep_audit = _fit_q4_representation(
            combined_h, combined_y, train_rows, val_rows, n_components=8,
            seed=seed + held_out,
        )
        q_train = q_train_all[sample_idx]
        y_train_q = train_y[sample_idx]

        # Generate Soft Labels for Knowledge Distillation
        train_mask = folds != held_out
        clf = _classical_expert(all_features_mat[train_mask], labels[train_mask], seed)
        clf.fit(all_features_mat[train_mask], labels[train_mask])
        
        # We need the classical features for the exact subset `sample_idx`.
        # Note: `sample_idx` is indexed relative to `train_y` (which corresponds to `train_mask`).
        c_train_q = all_features_mat[train_mask][sample_idx]
        y_soft_q = clf.predict_proba(c_train_q)[:, 1]

        # VQC
        vqc_logits, vqc_audit = _train_vqc(
            q_train, y_train_q, y_soft_q, q_val,
            n_qubits=4, epochs=epochs, batch_size=batch_size,
            seed=seed + held_out * 100, device=device,
        )
        s_q_a_vqc[val_idx] = vqc_logits

        # MLP control
        s_q_a_mlp[val_idx] = _train_mlp_control(
            q_train, y_train_q, q_val, seed + held_out * 100 + 1
        )

        # Logistic control
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        lr.fit(q_train, y_train_q)
        s_q_a_logistic[val_idx] = lr.decision_function(q_val)
        rbf_score, laplacian_score = _kernel_controls(
            q_train, y_train_q, q_val, seed + held_out
        )
        s_q_a_rbf[val_idx] = rbf_score
        s_q_a_laplacian[val_idx] = laplacian_score

        quantum_audits_a.append({
            "fold": int(held_out),
            "representation": rep_audit,
            "vqc": vqc_audit,
        })
        print(f"    VQC val AUPRC: {average_precision_score(labels[val_idx], expit(vqc_logits)):.4f}", flush=True)

    print(f"\nA1 (VQC standalone):       {_metrics(labels, s_q_a_vqc, 'A1_VQC')}", flush=True)
    print(f"A2 (MLP control):          {_metrics(labels, s_q_a_mlp, 'A2_MLP')}", flush=True)
    print(f"A2 (Logistic control):     {_metrics(labels, s_q_a_logistic, 'A2_Logistic')}", flush=True)
    print(f"A2 (RBF control):          {_metrics(labels, s_q_a_rbf, 'A2_RBF')}", flush=True)
    print(f"A2 (Laplacian control):    {_metrics(labels, s_q_a_laplacian, 'A2_Laplacian')}", flush=True)

    # A3: Fusion (classical + VQC)
    a3_logits, a3_coefs = _cross_fitted_fusion(s_c_a, s_q_a_vqc, labels, folds)
    print(f"A3 (Fusion C+VQC):         {_metrics(labels, a3_logits, 'A3_Fusion')}", flush=True)

    # A4: Fusion (classical + MLP) — all-classical control
    a4_logits, a4_coefs = _cross_fitted_fusion(s_c_a, s_q_a_mlp, labels, folds)
    print(f"A4 (Fusion C+MLP):         {_metrics(labels, a4_logits, 'A4_AllClassical')}", flush=True)

    # A5: Quantum-removal control (shuffle sQ)
    rng = np.random.default_rng(seed)
    s_q_a_shuffled = s_q_a_vqc.copy()
    for fold_id in np.unique(folds):
        rows = np.where(folds == fold_id)[0]
        s_q_a_shuffled[rows] = s_q_a_vqc[rng.permutation(rows)]
    a5_logits, _ = _cross_fitted_fusion(s_c_a, s_q_a_shuffled, labels, folds)
    print(f"A5 (Fusion C+shuffled):    {_metrics(labels, a5_logits, 'A5_Shuffle')}", flush=True)

    # ===================================================================
    # ROUTE B — clinically-disjoint
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ROUTE B: QRS/rhythm(60) → HistGB  ∥  ST/T(46) → PLS-q4 → VQC", flush=True)
    print("=" * 70, flush=True)

    features_b_c = all_features_mat[:, route_b_c_idx]
    features_b_q = all_features_mat[:, route_b_q_idx]

    # B0: Classical half (QRS/rhythm only)
    s_c_b = np.full(len(labels), np.nan)
    for held_out_value in sorted(np.unique(folds)):
        held_out = int(held_out_value)
        train_mask = folds != held_out
        val_mask = folds == held_out
        clf = _classical_expert(features_b_c[train_mask], labels[train_mask], seed)
        clf.fit(features_b_c[train_mask], labels[train_mask])
        s_c_b[val_mask] = _safe_logit(clf.predict_proba(features_b_c[val_mask])[:, 1])
    print(f"B0 (Classical QRS/rhythm): {_metrics(labels, s_c_b, 'B0')}", flush=True)

    # B1: Quantum half (ST/T → PLS-q4 → VQC)
    s_q_b_vqc = np.full(len(labels), np.nan)
    s_q_b_mlp = np.full(len(labels), np.nan)
    s_q_b_logistic = np.full(len(labels), np.nan)
    s_q_b_rbf = np.full(len(labels), np.nan)
    s_q_b_laplacian = np.full(len(labels), np.nan)

    for held_out_value in sorted(np.unique(folds)):
        held_out = int(held_out_value)
        print(f"\n  Route B Fold {held_out}:", flush=True)
        train_mask = folds != held_out
        val_mask = folds == held_out
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        sample_idx = _patient_unique_sample(
            labels[train_idx], hard_neg[train_idx], patient_ids[train_idx],
            per_class=per_class, seed=seed + held_out,
        )
        q_train_all, q_val, _ = _fit_q4_representation(
            features_b_q, labels, train_idx, val_idx, n_components=8,
            seed=seed + held_out,
        )
        q_train = q_train_all[sample_idx]
        y_train_q = labels[train_idx][sample_idx]

        # Generate Soft Labels for Route B (using QRS/rhythm features)
        clf_b = _classical_expert(features_b_c[train_mask], labels[train_mask], seed)
        clf_b.fit(features_b_c[train_mask], labels[train_mask])
        c_train_q_b = features_b_c[train_mask][sample_idx]
        y_soft_q_b = clf_b.predict_proba(c_train_q_b)[:, 1]

        vqc_logits, _ = _train_vqc(
            q_train, y_train_q, y_soft_q_b, q_val,
            n_qubits=4, epochs=epochs, batch_size=batch_size,
            seed=seed + held_out * 100 + 10, device=device,
        )
        s_q_b_vqc[val_idx] = vqc_logits
        s_q_b_mlp[val_idx] = _train_mlp_control(
            q_train, y_train_q, q_val, seed + held_out * 100 + 11
        )

        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        lr.fit(q_train, y_train_q)
        s_q_b_logistic[val_idx] = lr.decision_function(q_val)
        rbf_score, laplacian_score = _kernel_controls(
            q_train, y_train_q, q_val, seed + held_out + 1000
        )
        s_q_b_rbf[val_idx] = rbf_score
        s_q_b_laplacian[val_idx] = laplacian_score

        print(f"    VQC val AUPRC: {average_precision_score(labels[val_idx], expit(vqc_logits)):.4f}", flush=True)

    print(f"\nB1 (VQC ST/T standalone):  {_metrics(labels, s_q_b_vqc, 'B1_VQC')}", flush=True)
    print(f"B1 (MLP control):          {_metrics(labels, s_q_b_mlp, 'B1_MLP')}", flush=True)
    print(f"B1 (Logistic control):     {_metrics(labels, s_q_b_logistic, 'B1_Logistic')}", flush=True)
    print(f"B1 (RBF control):          {_metrics(labels, s_q_b_rbf, 'B1_RBF')}", flush=True)
    print(f"B1 (Laplacian control):    {_metrics(labels, s_q_b_laplacian, 'B1_Laplacian')}", flush=True)

    # B2: Fusion (QRS/rhythm + ST/T VQC)
    b2_logits, b2_coefs = _cross_fitted_fusion(s_c_b, s_q_b_vqc, labels, folds)
    print(f"B2 (Fusion QRS+VQC):       {_metrics(labels, b2_logits, 'B2_Fusion')}", flush=True)

    # B3: Oracle (all 106 HistGB, same as A0)
    print(f"B3 (All-feature oracle):   {_metrics(labels, s_c_a, 'B3_Oracle')}", flush=True)

    # B4: Route-swap (ST/T → classical, QRS/rhythm → VQC)
    s_c_b4 = np.full(len(labels), np.nan)
    s_q_b4 = np.full(len(labels), np.nan)
    for held_out_value in sorted(np.unique(folds)):
        held_out = int(held_out_value)
        train_mask = folds != held_out
        val_mask = folds == held_out
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        # Classical on ST/T
        clf = _classical_expert(features_b_q[train_mask], labels[train_mask], seed)
        clf.fit(features_b_q[train_mask], labels[train_mask])
        s_c_b4[val_mask] = _safe_logit(clf.predict_proba(features_b_q[val_mask])[:, 1])

        # VQC on QRS/rhythm
        sample_idx = _patient_unique_sample(
            labels[train_idx], hard_neg[train_idx], patient_ids[train_idx],
            per_class=per_class, seed=seed + held_out,
        )
        q_train_all, q_val, _ = _fit_q4_representation(
            features_b_c, labels, train_idx, val_idx, n_components=8,
            seed=seed + held_out + 2000,
        )
        q_train = q_train_all[sample_idx]
        y_train_q = labels[train_idx][sample_idx]
        
        # Generate Soft Labels using ST/T (which is the classical route for B4)
        c_train_q_swap = features_b_q[train_mask][sample_idx]
        y_soft_q_swap = clf.predict_proba(c_train_q_swap)[:, 1]

        vqc_logits, _ = _train_vqc(
            q_train, y_train_q, y_soft_q_swap, q_val,
            n_qubits=4, epochs=epochs, batch_size=batch_size,
            seed=seed + held_out * 100 + 20, device=device,
        )
        s_q_b4[val_idx] = vqc_logits

    b4_logits, _ = _cross_fitted_fusion(s_c_b4, s_q_b4, labels, folds)
    print(f"B4 (Route-swap fusion):    {_metrics(labels, b4_logits, 'B4_Swap')}", flush=True)

    # B5: All-classical split (QRS/rhythm HistGB + ST/T MLP)
    b5_logits, _ = _cross_fitted_fusion(s_c_b, s_q_b_mlp, labels, folds)
    print(f"B5 (All-classical split):  {_metrics(labels, b5_logits, 'B5_ClassicalSplit')}", flush=True)

    # ===================================================================
    # Bootstrap comparisons
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("PAIRED PATIENT-CLUSTER BOOTSTRAP", flush=True)
    print("=" * 70, flush=True)

    comparisons = [
        ("A3_minus_A0", a3_logits, s_c_a, "Route A: Fusion vs Classical"),
        ("A3_minus_A4", a3_logits, a4_logits, "Route A: VQC fusion vs MLP fusion"),
        ("A3_minus_A5", a3_logits, a5_logits, "Route A: VQC fusion vs shuffled"),
        ("B2_minus_B0", b2_logits, s_c_b, "Route B: Fusion vs QRS-only"),
        ("B2_minus_B3", b2_logits, s_c_a, "Route B: Fusion vs all-feature oracle"),
        ("B2_minus_B4", b2_logits, b4_logits, "Route B: Fusion vs route-swap"),
        ("B2_minus_B5", b2_logits, b5_logits, "Route B: VQC fusion vs MLP fusion"),
    ]

    bootstrap_results = {}
    for name, logits_a, logits_b, description in comparisons:
        result = _paired_patient_bootstrap(
            labels, patient_ids, expit(logits_a), expit(logits_b),
            iterations=2000, seed=seed, comparison=name,
        )
        bootstrap_results[name] = {
            "description": description,
            **{k: v.item() if hasattr(v, "item") else v for k, v in result.items()},
        }
        ci = result.get("delta_auprc", {})
        print(f"  {name}: ΔAUPRC {ci.get('mean', 'N/A'):.4f} "
              f"[{ci.get('ci95_low', 'N/A'):.4f}, {ci.get('ci95_high', 'N/A'):.4f}]  "
              f"({description})", flush=True)

    # ===================================================================
    # Save all artifacts
    # ===================================================================
    print("\nSaving artifacts...", flush=True)

    all_metrics = []
    for name, logits in [
        ("A0_Classical106", s_c_a),
        ("A1_VQC_transformer_q4", s_q_a_vqc),
        ("A2_MLP_transformer_q4", s_q_a_mlp),
        ("A2_Logistic_transformer_q4", s_q_a_logistic),
        ("A2_RBF_transformer_q4", s_q_a_rbf),
        ("A2_Laplacian_transformer_q4", s_q_a_laplacian),
        ("A3_Fusion_C_VQC", a3_logits),
        ("A4_Fusion_C_MLP", a4_logits),
        ("A5_Fusion_Shuffled", a5_logits),
        ("B0_Classical_QRS", s_c_b),
        ("B1_VQC_STT", s_q_b_vqc),
        ("B1_MLP_STT", s_q_b_mlp),
        ("B1_Logistic_STT", s_q_b_logistic),
        ("B1_RBF_STT", s_q_b_rbf),
        ("B1_Laplacian_STT", s_q_b_laplacian),
        ("B2_Fusion_QRS_VQC", b2_logits),
        ("B3_Oracle_All106", s_c_a),
        ("B4_RouteSwap", b4_logits),
        ("B5_ClassicalSplit", b5_logits),
    ]:
        all_metrics.append(_metrics(labels, logits, name))

    pd.DataFrame(all_metrics).to_csv(output_dir / "all_model_metrics.csv", index=False)

    # Save fusion coefficients
    with open(output_dir / "route_a_fusion_coefficients.json", "w") as f:
        json.dump(a3_coefs, f, indent=2, default=_json_default)
    with open(output_dir / "route_b_fusion_coefficients.json", "w") as f:
        json.dump(b2_coefs, f, indent=2, default=_json_default)

    # Save bootstrap results
    with open(output_dir / "bootstrap_comparisons.json", "w") as f:
        json.dump(bootstrap_results, f, indent=2, default=_json_default)

    # Save quantum training audits
    with open(output_dir / "quantum_training_audit.json", "w") as f:
        json.dump(quantum_audits_a, f, indent=2, default=_json_default)

    # Save OOF predictions
    predictions = pd.DataFrame({
        "ecg_id": record_ids,
        "patient_id": patient_ids,
        "fold": folds,
        "label": labels,
        "hard_negative": hard_neg,
        "s_c_a": s_c_a,
        "s_q_a_vqc": s_q_a_vqc,
        "s_q_a_mlp": s_q_a_mlp,
        "s_q_a_logistic": s_q_a_logistic,
        "s_q_a_rbf": s_q_a_rbf,
        "s_q_a_laplacian": s_q_a_laplacian,
        "s_fusion_a3": a3_logits,
        "s_fusion_a4": a4_logits,
        "s_fusion_a5": a5_logits,
        "s_c_b": s_c_b,
        "s_q_b_vqc": s_q_b_vqc,
        "s_q_b_mlp": s_q_b_mlp,
        "s_q_b_logistic": s_q_b_logistic,
        "s_q_b_rbf": s_q_b_rbf,
        "s_q_b_laplacian": s_q_b_laplacian,
        "s_fusion_b2": b2_logits,
        "s_fusion_b4": b4_logits,
        "s_fusion_b5": b5_logits,
    })
    predictions.to_csv(output_dir / "branch_oof_predictions.csv", index=False)

    # Complementarity analysis
    from scipy.stats import spearmanr
    rho_a, _ = spearmanr(s_c_a, s_q_a_vqc)
    rho_b, _ = spearmanr(s_c_b, s_q_b_vqc)
    complementarity = {
        "route_a_classical_vqc_spearman": float(rho_a),
        "route_b_classical_vqc_spearman": float(rho_b),
    }
    with open(output_dir / "complementarity_report.json", "w") as f:
        json.dump(complementarity, f, indent=2, default=_json_default)

    print("\n" + "=" * 70, flush=True)
    print("INDEPENDENT DUAL-ROUTE FUSION SCREEN COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"Artifacts saved to: {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independent dual-route classical/quantum fusion screen"
    )
    parser.add_argument("--representations", type=Path, required=True,
                        help="Directory with outer_fold_N_representations.npz")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--route-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    run_independent_dual_route_screen(
        representations_dir=args.representations,
        metadata_path=args.metadata,
        features_csv=args.features,
        manifest_path=args.manifest,
        route_config_path=args.route_config,
        output_dir=args.output,
        per_class=args.per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        preflight_only=args.preflight_only,
    )


if __name__ == "__main__":
    main()
