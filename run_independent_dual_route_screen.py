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
from aquire_preprocessing.models_quantum import DirectQuantumClassifier
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
        "method": "train-only median + robust scale + supervised PLS + quantile angles",
        "fit_records": int(len(fit_idx)),
        "n_components": int(n_components),
        "q_fit_range": [float(q_fit.min()), float(q_fit.max())],
    }
    return q_fit.astype(np.float32), q_transform.astype(np.float32), audit


def _train_vqc(
    q_train: np.ndarray,
    y_train: np.ndarray,
    q_val: np.ndarray,
    n_qubits: int,
    epochs: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, dict]:
    """Train a DirectQuantumClassifier and return validation logits."""
    import torch

    _seed_torch(seed)
    model = DirectQuantumClassifier(n_qubits=n_qubits, n_layers=2, topology="ring")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    pos_weight = torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
    ).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    x_t = torch.tensor(q_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    indices = np.arange(len(q_train))
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(epochs):
        rng.shuffle(indices)
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad()
            logits = model(x_t[batch])
            loss = criterion(logits, y_t[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        val_logits = model(torch.tensor(q_val, dtype=torch.float32).to(device))
        val_logits = val_logits.cpu().numpy()

    audit = {
        "final_loss": losses[-1] if losses else float("nan"),
        "loss_declined": bool(len(losses) >= 2 and losses[-1] < losses[0]),
        "epochs": epochs,
    }
    return val_logits, audit


def _train_mlp_control(
    q_train: np.ndarray,
    y_train: np.ndarray,
    q_val: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Parameter-count-matched MLP on same q4 coordinates."""
    mlp = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        max_iter=500,
        random_state=seed,
    )
    mlp.fit(q_train, y_train)
    return _safe_logit(mlp.predict_proba(q_val)[:, 1])


def _cross_fitted_fusion(
    s_c: np.ndarray,
    s_q: np.ndarray,
    labels: np.ndarray,
    folds: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    """Cross-fitted one-neuron logistic fusion on OOF branch logits."""
    fusion_logits = np.full(len(labels), np.nan)
    coef_records = []
    unique_folds = sorted(np.unique(folds).astype(int))

    for meta_fold in unique_folds:
        meta_val = folds == meta_fold
        meta_train = ~meta_val

        X_train = np.column_stack([s_c[meta_train], s_q[meta_train]])
        X_val = np.column_stack([s_c[meta_val], s_q[meta_val]])
        y_train = labels[meta_train]

        # Standardize from meta-training only
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0).clip(1e-6)
        X_train = (X_train - mu) / sigma
        X_val = (X_val - mu) / sigma

        lr = LogisticRegression(C=1.0, max_iter=2000, random_state=seed)
        lr.fit(X_train, y_train)
        fusion_logits[meta_val] = lr.decision_function(X_val)
        coef_records.append({
            "meta_fold": int(meta_fold),
            "intercept": float(lr.intercept_[0]),
            "beta_c": float(lr.coef_[0, 0]),
            "beta_q": float(lr.coef_[0, 1]),
        })

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
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_route_config(route_config_path)

    # Load metadata
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")
    features_raw = pd.read_csv(features_csv).set_index("ecg_id")
    manifest = load_feature_manifest(manifest_path, features_raw.columns)
    approved = manifest["approved_features"]
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
    all_features_mat = (
        joined[approved]
        .select_dtypes(include=[np.number])
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    route_b_c_idx = [approved.index(f) for f in route_b_classical if f in approved]
    route_b_q_idx = [approved.index(f) for f in route_b_quantum if f in approved]

    # Verify disjointness
    assert set(route_b_c_idx).isdisjoint(set(route_b_q_idx)), "Route B features overlap!"
    print(f"Route B: {len(route_b_c_idx)} classical, {len(route_b_q_idx)} quantum features", flush=True)

    # Load Transformer representations
    print("Loading Transformer representations...", flush=True)
    h128_oof = np.full((len(joined), 128), np.nan, dtype=np.float32)
    for fold_id in sorted(np.unique(folds)):
        rep_path = representations_dir / f"outer_fold_{fold_id}_representations.npz"
        if not rep_path.exists():
            raise FileNotFoundError(f"Missing representation: {rep_path}")
        rep = np.load(rep_path)
        rep_ids = rep["ecg_ids"]
        rep_h = rep["embeddings"]
        fold_mask = folds == fold_id
        fold_records = record_ids[fold_mask]
        id_to_row = {int(eid): i for i, eid in enumerate(rep_ids)}
        for local_idx, eid in enumerate(fold_records):
            if int(eid) in id_to_row:
                h128_oof[np.where(fold_mask)[0][local_idx]] = rep_h[id_to_row[int(eid)]]

    if not np.isfinite(h128_oof).all():
        n_missing = (~np.isfinite(h128_oof).all(axis=1)).sum()
        print(f"WARNING: {n_missing} records missing h128 representations", flush=True)

    # ===================================================================
    # ROUTE A — representation-disjoint
    # ===================================================================
    print("\n" + "=" * 70, flush=True)
    print("ROUTE A: Clinical(106) → HistGB  ∥  h128 → PLS-q4 → VQC", flush=True)
    print("=" * 70, flush=True)

    # A0: Classical reference (all 106 features)
    s_c_a = np.full(len(labels), np.nan)
    for held_out in sorted(np.unique(folds)):
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
    quantum_audits_a = []

    for held_out in sorted(np.unique(folds)):
        print(f"\n  Route A Fold {held_out}:", flush=True)
        train_mask = folds != held_out
        val_mask = folds == held_out
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        # Subsample for quantum
        sample_idx = _patient_unique_sample(
            labels[train_idx], hard_neg[train_idx], patient_ids[train_idx],
            per_class=per_class, seed=seed + held_out,
        )
        q_train_idx = train_idx[sample_idx]

        # PLS on h128 → q4 (direct label, NOT residual)
        q_train, q_val, rep_audit = _fit_q4_representation(
            h128_oof, labels, q_train_idx, val_idx, n_components=4, seed=seed,
        )
        y_train_q = labels[q_train_idx]

        # VQC
        vqc_logits, vqc_audit = _train_vqc(
            q_train, y_train_q, q_val,
            n_qubits=4, epochs=epochs, batch_size=batch_size,
            seed=seed, device=device,
        )
        s_q_a_vqc[val_idx] = vqc_logits

        # MLP control
        s_q_a_mlp[val_idx] = _train_mlp_control(q_train, y_train_q, q_val, seed)

        # Logistic control
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        lr.fit(q_train, y_train_q)
        s_q_a_logistic[val_idx] = lr.decision_function(q_val)

        quantum_audits_a.append({
            "fold": int(held_out),
            "representation": rep_audit,
            "vqc": vqc_audit,
        })
        print(f"    VQC val AUPRC: {average_precision_score(labels[val_idx], expit(vqc_logits)):.4f}", flush=True)

    print(f"\nA1 (VQC standalone):       {_metrics(labels, s_q_a_vqc, 'A1_VQC')}", flush=True)
    print(f"A2 (MLP control):          {_metrics(labels, s_q_a_mlp, 'A2_MLP')}", flush=True)
    print(f"A2 (Logistic control):     {_metrics(labels, s_q_a_logistic, 'A2_Logistic')}", flush=True)

    # A3: Fusion (classical + VQC)
    a3_logits, a3_coefs = _cross_fitted_fusion(s_c_a, s_q_a_vqc, labels, folds, seed)
    print(f"A3 (Fusion C+VQC):         {_metrics(labels, a3_logits, 'A3_Fusion')}", flush=True)

    # A4: Fusion (classical + MLP) — all-classical control
    a4_logits, a4_coefs = _cross_fitted_fusion(s_c_a, s_q_a_mlp, labels, folds, seed)
    print(f"A4 (Fusion C+MLP):         {_metrics(labels, a4_logits, 'A4_AllClassical')}", flush=True)

    # A5: Quantum-removal control (shuffle sQ)
    rng = np.random.default_rng(seed)
    s_q_a_shuffled = rng.permutation(s_q_a_vqc)
    a5_logits, _ = _cross_fitted_fusion(s_c_a, s_q_a_shuffled, labels, folds, seed)
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
    for held_out in sorted(np.unique(folds)):
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

    for held_out in sorted(np.unique(folds)):
        print(f"\n  Route B Fold {held_out}:", flush=True)
        train_mask = folds != held_out
        val_mask = folds == held_out
        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]

        sample_idx = _patient_unique_sample(
            labels[train_idx], hard_neg[train_idx], patient_ids[train_idx],
            per_class=per_class, seed=seed + held_out,
        )
        q_train_idx = train_idx[sample_idx]

        q_train, q_val, _ = _fit_q4_representation(
            features_b_q, labels, q_train_idx, val_idx, n_components=4, seed=seed,
        )
        y_train_q = labels[q_train_idx]

        vqc_logits, _ = _train_vqc(
            q_train, y_train_q, q_val,
            n_qubits=4, epochs=epochs, batch_size=batch_size,
            seed=seed, device=device,
        )
        s_q_b_vqc[val_idx] = vqc_logits
        s_q_b_mlp[val_idx] = _train_mlp_control(q_train, y_train_q, q_val, seed)

        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        lr.fit(q_train, y_train_q)
        s_q_b_logistic[val_idx] = lr.decision_function(q_val)

        print(f"    VQC val AUPRC: {average_precision_score(labels[val_idx], expit(vqc_logits)):.4f}", flush=True)

    print(f"\nB1 (VQC ST/T standalone):  {_metrics(labels, s_q_b_vqc, 'B1_VQC')}", flush=True)
    print(f"B1 (MLP control):          {_metrics(labels, s_q_b_mlp, 'B1_MLP')}", flush=True)
    print(f"B1 (Logistic control):     {_metrics(labels, s_q_b_logistic, 'B1_Logistic')}", flush=True)

    # B2: Fusion (QRS/rhythm + ST/T VQC)
    b2_logits, b2_coefs = _cross_fitted_fusion(s_c_b, s_q_b_vqc, labels, folds, seed)
    print(f"B2 (Fusion QRS+VQC):       {_metrics(labels, b2_logits, 'B2_Fusion')}", flush=True)

    # B3: Oracle (all 106 HistGB, same as A0)
    print(f"B3 (All-feature oracle):   {_metrics(labels, s_c_a, 'B3_Oracle')}", flush=True)

    # B4: Route-swap (ST/T → classical, QRS/rhythm → VQC)
    s_c_b4 = np.full(len(labels), np.nan)
    s_q_b4 = np.full(len(labels), np.nan)
    for held_out in sorted(np.unique(folds)):
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
        q_train_idx = train_idx[sample_idx]
        q_train, q_val, _ = _fit_q4_representation(
            features_b_c, labels, q_train_idx, val_idx, n_components=4, seed=seed,
        )
        vqc_logits, _ = _train_vqc(
            q_train, labels[q_train_idx], q_val,
            n_qubits=4, epochs=epochs, batch_size=batch_size,
            seed=seed, device=device,
        )
        s_q_b4[val_idx] = vqc_logits

    b4_logits, _ = _cross_fitted_fusion(s_c_b4, s_q_b4, labels, folds, seed)
    print(f"B4 (Route-swap fusion):    {_metrics(labels, b4_logits, 'B4_Swap')}", flush=True)

    # B5: All-classical split (QRS/rhythm HistGB + ST/T MLP)
    b5_logits, _ = _cross_fitted_fusion(s_c_b, s_q_b_mlp, labels, folds, seed)
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
            labels, expit(logits_a), expit(logits_b), patient_ids,
            n_iterations=2000, seed=seed,
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
        ("A1_VQC_h128", s_q_a_vqc),
        ("A2_MLP_h128", s_q_a_mlp),
        ("A2_Logistic_h128", s_q_a_logistic),
        ("A3_Fusion_C_VQC", a3_logits),
        ("A4_Fusion_C_MLP", a4_logits),
        ("A5_Fusion_Shuffled", a5_logits),
        ("B0_Classical_QRS", s_c_b),
        ("B1_VQC_STT", s_q_b_vqc),
        ("B1_MLP_STT", s_q_b_mlp),
        ("B1_Logistic_STT", s_q_b_logistic),
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
        "s_fusion_a3": a3_logits,
        "s_fusion_a4": a4_logits,
        "s_fusion_a5": a5_logits,
        "s_c_b": s_c_b,
        "s_q_b_vqc": s_q_b_vqc,
        "s_q_b_mlp": s_q_b_mlp,
        "s_q_b_logistic": s_q_b_logistic,
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
    )


if __name__ == "__main__":
    main()
