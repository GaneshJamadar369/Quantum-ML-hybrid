"""Phase 6Q-C1: patient-safe projected IQP kernel screening.

This is a development experiment on folds 1-8.  It learns a supervised PLS
representation inside each outer training fold, selects a shallow projected IQP
map by centered kernel-target alignment on training records only, and evaluates
it against matched classical kernels using the same representation and sample
budget.  Folds 9 and 10 are rejected by the access guard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.svm import SVC

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_quantum import ProjectedIQPFeatureMap
from run_quantum_baselines import _paired_patient_bootstrap


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _fold_local_pls(
    features: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    x_train = scaler.fit_transform(imputer.fit_transform(features[train_idx]))
    x_val = scaler.transform(imputer.transform(features[val_idx]))
    pls = PLSRegression(n_components=n_components, scale=False, max_iter=1000)
    z_train = pls.fit_transform(x_train, labels[train_idx])[0]
    z_val = pls.transform(x_val)
    quantiles = QuantileTransformer(
        n_quantiles=min(256, len(z_train)),
        output_distribution="uniform",
        random_state=seed,
    )
    z_train = (2.0 * quantiles.fit_transform(z_train) - 1.0) * np.pi
    z_val = (2.0 * quantiles.transform(z_val) - 1.0) * np.pi
    return z_train.astype(np.float64), z_val.astype(np.float64), {
        "method": "training-only median + robust scale + supervised PLS + quantile angles",
        "n_components": int(n_components),
        "x_rotation_squared_norm": float(np.sum(pls.x_rotations_ ** 2)),
    }


def _stratified_training_sample(
    labels: np.ndarray,
    hard_negative: np.ndarray,
    per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    hard = np.flatnonzero((labels == 0) & hard_negative)
    other = np.flatnonzero((labels == 0) & ~hard_negative)
    n_pos = min(per_class, len(positive))
    n_hard = min(per_class // 2, len(hard))
    n_other = min(per_class - n_hard, len(other))
    if n_hard + n_other < n_pos:
        remaining = np.setdiff1d(np.flatnonzero(labels == 0), np.r_[hard[:0], other[:0]])
        n_pos = min(n_pos, len(remaining))
        negative = rng.choice(remaining, n_pos, replace=False)
    else:
        negative = np.r_[
            rng.choice(hard, n_hard, replace=False),
            rng.choice(other, n_other, replace=False),
        ]
    selected = np.r_[rng.choice(positive, len(negative), replace=False), negative]
    return rng.permutation(selected)


def _centered_alignment(kernel: np.ndarray, labels: np.ndarray) -> float:
    n = len(kernel)
    center = np.eye(n) - np.ones((n, n)) / n
    k_centered = center @ kernel @ center
    signed = 2.0 * np.asarray(labels, dtype=float) - 1.0
    target = np.outer(signed, signed)
    target = center @ target @ center
    denominator = np.linalg.norm(k_centered) * np.linalg.norm(target)
    return float(np.sum(k_centered * target) / denominator) if denominator else 0.0


def _kernel_health(kernel: np.ndarray) -> dict:
    matrix = (np.asarray(kernel) + np.asarray(kernel).T) / 2.0
    eig = np.linalg.eigvalsh(matrix)
    positive = np.clip(eig, 0.0, None)
    probability = positive / positive.sum() if positive.sum() else positive
    effective_rank = float(np.exp(-np.sum(probability[probability > 0] * np.log(probability[probability > 0]))))
    offdiag = matrix[~np.eye(len(matrix), dtype=bool)]
    return {
        "effective_rank": effective_rank,
        "minimum_eigenvalue": float(eig.min()),
        "condition_ridge_1e-6": float(np.linalg.cond(matrix + 1e-6 * np.eye(len(matrix)))),
        "offdiagonal_mean": float(offdiag.mean()),
        "offdiagonal_std": float(offdiag.std()),
    }


def _candidate_search(
    representations: dict[int, np.ndarray],
    labels: np.ndarray,
    seed: int,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    n = min(128, len(positive), len(negative))
    search_idx = np.r_[rng.choice(positive, n, replace=False), rng.choice(negative, n, replace=False)]
    search_y = labels[search_idx]

    initial = []
    for qubits in (4, 6, 8):
        for scale in (0.25, 0.5, 1.0):
            initial.append({
                "n_qubits": qubits,
                "n_layers": 1,
                "feature_scale": scale,
                "interaction_scale": 0.5,
                "topology": "ring",
                "mixing_seed": seed + qubits,
            })

    def evaluate(config: dict) -> dict:
        fmap = ProjectedIQPFeatureMap(**config)
        q = fmap.transform(representations[config["n_qubits"]][search_idx])
        base_kernel, base_gamma = fmap.rbf_kernel(q)
        best = None
        for factor in (0.25, 1.0, 4.0):
            kernel, gamma = fmap.rbf_kernel(q, gamma=base_gamma * factor)
            health = _kernel_health(kernel)
            alignment = _centered_alignment(kernel, search_y)
            result = {
                **config,
                "gamma": gamma,
                "gamma_factor": factor,
                "alignment": alignment,
                **health,
            }
            if health["offdiagonal_std"] < 1e-4 or health["effective_rank"] < 2.0:
                result["selection_score"] = -np.inf
            else:
                result["selection_score"] = alignment
            if best is None or result["selection_score"] > best["selection_score"]:
                best = result
        return best

    evaluated = [evaluate(config) for config in initial]
    top = sorted(evaluated, key=lambda row: row["selection_score"], reverse=True)[:3]
    expanded = []
    for parent in top:
        base = {key: parent[key] for key in (
            "n_qubits", "feature_scale", "interaction_scale", "mixing_seed"
        )}
        expanded.extend([
            {**base, "n_layers": 2, "topology": "ring"},
            {**base, "n_layers": 1, "topology": "ladder", "interaction_scale": 0.25},
        ])
    evaluated.extend(evaluate(config) for config in expanded)
    finite = [row for row in evaluated if np.isfinite(row["selection_score"])]
    if not finite:
        raise RuntimeError("All projected quantum kernels failed the health gate")
    winner = max(finite, key=lambda row: row["selection_score"])
    return winner, evaluated


def _precomputed_svc(
    train_kernel: np.ndarray,
    train_labels: np.ndarray,
    val_kernel: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    best_c, best_score = None, -np.inf
    for c in (0.1, 1.0, 10.0):
        scores = []
        for fit_idx, score_idx in cv.split(train_kernel, train_labels):
            model = SVC(kernel="precomputed", C=c, class_weight="balanced")
            model.fit(train_kernel[np.ix_(fit_idx, fit_idx)], train_labels[fit_idx])
            decision = model.decision_function(train_kernel[np.ix_(score_idx, fit_idx)])
            scores.append(average_precision_score(train_labels[score_idx], decision))
        score = float(np.mean(scores))
        if score > best_score:
            best_c, best_score = c, score
    model = SVC(kernel="precomputed", C=best_c, class_weight="balanced")
    model.fit(train_kernel, train_labels)
    return model.decision_function(val_kernel), float(best_c)


def _sqdist(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    from sklearn.metrics import pairwise_distances
    return pairwise_distances(left, right, metric="sqeuclidean")


def _classical_kernel_predictions(
    name: str,
    z_train: np.ndarray,
    y_train: np.ndarray,
    z_val: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    train_distance = _sqdist(z_train, z_train)
    val_distance = _sqdist(z_val, z_train)
    positive = train_distance[np.triu_indices_from(train_distance, k=1)]
    positive = positive[positive > 1e-12]
    base_gamma = 1.0 / float(np.median(positive)) if len(positive) else 1.0
    candidates = []
    if name in {"rbf", "laplacian"}:
        for factor in (0.25, 1.0, 4.0):
            gamma = base_gamma * factor
            if name == "rbf":
                candidates.append((gamma, np.exp(-gamma * train_distance), np.exp(-gamma * val_distance)))
            else:
                train_l1 = np.abs(z_train[:, None, :] - z_train[None, :, :]).sum(axis=2)
                val_l1 = np.abs(z_val[:, None, :] - z_train[None, :, :]).sum(axis=2)
                candidates.append((gamma, np.exp(-np.sqrt(gamma) * train_l1), np.exp(-np.sqrt(gamma) * val_l1)))
    elif name == "product_cosine":
        train_delta = z_train[:, None, :] - z_train[None, :, :]
        val_delta = z_val[:, None, :] - z_train[None, :, :]
        candidates.append((None, np.prod(np.cos(train_delta / 2.0) ** 2, axis=2), np.prod(np.cos(val_delta / 2.0) ** 2, axis=2)))
    else:
        raise ValueError(name)

    best = None
    for gamma, train_kernel, val_kernel in candidates:
        decision, c = _precomputed_svc(train_kernel, y_train, val_kernel, seed)
        # Select gamma by the same inner-CV routine, recomputed without touching
        # outer labels. The returned outer decisions are stored only for the
        # selected candidate.
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        inner = []
        for fit_idx, score_idx in cv.split(train_kernel, y_train):
            model = SVC(kernel="precomputed", C=c, class_weight="balanced")
            model.fit(train_kernel[np.ix_(fit_idx, fit_idx)], y_train[fit_idx])
            score = model.decision_function(train_kernel[np.ix_(score_idx, fit_idx)])
            inner.append(average_precision_score(y_train[score_idx], score))
        row = {"decision": decision, "C": c, "gamma": gamma, "inner_auprc": float(np.mean(inner))}
        if best is None or row["inner_auprc"] > best["inner_auprc"]:
            best = row
    return best.pop("decision"), best


def run_search(
    metadata_path: Path,
    features_path: Path,
    manifest_path: Path,
    output_dir: Path,
    per_class: int = 500,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")
    features = pd.read_csv(features_path).set_index("ecg_id")
    manifest = load_feature_manifest(manifest_path, features.columns)
    approved = manifest["approved_features"]
    joined = metadata.join(features[approved], how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")]
    folds = joined.strat_fold.to_numpy(dtype=int)
    labels = joined.mi_label.to_numpy(dtype=int)
    patient_ids = joined.patient_id.to_numpy()
    record_ids = joined.index.to_numpy()
    hard_negative = joined.hard_negative.to_numpy(dtype=bool)
    guard_fold_access(folds, purpose="tuning")
    x = joined[approved].select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).to_numpy(float)

    model_names = ["projected_iqp", "rbf", "laplacian", "product_cosine"]
    oof = {name: np.full(len(joined), np.nan) for name in model_names}
    audits = []
    for held_out in sorted(np.unique(folds)):
        checkpoint = output_dir / f"fold_{held_out}.npz"
        if checkpoint.exists():
            saved = np.load(checkpoint, allow_pickle=False)
            val_idx = saved["val_idx"].astype(int)
            for name in model_names:
                oof[name][val_idx] = saved[name]
            audits.append(json.loads(str(saved["audit"].item())))
            print(f"Fold {held_out}: resumed", flush=True)
            continue

        train_idx = np.flatnonzero(folds != held_out)
        val_idx = np.flatnonzero(folds == held_out)
        relative_sample = _stratified_training_sample(
            labels[train_idx], hard_negative[train_idx], per_class, seed + held_out
        )
        selected_global = train_idx[relative_sample]
        representation_cache = {}
        audits_by_n = {}
        for n_qubits in (4, 6, 8):
            z_train_all, z_val, audit = _fold_local_pls(
                x, labels, train_idx, val_idx, n_qubits, seed + held_out
            )
            representation_cache[n_qubits] = (z_train_all[relative_sample], z_val)
            audits_by_n[n_qubits] = audit
        sample_representations = {
            n: pair[0] for n, pair in representation_cache.items()
        }
        winner, candidates = _candidate_search(
            sample_representations,
            labels[selected_global],
            seed + held_out,
        )
        n_qubits = int(winner["n_qubits"])
        z_train, z_val = representation_cache[n_qubits]
        y_train = labels[selected_global]
        config = {key: winner[key] for key in (
            "n_qubits", "n_layers", "feature_scale", "interaction_scale", "topology", "mixing_seed"
        )}
        fmap = ProjectedIQPFeatureMap(**config)
        q_train = fmap.transform(z_train)
        q_val = fmap.transform(z_val)
        train_kernel, _ = fmap.rbf_kernel(q_train, gamma=winner["gamma"])
        val_kernel, _ = fmap.rbf_kernel(q_val, q_train, gamma=winner["gamma"])
        decision, c = _precomputed_svc(
            train_kernel, y_train, val_kernel, seed + held_out
        )
        oof["projected_iqp"][val_idx] = expit(decision)
        controls = {}
        for name in ("rbf", "laplacian", "product_cosine"):
            decision, control_audit = _classical_kernel_predictions(
                name, z_train, y_train, z_val, seed + held_out
            )
            oof[name][val_idx] = expit(decision)
            controls[name] = control_audit
        audit = {
            "held_out_fold": int(held_out),
            "training_records": int(len(selected_global)),
            "representation": audits_by_n[n_qubits],
            "winner": {**winner, "C": c},
            "candidate_count": len(candidates),
            "controls": controls,
        }
        audits.append(audit)
        temporary = checkpoint.with_suffix(".npz.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                val_idx=val_idx,
                audit=json.dumps(audit, sort_keys=True, default=_json_default),
                **{name: oof[name][val_idx] for name in model_names},
            )
        temporary.replace(checkpoint)
        print(
            f"Fold {held_out}: projected-IQP config n={n_qubits}, "
            f"alignment={winner['alignment']:.4f}",
            flush=True,
        )

    rows = []
    prediction_rows = []
    for name, probability in oof.items():
        if not np.isfinite(probability).all():
            raise RuntimeError(f"Incomplete OOF predictions for {name}")
        rows.append({
            "model": name,
            "auprc": float(average_precision_score(labels, probability)),
            "auroc": float(roc_auc_score(labels, probability)),
            "probability_status": "uncalibrated_monotonic_sigmoid_of_margin",
        })
        for rec, patient, fold, y, hard, p in zip(
            record_ids, patient_ids, folds, labels, hard_negative, probability
        ):
            prediction_rows.append({
                "ecg_id": int(rec),
                "patient_id": int(patient),
                "strat_fold": int(fold),
                "y_true": int(y),
                "hard_negative": bool(hard),
                "model": name,
                "score": float(p),
            })
    metrics = pd.DataFrame(rows).sort_values("auprc", ascending=False)
    metrics.to_csv(output_dir / "projected_kernel_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(output_dir / "projected_kernel_oof_predictions.csv", index=False)
    (output_dir / "projected_kernel_fold_audits.json").write_text(
        json.dumps(audits, indent=2, default=_json_default)
    )
    for control in ("rbf", "laplacian", "product_cosine"):
        comparison = _paired_patient_bootstrap(
            labels,
            patient_ids,
            oof["projected_iqp"],
            oof[control],
            iterations=2000,
            seed=seed,
            comparison=f"projected_iqp_minus_{control}",
        )
        (output_dir / f"paired_projected_iqp_vs_{control}.json").write_text(
            json.dumps(comparison, indent=2, default=_json_default)
        )
    print(metrics.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_search(
        args.metadata,
        args.features,
        args.manifest,
        args.output,
        per_class=args.per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
