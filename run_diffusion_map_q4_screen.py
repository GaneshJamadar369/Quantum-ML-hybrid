"""Fold-safe landmark Diffusion-Map q4 screen for the retained VQC.

Only the Transformer h128 -> q4 compressor changes.  The existing four-qubit
two-layer ring VQC, JS objective, patient-unique sample, train-reference score
alignment, calibration, matched q4 controls, and clinical fusion remain fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.diffusion_map import LandmarkDiffusionMap
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from run_divergence_distillation_screen import (
    _metrics,
    _oof_calibrated_teacher,
    _safe_logit,
    _validate_representations,
)
from run_independent_dual_route_screen import _cross_fitted_fusion
from run_nested_q4_optimization import _mlp, _operating_point
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _json_default, _patient_unique_sample
from run_train_reference_score_alignment import (
    _cross_fitted_calibration,
    _fit_restart,
)


FROZEN_CLASSICAL_CEILING_AUPRC = 0.8380154031265051


def _effective_rank(values: np.ndarray) -> float:
    singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
    energy = np.square(singular)
    probability = energy / max(float(energy.sum()), 1e-12)
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


def _geometry_audit(h: np.ndarray, q: np.ndarray, seed: int, limit: int = 1000) -> dict:
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(h), size=min(limit, len(h)), replace=False)
    hs, qs = np.asarray(h[selected], float), np.asarray(q[selected], float)
    # Scale each h dimension robustly for a distance diagnostic only.
    median = np.median(hs, axis=0)
    scale = np.quantile(hs, 0.75, axis=0) - np.quantile(hs, 0.25, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    hs = (hs - median) / scale
    h_distance, q_distance = pdist(hs), pdist(qs)
    distance_rho = float(spearmanr(h_distance, q_distance).statistic)
    neighbors = min(16, len(hs))
    if neighbors < 2:
        overlap = float("nan")
    else:
        h_index = NearestNeighbors(n_neighbors=neighbors).fit(hs).kneighbors(return_distance=False)
        q_index = NearestNeighbors(n_neighbors=neighbors).fit(qs).kneighbors(return_distance=False)
        overlap = float(
            np.mean(
                [
                    len(set(a[1:]) & set(b[1:])) / max(len(a) - 1, 1)
                    for a, b in zip(h_index, q_index)
                ]
            )
        )
    return {
        "records": int(len(selected)),
        "h128_effective_rank": _effective_rank(hs),
        "q4_effective_rank": _effective_rank(qs),
        "pairwise_distance_spearman": distance_rho,
        "knn15_overlap": overlap,
    }


def _load_pls_reference(path: Path, joined: pd.DataFrame) -> pd.DataFrame:
    reference = pd.read_csv(path).set_index("ecg_id")
    required = {
        "patient_id",
        "strat_fold",
        "y_true",
        "vqc_train_cdf",
        "q4_mlp_ensemble",
        "fusion_vqc_train_cdf",
        "fusion_q4_mlp",
    }
    missing = required - set(reference.columns)
    if missing:
        raise KeyError(f"PLS reference missing columns: {sorted(missing)}")
    if set(reference.index.astype(int)) != set(joined.index.astype(int)):
        raise ValueError("PLS reference record IDs disagree with the development cohort")
    reference = reference.loc[joined.index]
    for column, expected in (
        ("patient_id", joined.patient_id.to_numpy(int)),
        ("strat_fold", joined.strat_fold.to_numpy(int)),
        ("y_true", joined.mi_label.to_numpy(int)),
    ):
        if not np.array_equal(reference[column].to_numpy(int), expected):
            raise ValueError(f"PLS reference {column} is misaligned")
    return reference


def _run(joined, clinical, paths, reference, args, device):
    labels = joined.mi_label.to_numpy(int)
    folds = joined.strat_fold.to_numpy(int)
    patients = joined.patient_id.to_numpy(int)
    hard = joined.hard_negative.to_numpy(bool)
    record_ids = joined.index.to_numpy(int)
    row_for_id = {int(value): row for row, value in enumerate(record_ids)}
    names = (
        "vqc_raw",
        "vqc_train_cdf",
        "vqc_no_entanglement_cdf",
        "q4_logistic",
        "q4_mlp_ensemble",
        "clinical_teacher",
    )
    scores = {name: np.full(len(joined), np.nan) for name in names}
    q4_oof = np.full((len(joined), 4), np.nan, dtype=np.float32)
    observables = np.full((len(joined), 8), np.nan, dtype=np.float32)
    fold_audits = []

    for outer, path in sorted(paths.items()):
        print(f"diffusion-map outer={outer}", flush=True)
        with np.load(path, allow_pickle=False) as data:
            train_ids = data["train_record_ids"].astype(int)
            val_ids = data["val_record_ids"].astype(int)
            train_h = data["train_embeddings"].astype(np.float32)
            val_h = data["val_embeddings"].astype(np.float32)
            train_y = data["train_labels"].astype(int)
            train_patients = data["train_patient_ids"].astype(int)
        train_rows = np.asarray([row_for_id[int(value)] for value in train_ids])
        val_rows = np.asarray([row_for_id[int(value)] for value in val_ids])
        train_folds = folds[train_rows]
        mapper = LandmarkDiffusionMap(
            n_components=4,
            n_landmarks=args.landmarks,
            graph_neighbors=args.graph_neighbors,
            bandwidth_neighbors=args.bandwidth_neighbors,
            alpha=args.alpha,
            random_state=args.seed + outer,
        )
        train_q = mapper.fit_transform(train_h, train_patients)
        val_q = mapper.transform(val_h)
        q4_oof[val_rows] = val_q
        teacher_train, teacher_val, teacher_audit = _oof_calibrated_teacher(
            clinical[train_rows], train_y, train_folds, clinical[val_rows], args.seed + outer * 10
        )
        scores["clinical_teacher"][val_rows] = teacher_val
        sample = _patient_unique_sample(
            train_y,
            hard[train_rows],
            train_patients,
            per_class=args.per_class,
            seed=args.seed + outer,
        )
        runs = []
        for restart in range(args.restarts):
            runs.append(
                _fit_restart(
                    train_q[sample],
                    train_y[sample],
                    teacher_train[sample],
                    train_q,
                    val_q,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=args.seed + outer * 1000 + restart * 100_000,
                    device=device,
                )
            )
        scores["vqc_raw"][val_rows] = np.mean([run["raw_logit"] for run in runs], axis=0)
        scores["vqc_train_cdf"][val_rows] = np.mean([run["cdf_score"] for run in runs], axis=0)
        observables[val_rows] = np.mean([run["observables"] for run in runs], axis=0)
        no_ent = _fit_restart(
            train_q[sample],
            train_y[sample],
            teacher_train[sample],
            train_q,
            val_q,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed + outer * 1000,
            device=device,
            entanglement=False,
        )
        scores["vqc_no_entanglement_cdf"][val_rows] = no_ent["cdf_score"]
        scores["q4_logistic"][val_rows] = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000
        ).fit(train_q[sample], train_y[sample]).decision_function(val_q)
        mlp_logits = []
        for restart in range(args.restarts):
            probability = (
                _mlp((1e-3, 1e-3), args.seed + outer * 1000 + restart)
                .fit(train_q[sample], train_y[sample])
                .predict_proba(val_q)[:, 1]
            )
            mlp_logits.append(_safe_logit(probability))
        scores["q4_mlp_ensemble"][val_rows] = np.mean(mlp_logits, axis=0)
        fold_audits.append(
            {
                "outer_fold": int(outer),
                "diffusion_map": mapper.fit_audit_,
                "geometry": _geometry_audit(val_h, val_q, args.seed + outer),
                "teacher": teacher_audit,
                "sample_records": int(len(sample)),
                "sample_patients": int(len(np.unique(train_patients[sample]))),
                "restarts": [run["audit"] for run in runs],
                "no_entanglement": no_ent["audit"],
            }
        )

    calibrated, calibration_audits = {}, {}
    for name in (
        "vqc_raw",
        "vqc_train_cdf",
        "vqc_no_entanglement_cdf",
        "q4_logistic",
        "q4_mlp_ensemble",
    ):
        if not np.isfinite(scores[name]).all():
            raise RuntimeError(f"Incomplete score: {name}")
        calibrated[name], calibration_audits[name] = _cross_fitted_calibration(
            scores[name], labels, folds
        )
    calibrated["clinical_teacher"] = scores["clinical_teacher"]
    fusion_vqc, fusion_vqc_audit = _cross_fitted_fusion(
        _safe_logit(scores["clinical_teacher"]),
        _safe_logit(calibrated["vqc_train_cdf"]),
        labels,
        folds,
    )
    fusion_mlp, fusion_mlp_audit = _cross_fitted_fusion(
        _safe_logit(scores["clinical_teacher"]),
        _safe_logit(calibrated["q4_mlp_ensemble"]),
        labels,
        folds,
    )
    calibrated["fusion_vqc_train_cdf"] = expit(fusion_vqc)
    calibrated["fusion_q4_mlp"] = expit(fusion_mlp)
    for column in (
        "vqc_train_cdf",
        "q4_mlp_ensemble",
        "fusion_vqc_train_cdf",
        "fusion_q4_mlp",
    ):
        calibrated[f"pls_reference_{column}"] = reference[column].to_numpy(float)

    metrics = {name: _metrics(labels, probability) for name, probability in calibrated.items()}
    operating = {
        name: _operating_point(labels, probability, hard)
        for name, probability in calibrated.items()
    }
    comparisons = {}
    pairs = (
        ("vqc_train_cdf", "q4_logistic"),
        ("vqc_train_cdf", "q4_mlp_ensemble"),
        ("vqc_train_cdf", "vqc_no_entanglement_cdf"),
        ("vqc_train_cdf", "pls_reference_vqc_train_cdf"),
        ("fusion_vqc_train_cdf", "fusion_q4_mlp"),
        ("fusion_vqc_train_cdf", "pls_reference_fusion_vqc_train_cdf"),
        ("fusion_vqc_train_cdf", "pls_reference_fusion_q4_mlp"),
    )
    for left, right in pairs:
        key = f"{left}_minus_{right}"
        comparisons[key] = _paired_patient_bootstrap(
            labels,
            patients,
            calibrated[left],
            calibrated[right],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
            comparison=key,
        )

    frame = pd.DataFrame(
        {
            "ecg_id": record_ids,
            "patient_id": patients,
            "strat_fold": folds,
            "y_true": labels,
            "hard_negative": hard,
            **{f"raw_score_{name}": value for name, value in scores.items()},
            **calibrated,
            **{f"q4_{index}": q4_oof[:, index] for index in range(4)},
            **{f"quantum_observable_{index}": observables[:, index] for index in range(8)},
        }
    )
    frame.to_csv(args.output / "oof_predictions_and_observables.csv", index=False)
    for filename, value in (
        ("metrics.json", metrics),
        ("operating_points.json", operating),
        ("bootstrap.json", comparisons),
        ("fold_audits.json", fold_audits),
        ("calibration_audits.json", calibration_audits),
        ("fusion_audits.json", {"vqc": fusion_vqc_audit, "mlp": fusion_mlp_audit}),
    ):
        (args.output / filename).write_text(json.dumps(value, indent=2, default=_json_default))

    representation_comparison = comparisons[
        "vqc_train_cdf_minus_pls_reference_vqc_train_cdf"
    ]["delta_auprc"]
    matched_name = max(
        ("q4_logistic", "q4_mlp_ensemble"), key=lambda name: metrics[name]["auprc"]
    )
    matched_comparison = comparisons[f"vqc_train_cdf_minus_{matched_name}"]["delta_auprc"]
    verdict = {
        "diffusion_vqc_auprc": metrics["vqc_train_cdf"]["auprc"],
        "pls_vqc_auprc": metrics["pls_reference_vqc_train_cdf"]["auprc"],
        "best_matched_control": matched_name,
        "best_matched_control_auprc": metrics[matched_name]["auprc"],
        "diffusion_quantum_fusion_auprc": metrics["fusion_vqc_train_cdf"]["auprc"],
        "frozen_classical_ceiling_auprc": FROZEN_CLASSICAL_CEILING_AUPRC,
        "representation_gate": bool(
            representation_comparison["mean"] >= 0.003
            and representation_comparison["ci95_low"] > 0.0
        ),
        "matched_quantum_gate": bool(
            matched_comparison["mean"] >= 0.003 and matched_comparison["ci95_low"] > 0.0
        ),
        "system_gate": bool(
            metrics["fusion_vqc_train_cdf"]["auprc"] > FROZEN_CLASSICAL_CEILING_AUPRC
        ),
        "advance_to_confirmation": False,
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    verdict["advance_to_confirmation"] = bool(
        verdict["representation_gate"]
        and verdict["matched_quantum_gate"]
        and verdict["system_gate"]
    )
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)


def run(args):
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    feature_frame = pd.read_csv(args.features).set_index("ecg_id")
    approved = load_feature_manifest(args.manifest, feature_frame.columns)["approved_features"]
    joined = metadata.join(feature_frame[approved], how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")].copy()
    guard_fold_access(joined.strat_fold.to_numpy(int), purpose="tuning")
    paths = _validate_representations(args.representations, joined)
    reference = _load_pls_reference(args.pls_reference, joined)
    preflight = {
        "records": int(len(joined)),
        "patients": int(joined.patient_id.nunique()),
        "folds": sorted(joined.strat_fold.unique().astype(int).tolist()),
        "representation": "outer-train-only landmark diffusion map",
        "landmarks": int(args.landmarks),
        "graph_neighbors": int(args.graph_neighbors),
        "bandwidth_neighbors": int(args.bandwidth_neighbors),
        "alpha": float(args.alpha),
        "seed": int(args.seed),
        "vqc_restarts": int(args.restarts),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2))
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _run(joined, joined[approved].to_numpy(np.float32), paths, reference, args, device)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pls-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--landmarks", type=int, default=1024)
    parser.add_argument("--graph-neighbors", type=int, default=32)
    parser.add_argument("--bandwidth-neighbors", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

