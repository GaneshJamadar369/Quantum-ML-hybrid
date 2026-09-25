"""Stabilized q4 VQC with deployable train-reference score alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from run_advanced_quantum_fusion_screen import _fit_q4, _seed
from run_divergence_distillation_screen import (
    _metrics,
    _oof_calibrated_teacher,
    _safe_logit,
    _validate_representations,
)
from run_independent_dual_route_screen import _cross_fitted_fusion
from run_nested_q4_optimization import Candidate, _combined_loss, _mlp, _model, _operating_point
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _json_default, _patient_unique_sample


FIXED_CANDIDATE = Candidate("narrow_js", 5e-3, 0.5, 0.35, 0.0)


def _train_reference_scores(reference_logits: np.ndarray, validation_logits: np.ndarray):
    reference = np.asarray(reference_logits, dtype=float)
    validation = np.asarray(validation_logits, dtype=float)
    mean, scale = float(reference.mean()), float(reference.std())
    scale = max(scale, 1e-6)
    z_score = (validation - mean) / scale
    ordered = np.sort(reference)
    cdf_score = (np.searchsorted(ordered, validation, side="right") + 0.5) / (len(ordered) + 1.0)
    return z_score, cdf_score, {"reference_mean": mean, "reference_std": scale}


def _cross_fitted_calibration(score: np.ndarray, labels: np.ndarray, folds: np.ndarray):
    probability = np.full(len(labels), np.nan)
    audits = []
    for held_out in sorted(np.unique(folds).astype(int)):
        validation = folds == held_out
        model = LogisticRegression(C=1.0, max_iter=2000).fit(
            score[~validation].reshape(-1, 1), labels[~validation]
        )
        probability[validation] = model.predict_proba(score[validation].reshape(-1, 1))[:, 1]
        audits.append(
            {
                "held_out_fold": held_out,
                "coefficient": float(model.coef_[0, 0]),
                "intercept": float(model.intercept_[0]),
            }
        )
    if not np.isfinite(probability).all():
        raise RuntimeError("Incomplete cross-fitted calibration")
    return probability, audits


def _fit_restart(
    sample_x,
    sample_y,
    sample_teacher,
    reference_x,
    validation_x,
    *,
    epochs,
    batch_size,
    seed,
    device,
    entanglement=True,
):
    import torch

    _seed(seed)
    model = _model(FIXED_CANDIDATE, device, entanglement=entanglement)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FIXED_CANDIDATE.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=2e-4
    )
    x = torch.from_numpy(sample_x.astype(np.float32)).to(device)
    y = torch.from_numpy(sample_y.astype(np.float32)).to(device)
    teacher = torch.from_numpy(sample_teacher.astype(np.float32)).to(device)
    order, rng = np.arange(len(sample_x)), np.random.default_rng(seed)
    loss_history, gradient_history = [], []
    for _ in range(epochs):
        rng.shuffle(order)
        running, seen, gradients = 0.0, 0, []
        model.train()
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch])
            loss = _combined_loss(logits, y[batch], teacher[batch], FIXED_CANDIDATE)
            loss.backward()
            gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
            optimizer.step()
            running += float(loss.detach()) * len(batch)
            seen += len(batch)
        scheduler.step()
        loss_history.append(running / max(seen, 1))
        gradient_history.append(float(np.median(gradients)))
    model.eval()
    with torch.inference_mode():
        reference = model(torch.from_numpy(reference_x.astype(np.float32)).to(device)).cpu().numpy()
        validation_tensor = torch.from_numpy(validation_x.astype(np.float32)).to(device)
        validation = model(validation_tensor).cpu().numpy()
        observables = model.quantum_observables(validation_tensor).cpu().numpy()
    z_score, cdf_score, reference_audit = _train_reference_scores(reference, validation)
    parameters = {
        "feature_scales": model.feature_scales.detach().cpu().numpy().tolist(),
        "rotations": model.rotations.detach().cpu().numpy().tolist(),
        "interactions": model.interactions.detach().cpu().numpy().tolist(),
        "readout_weight": model.readout.weight.detach().cpu().numpy().tolist(),
        "readout_bias": model.readout.bias.detach().cpu().numpy().tolist(),
    }
    return {
        "raw_logit": validation,
        "z_score": z_score,
        "cdf_score": cdf_score,
        "observables": observables,
        "audit": {
            **reference_audit,
            "initial_loss": float(loss_history[0]),
            "final_loss": float(loss_history[-1]),
            "gradient_min": float(np.min(gradient_history)),
            "gradient_max": float(np.max(gradient_history)),
            "loss_history": loss_history,
            "gradient_history": gradient_history,
            "parameters": parameters,
            "entanglement": bool(entanglement),
        },
    }


def _run_seed(seed, joined, clinical, paths, args, device):
    labels = joined.mi_label.to_numpy(int)
    folds = joined.strat_fold.to_numpy(int)
    patients = joined.patient_id.to_numpy(int)
    hard = joined.hard_negative.to_numpy(bool)
    record_ids = joined.index.to_numpy(int)
    row_for_id = {int(value): row for row, value in enumerate(record_ids)}
    raw = np.full(len(joined), np.nan)
    z_score = np.full(len(joined), np.nan)
    cdf_score = np.full(len(joined), np.nan)
    no_ent_raw = np.full(len(joined), np.nan)
    no_ent_cdf = np.full(len(joined), np.nan)
    q4_logistic = np.full(len(joined), np.nan)
    q4_mlp = np.full(len(joined), np.nan)
    clinical_teacher = np.full(len(joined), np.nan)
    q4_oof = np.full((len(joined), 4), np.nan, dtype=np.float32)
    observable_oof = np.full((len(joined), 8), np.nan, dtype=np.float32)
    restart_columns = {
        restart: np.full(len(joined), np.nan) for restart in range(args.restarts)
    }
    fold_audits = []
    for outer in sorted(paths):
        print(f"seed={seed} outer={outer}", flush=True)
        with np.load(paths[outer], allow_pickle=False) as data:
            train_ids = data["train_record_ids"].astype(int)
            val_ids = data["val_record_ids"].astype(int)
            train_h = data["train_embeddings"].astype(np.float32)
            val_h = data["val_embeddings"].astype(np.float32)
            train_y = data["train_labels"].astype(int)
            train_patients = data["train_patient_ids"].astype(int)
        train_rows = np.asarray([row_for_id[int(value)] for value in train_ids])
        val_rows = np.asarray([row_for_id[int(value)] for value in val_ids])
        train_folds = folds[train_rows]
        train_q, val_q, q_audit = _fit_q4(train_h, train_y, val_h, seed + outer)
        q4_oof[val_rows] = val_q
        teacher_train, teacher_val, teacher_audit = _oof_calibrated_teacher(
            clinical[train_rows], train_y, train_folds, clinical[val_rows], seed + outer * 10
        )
        clinical_teacher[val_rows] = teacher_val
        sample = _patient_unique_sample(
            train_y,
            hard[train_rows],
            train_patients,
            per_class=args.per_class,
            seed=seed + outer,
        )
        runs = []
        for restart in range(args.restarts):
            run = _fit_restart(
                train_q[sample],
                train_y[sample],
                teacher_train[sample],
                train_q,
                val_q,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=seed + outer * 1000 + restart * 100_000,
                device=device,
            )
            runs.append(run)
            restart_columns[restart][val_rows] = run["raw_logit"]
        raw[val_rows] = np.mean([run["raw_logit"] for run in runs], axis=0)
        z_score[val_rows] = np.mean([run["z_score"] for run in runs], axis=0)
        cdf_score[val_rows] = np.mean([run["cdf_score"] for run in runs], axis=0)
        observable_oof[val_rows] = np.mean([run["observables"] for run in runs], axis=0)
        no_ent = _fit_restart(
            train_q[sample],
            train_y[sample],
            teacher_train[sample],
            train_q,
            val_q,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=seed + outer * 1000,
            device=device,
            entanglement=False,
        )
        no_ent_raw[val_rows] = no_ent["raw_logit"]
        no_ent_cdf[val_rows] = no_ent["cdf_score"]
        q4_logistic[val_rows] = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000
        ).fit(train_q[sample], train_y[sample]).decision_function(val_q)
        mlp_runs = []
        for restart in range(args.restarts):
            mlp_runs.append(
                _safe_logit(
                    _mlp((1e-3, 1e-3), seed + outer * 1000 + restart)
                    .fit(train_q[sample], train_y[sample])
                    .predict_proba(val_q)[:, 1]
                )
            )
        q4_mlp[val_rows] = np.mean(mlp_runs, axis=0)
        fold_audits.append(
            {
                "seed": seed,
                "outer_fold": outer,
                "q4_label_correlations": q_audit,
                "teacher": teacher_audit,
                "sample_records": int(len(sample)),
                "sample_patients": int(len(np.unique(train_patients[sample]))),
                "restarts": [run["audit"] for run in runs],
                "no_entanglement": no_ent["audit"],
            }
        )
    scores = {
        "vqc_raw": raw,
        "vqc_train_z": z_score,
        "vqc_train_cdf": cdf_score,
        "vqc_no_entanglement_raw": no_ent_raw,
        "vqc_no_entanglement_cdf": no_ent_cdf,
        "q4_logistic": q4_logistic,
        "q4_mlp_ensemble": q4_mlp,
    }
    calibrated, calibration_audits = {}, {}
    for name, score in scores.items():
        if not np.isfinite(score).all():
            raise RuntimeError(f"Incomplete score: {name}")
        calibrated[name], calibration_audits[name] = _cross_fitted_calibration(score, labels, folds)
    fusion_cdf, fusion_cdf_audit = _cross_fitted_fusion(
        _safe_logit(clinical_teacher), _safe_logit(calibrated["vqc_train_cdf"]), labels, folds
    )
    fusion_mlp, fusion_mlp_audit = _cross_fitted_fusion(
        _safe_logit(clinical_teacher), _safe_logit(calibrated["q4_mlp_ensemble"]), labels, folds
    )
    calibrated["clinical_teacher"] = clinical_teacher
    calibrated["fusion_vqc_train_cdf"] = expit(fusion_cdf)
    calibrated["fusion_q4_mlp"] = expit(fusion_mlp)
    metrics = {name: _metrics(labels, probability) for name, probability in calibrated.items()}
    operating = {
        name: _operating_point(labels, probability, hard)
        for name, probability in calibrated.items()
    }
    comparisons = {}
    for left, right in (
        ("vqc_train_cdf", "q4_logistic"),
        ("vqc_train_cdf", "q4_mlp_ensemble"),
        ("vqc_train_cdf", "vqc_raw"),
        ("vqc_train_cdf", "vqc_no_entanglement_cdf"),
        ("fusion_vqc_train_cdf", "fusion_q4_mlp"),
    ):
        key = f"{left}_minus_{right}"
        comparisons[key] = _paired_patient_bootstrap(
            labels,
            patients,
            calibrated[left],
            calibrated[right],
            iterations=args.bootstrap_iterations,
            seed=seed,
            comparison=key,
        )
    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "ecg_id": record_ids,
            "patient_id": patients,
            "strat_fold": folds,
            "y_true": labels,
            "hard_negative": hard,
            **{f"score_{name}": value for name, value in scores.items()},
            **calibrated,
            **{f"restart_{index}_raw_logit": value for index, value in restart_columns.items()},
            **{f"q4_{index}": q4_oof[:, index] for index in range(4)},
            **{f"quantum_observable_{index}": observable_oof[:, index] for index in range(8)},
        }
    )
    frame.to_csv(seed_dir / "oof_predictions_and_observables.csv", index=False)
    for filename, value in (
        ("metrics.json", metrics),
        ("operating_points.json", operating),
        ("bootstrap.json", comparisons),
        ("fold_audits.json", fold_audits),
        ("calibration_audits.json", calibration_audits),
        ("fusion_audits.json", {"cdf": fusion_cdf_audit, "mlp": fusion_mlp_audit}),
    ):
        (seed_dir / filename).write_text(json.dumps(value, indent=2, default=_json_default))
    verdict = {
        "seed": seed,
        "vqc_raw_auprc": metrics["vqc_raw"]["auprc"],
        "vqc_train_cdf_auprc": metrics["vqc_train_cdf"]["auprc"],
        "q4_logistic_auprc": metrics["q4_logistic"]["auprc"],
        "q4_mlp_ensemble_auprc": metrics["q4_mlp_ensemble"]["auprc"],
        "fusion_vqc_train_cdf_auprc": metrics["fusion_vqc_train_cdf"]["auprc"],
        "fusion_q4_mlp_auprc": metrics["fusion_q4_mlp"]["auprc"],
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    (seed_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
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
    preflight = {
        "records": int(len(joined)),
        "patients": int(joined.patient_id.nunique()),
        "folds": sorted(joined.strat_fold.unique().astype(int).tolist()),
        "seeds": args.seeds,
        "restarts": args.restarts,
        "candidate": FIXED_CANDIDATE.__dict__,
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2))
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    clinical = joined[approved].to_numpy(np.float32)
    for seed in args.seeds:
        _run_seed(int(seed), joined, clinical, paths, args, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
