"""Paired q4 screens for layerwise frequency scaling and tied equilibrium cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.neurips_quantum import (
    LayerwiseFrequencyQuantumClassifier,
    TiedEquilibriumQuantumClassifier,
)
from run_advanced_quantum_fusion_screen import _fit_q4, _seed
from run_divergence_distillation_screen import (
    _metrics,
    _oof_calibrated_teacher,
    _safe_logit,
    _validate_representations,
)
from run_independent_dual_route_screen import _cross_fitted_fusion
from run_nested_q4_optimization import _combined_loss, _operating_point
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _json_default, _patient_unique_sample
from run_train_reference_score_alignment import (
    FIXED_CANDIDATE,
    _cross_fitted_calibration,
    _train_reference_scores,
)


REFERENCE_COLUMNS = (
    "vqc_train_cdf",
    "q4_logistic",
    "q4_mlp_ensemble",
    "clinical_teacher",
    "fusion_vqc_train_cdf",
    "fusion_q4_mlp",
)


def _model(arm: str, *, steps: int, damping: float, device):
    if arm == "frequency":
        model = LayerwiseFrequencyQuantumClassifier(initial_scale=0.5)
    elif arm == "equilibrium":
        model = TiedEquilibriumQuantumClassifier(
            steps=steps, damping=damping, initial_scale=0.5
        )
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return model.to(device)


def _load_reference(path: Path, joined: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_csv(path).set_index("ecg_id")
    required = {
        "patient_id",
        "strat_fold",
        "y_true",
        *REFERENCE_COLUMNS,
        *(f"q4_{index}" for index in range(4)),
    }
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"Retained reference missing columns: {sorted(missing)}")
    if set(frame.index.astype(int)) != set(joined.index.astype(int)):
        raise ValueError("Retained reference and development cohort have different ECG IDs")
    frame = frame.loc[joined.index]
    for column, expected in (
        ("patient_id", joined.patient_id.to_numpy(int)),
        ("strat_fold", joined.strat_fold.to_numpy(int)),
        ("y_true", joined.mi_label.to_numpy(int)),
    ):
        if not np.array_equal(frame[column].to_numpy(int), expected):
            raise ValueError(f"Retained reference {column} is misaligned")
    if not np.isfinite(frame[list(REFERENCE_COLUMNS)].to_numpy(float)).all():
        raise ValueError("Retained reference contains non-finite scores")
    return frame


def _fit_restart(
    arm,
    sample_x,
    sample_y,
    sample_teacher,
    reference_x,
    validation_x,
    *,
    epochs,
    batch_size,
    seed,
    steps,
    damping,
    device,
):
    import torch

    _seed(seed)
    model = _model(arm, steps=steps, damping=damping, device=device)
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
        reference_tensor = torch.from_numpy(reference_x.astype(np.float32)).to(device)
        validation_tensor = torch.from_numpy(validation_x.astype(np.float32)).to(device)
        reference_logits = model(reference_tensor).cpu().numpy()
        validation_logits = model(validation_tensor).cpu().numpy()
        observables = model.quantum_observables(validation_tensor).cpu().numpy()
        convergence = None
        if arm == "equilibrium":
            convergence = model.convergence_residual(validation_tensor).cpu().numpy()
    _, cdf_score, reference_audit = _train_reference_scores(
        reference_logits, validation_logits
    )
    parameters = {
        "feature_scales": model.scale_values().detach().cpu().numpy().tolist(),
        "rotations": model.rotations.detach().cpu().numpy().tolist(),
        "interactions": model.interactions.detach().cpu().numpy().tolist(),
        "readout_weight": model.readout.weight.detach().cpu().numpy().tolist(),
        "readout_bias": model.readout.bias.detach().cpu().numpy().tolist(),
    }
    if hasattr(model, "feature_phases"):
        parameters["feature_phases"] = model.feature_phases.detach().cpu().numpy().tolist()
    audit = {
        **reference_audit,
        "arm": arm,
        "parameters_count": int(sum(value.numel() for value in model.parameters())),
        "initial_loss": float(loss_history[0]),
        "final_loss": float(loss_history[-1]),
        "gradient_min": float(np.min(gradient_history)),
        "gradient_max": float(np.max(gradient_history)),
        "loss_history": loss_history,
        "gradient_history": gradient_history,
        "parameters": parameters,
    }
    if convergence is not None:
        audit["convergence_residual"] = {
            "median": float(np.median(convergence)),
            "p90": float(np.quantile(convergence, 0.90)),
            "p99": float(np.quantile(convergence, 0.99)),
            "maximum": float(np.max(convergence)),
            "fraction_below_0_05": float(np.mean(convergence < 0.05)),
        }
    return {
        "raw_logit": validation_logits,
        "cdf_score": cdf_score,
        "observables": observables,
        "audit": audit,
    }


def _run(joined, clinical, paths, reference, args, device):
    labels = joined.mi_label.to_numpy(int)
    folds = joined.strat_fold.to_numpy(int)
    patients = joined.patient_id.to_numpy(int)
    hard = joined.hard_negative.to_numpy(bool)
    record_ids = joined.index.to_numpy(int)
    row_for_id = {int(value): row for row, value in enumerate(record_ids)}
    raw = np.full(len(joined), np.nan)
    cdf = np.full(len(joined), np.nan)
    q4_oof = np.full((len(joined), 4), np.nan, dtype=np.float32)
    observable_oof = np.full((len(joined), 8), np.nan, dtype=np.float32)
    fold_audits = []
    for outer, path in sorted(paths.items()):
        print(f"arm={args.arm} outer={outer}", flush=True)
        with np.load(path, allow_pickle=False) as data:
            train_ids = data["train_record_ids"].astype(int)
            val_ids = data["val_record_ids"].astype(int)
            train_h = data["train_embeddings"].astype(np.float32)
            val_h = data["val_embeddings"].astype(np.float32)
            train_y = data["train_labels"].astype(int)
            train_patients = data["train_patient_ids"].astype(int)
        train_rows = np.asarray([row_for_id[int(value)] for value in train_ids])
        val_rows = np.asarray([row_for_id[int(value)] for value in val_ids])
        train_q, val_q, q_audit = _fit_q4(
            train_h, train_y, val_h, args.seed + outer
        )
        q4_oof[val_rows] = val_q
        teacher_train, _, teacher_audit = _oof_calibrated_teacher(
            clinical[train_rows],
            train_y,
            folds[train_rows],
            clinical[val_rows],
            args.seed + outer * 10,
        )
        sample = _patient_unique_sample(
            train_y,
            hard[train_rows],
            train_patients,
            per_class=args.per_class,
            seed=args.seed + outer,
        )
        runs = [
            _fit_restart(
                args.arm,
                train_q[sample],
                train_y[sample],
                teacher_train[sample],
                train_q,
                val_q,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed + outer * 1000 + restart * 100_000,
                steps=args.steps,
                damping=args.damping,
                device=device,
            )
            for restart in range(args.restarts)
        ]
        raw[val_rows] = np.mean([run["raw_logit"] for run in runs], axis=0)
        cdf[val_rows] = np.mean([run["cdf_score"] for run in runs], axis=0)
        observable_oof[val_rows] = np.mean(
            [run["observables"] for run in runs], axis=0
        )
        fold_audits.append(
            {
                "outer_fold": int(outer),
                "q4_label_correlations": q_audit,
                "teacher": teacher_audit,
                "sample_records": int(len(sample)),
                "sample_patients": int(len(np.unique(train_patients[sample]))),
                "restarts": [run["audit"] for run in runs],
            }
        )
    if not np.isfinite(raw).all() or not np.isfinite(cdf).all():
        raise RuntimeError("Incomplete candidate OOF scores")
    reference_q4 = reference[[f"q4_{index}" for index in range(4)]].to_numpy(float)
    q4_max_error = float(np.max(np.abs(q4_oof - reference_q4)))
    if q4_max_error > 1e-5:
        raise RuntimeError(
            f"Candidate q4 differs from retained paired reference: max error {q4_max_error}"
        )
    probability, calibration_audit = _cross_fitted_calibration(cdf, labels, folds)
    fusion_logit, fusion_audit = _cross_fitted_fusion(
        _safe_logit(reference["clinical_teacher"].to_numpy(float)),
        _safe_logit(probability),
        labels,
        folds,
    )
    fusion_probability = expit(fusion_logit)
    scores = {
        f"{args.arm}_vqc": probability,
        f"{args.arm}_fusion": fusion_probability,
        **{f"reference_{name}": reference[name].to_numpy(float) for name in REFERENCE_COLUMNS},
    }
    metrics = {name: _metrics(labels, values) for name, values in scores.items()}
    operating = {
        name: _operating_point(labels, values, hard) for name, values in scores.items()
    }
    comparisons = {}
    for left, right in (
        (f"{args.arm}_vqc", "reference_vqc_train_cdf"),
        (f"{args.arm}_vqc", "reference_q4_logistic"),
        (f"{args.arm}_vqc", "reference_q4_mlp_ensemble"),
        (f"{args.arm}_fusion", "reference_fusion_vqc_train_cdf"),
        (f"{args.arm}_fusion", "reference_fusion_q4_mlp"),
    ):
        key = f"{left}_minus_{right}"
        comparisons[key] = _paired_patient_bootstrap(
            labels,
            patients,
            scores[left],
            scores[right],
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
            "candidate_raw_logit": raw,
            "candidate_train_cdf": cdf,
            **scores,
            **{f"q4_{index}": q4_oof[:, index] for index in range(4)},
            **{
                f"quantum_observable_{index}": observable_oof[:, index]
                for index in range(8)
            },
        }
    )
    frame.to_csv(args.output / "oof_predictions_and_observables.csv", index=False)
    for filename, value in (
        ("metrics.json", metrics),
        ("operating_points.json", operating),
        ("bootstrap.json", comparisons),
        ("fold_audits.json", fold_audits),
        ("calibration_audit.json", calibration_audit),
        ("fusion_audit.json", fusion_audit),
    ):
        (args.output / filename).write_text(
            json.dumps(value, indent=2, default=_json_default)
        )
    verdict = {
        "arm": args.arm,
        "candidate_auprc": metrics[f"{args.arm}_vqc"]["auprc"],
        "retained_vqc_auprc": metrics["reference_vqc_train_cdf"]["auprc"],
        "candidate_fusion_auprc": metrics[f"{args.arm}_fusion"]["auprc"],
        "retained_fusion_auprc": metrics["reference_fusion_vqc_train_cdf"]["auprc"],
        "q4_max_absolute_error_vs_reference": q4_max_error,
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)


def run(args):
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    feature_frame = pd.read_csv(args.features).set_index("ecg_id")
    approved = load_feature_manifest(args.manifest, feature_frame.columns)["approved_features"]
    joined = metadata.join(feature_frame[approved], how="inner", validate="one_to_one")
    joined = joined[
        joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")
    ].copy()
    guard_fold_access(joined.strat_fold.to_numpy(int), purpose="tuning")
    paths = _validate_representations(args.representations, joined)
    reference = _load_reference(args.reference, joined)
    source_fingerprint = hashlib.sha256(args.reference.read_bytes()).hexdigest()
    preflight = {
        "arm": args.arm,
        "records": int(len(joined)),
        "patients": int(joined.patient_id.nunique()),
        "folds": sorted(joined.strat_fold.unique().astype(int).tolist()),
        "seed": int(args.seed),
        "restarts": int(args.restarts),
        "epochs": int(args.epochs),
        "steps": int(args.steps),
        "damping": float(args.damping),
        "reference_sha256": source_fingerprint,
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
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _run(
        joined,
        joined[approved].to_numpy(np.float32),
        paths,
        reference,
        args,
        device,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("frequency", "equilibrium"), required=True)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
