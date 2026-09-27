"""Paired q4 screen for adaptive k-local quantum measurements.

The representation, circuit, optimization budget, patient splits and score
alignment are frozen to the retained seed-42 q4 experiment.  The only quantum
change is the measurement family.  Each run also trains a fixed-Pauli ablation
and a parameter-matched classical q4 MLP on the identical samples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from aquire_preprocessing.adaptive_observables import (
    AdaptiveObservableQuantumClassifier,
    ParameterMatchedQ4MLP,
)
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
from run_nested_q4_optimization import _combined_loss, _operating_point
from run_neurips_q4_screen import REFERENCE_COLUMNS, _load_reference
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _json_default, _patient_unique_sample
from run_train_reference_score_alignment import (
    FIXED_CANDIDATE,
    _cross_fitted_calibration,
    _train_reference_scores,
)


ARMS = ("adaptive", "fixed_pauli", "matched_mlp")
CLASSICAL_CEILING_AUPRC = 0.8380154031


def _model(arm: str, *, locality: int, matched_parameters: int | None, device):
    if arm == "adaptive":
        model = AdaptiveObservableQuantumClassifier(locality)
    elif arm == "fixed_pauli":
        model = AdaptiveObservableQuantumClassifier(
            locality, trainable_observables=False
        )
    elif arm == "matched_mlp":
        if matched_parameters is None:
            raise ValueError("matched_parameters is required for matched_mlp")
        model = ParameterMatchedQ4MLP(matched_parameters)
    else:
        raise ValueError(f"Unknown arm: {arm}")
    return model.to(device)


def _fit_restart(
    arm,
    locality,
    matched_parameters,
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
):
    import torch

    _seed(seed)
    model = _model(
        arm,
        locality=locality,
        matched_parameters=matched_parameters,
        device=device,
    )
    trainable = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=FIXED_CANDIDATE.learning_rate, weight_decay=1e-4
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
            loss = _combined_loss(
                logits, y[batch], teacher[batch], FIXED_CANDIDATE
            )
            loss.backward()
            gradients.append(float(torch.nn.utils.clip_grad_norm_(trainable, 5.0)))
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
        observables = None
        if hasattr(model, "quantum_observables"):
            observables = model.quantum_observables(validation_tensor).cpu().numpy()
    _, cdf_score, reference_audit = _train_reference_scores(
        reference_logits, validation_logits
    )
    audit = {
        **reference_audit,
        "arm": arm,
        "locality": int(locality),
        "trainable_parameters": int(sum(value.numel() for value in trainable)),
        "total_parameters": int(sum(value.numel() for value in model.parameters())),
        "initial_loss": float(loss_history[0]),
        "final_loss": float(loss_history[-1]),
        "gradient_min": float(np.min(gradient_history)),
        "gradient_max": float(np.max(gradient_history)),
        "loss_history": loss_history,
        "gradient_history": gradient_history,
    }
    if isinstance(model, AdaptiveObservableQuantumClassifier):
        audit["parameters"] = {
            "feature_scales": model.scale_values().detach().cpu().numpy().tolist(),
            "rotations": model.rotations.detach().cpu().numpy().tolist(),
            "interactions": model.interactions.detach().cpu().numpy().tolist(),
            "observable_coefficients_normalized": (
                model.pauli_weights().detach().cpu().numpy().tolist()
            ),
            "observable_subsets": [list(value) for value in model.subsets],
            "readout_weight": model.readout.weight.detach().cpu().numpy().tolist(),
            "readout_bias": model.readout.bias.detach().cpu().numpy().tolist(),
        }
    else:
        audit["parameter_difference_from_adaptive"] = model.parameter_difference
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
    output_count = 6 if args.locality == 2 else 4
    adaptive_probe = AdaptiveObservableQuantumClassifier(args.locality)
    matched_parameters = sum(
        value.numel() for value in adaptive_probe.parameters() if value.requires_grad
    )
    raw = {arm: np.full(len(joined), np.nan) for arm in ARMS}
    cdf = {arm: np.full(len(joined), np.nan) for arm in ARMS}
    q4_oof = np.full((len(joined), 4), np.nan, dtype=np.float32)
    observables = {
        arm: np.full((len(joined), output_count), np.nan, dtype=np.float32)
        for arm in ("adaptive", "fixed_pauli")
    }
    fold_audits = []
    for outer, path in sorted(paths.items()):
        print(f"locality={args.locality} outer={outer}", flush=True)
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
        arm_audits = {}
        for arm_index, arm in enumerate(ARMS):
            runs = [
                _fit_restart(
                    arm,
                    args.locality,
                    matched_parameters,
                    train_q[sample],
                    train_y[sample],
                    teacher_train[sample],
                    train_q,
                    val_q,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    seed=(
                        args.seed
                        + outer * 1000
                        + arm_index * 10_000
                        + restart * 100_000
                    ),
                    device=device,
                )
                for restart in range(args.restarts)
            ]
            raw[arm][val_rows] = np.mean(
                [run["raw_logit"] for run in runs], axis=0
            )
            cdf[arm][val_rows] = np.mean(
                [run["cdf_score"] for run in runs], axis=0
            )
            if arm in observables:
                observables[arm][val_rows] = np.mean(
                    [run["observables"] for run in runs], axis=0
                )
            arm_audits[arm] = [run["audit"] for run in runs]
        fold_audits.append(
            {
                "outer_fold": int(outer),
                "q4_label_correlations": q_audit,
                "teacher": teacher_audit,
                "sample_records": int(len(sample)),
                "sample_patients": int(len(np.unique(train_patients[sample]))),
                "arms": arm_audits,
            }
        )

    if any(not np.isfinite(values).all() for values in (*raw.values(), *cdf.values())):
        raise RuntimeError("Incomplete candidate OOF scores")
    reference_q4 = reference[[f"q4_{index}" for index in range(4)]].to_numpy(float)
    q4_max_error = float(np.max(np.abs(q4_oof - reference_q4)))
    if q4_max_error > 1e-5:
        raise RuntimeError(
            f"Candidate q4 differs from retained paired reference: max error {q4_max_error}"
        )

    probabilities, calibration_audits = {}, {}
    fusion_probabilities, fusion_audits = {}, {}
    for arm in ARMS:
        probabilities[arm], calibration_audits[arm] = _cross_fitted_calibration(
            cdf[arm], labels, folds
        )
        fusion_logit, fusion_audits[arm] = _cross_fitted_fusion(
            _safe_logit(reference["clinical_teacher"].to_numpy(float)),
            _safe_logit(probabilities[arm]),
            labels,
            folds,
        )
        fusion_probabilities[arm] = expit(fusion_logit)

    prefix = f"ano_k{args.locality}"
    scores = {
        f"{prefix}_{arm}": probabilities[arm] for arm in ARMS
    }
    scores.update(
        {f"{prefix}_{arm}_fusion": fusion_probabilities[arm] for arm in ARMS}
    )
    scores.update(
        {f"reference_{name}": reference[name].to_numpy(float) for name in REFERENCE_COLUMNS}
    )
    metrics = {name: _metrics(labels, values) for name, values in scores.items()}
    operating = {
        name: _operating_point(labels, values, hard) for name, values in scores.items()
    }
    adaptive = f"{prefix}_adaptive"
    adaptive_fusion = f"{adaptive}_fusion"
    comparison_pairs = (
        (adaptive, "reference_vqc_train_cdf"),
        (adaptive, f"{prefix}_fixed_pauli"),
        (adaptive, f"{prefix}_matched_mlp"),
        (adaptive, "reference_q4_logistic"),
        (adaptive, "reference_q4_mlp_ensemble"),
        (adaptive_fusion, "reference_fusion_vqc_train_cdf"),
        (adaptive_fusion, f"{prefix}_fixed_pauli_fusion"),
        (adaptive_fusion, f"{prefix}_matched_mlp_fusion"),
        (adaptive_fusion, "reference_fusion_q4_mlp"),
    )
    comparisons = {}
    for left, right in comparison_pairs:
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

    frame_values = {
        "ecg_id": record_ids,
        "patient_id": patients,
        "strat_fold": folds,
        "y_true": labels,
        "hard_negative": hard,
        **{f"{prefix}_{arm}_raw_logit": raw[arm] for arm in ARMS},
        **{f"{prefix}_{arm}_train_cdf": cdf[arm] for arm in ARMS},
        **scores,
        **{f"q4_{index}": q4_oof[:, index] for index in range(4)},
    }
    for arm, values in observables.items():
        for index in range(output_count):
            frame_values[f"{prefix}_{arm}_observable_{index}"] = values[:, index]
    pd.DataFrame(frame_values).to_csv(
        args.output / "oof_predictions_and_observables.csv", index=False
    )
    for filename, value in (
        ("metrics.json", metrics),
        ("operating_points.json", operating),
        ("bootstrap.json", comparisons),
        ("fold_audits.json", fold_audits),
        ("calibration_audit.json", calibration_audits),
        ("fusion_audit.json", fusion_audits),
    ):
        (args.output / filename).write_text(
            json.dumps(value, indent=2, default=_json_default)
        )

    retained_delta = comparisons[
        f"{adaptive}_minus_reference_vqc_train_cdf"
    ]["delta_auprc"]
    fixed_delta = comparisons[
        f"{adaptive}_minus_{prefix}_fixed_pauli"
    ]["delta_auprc"]
    mlp_delta = comparisons[
        f"{adaptive}_minus_{prefix}_matched_mlp"
    ]["delta_auprc"]
    verdict = {
        "locality": int(args.locality),
        "adaptive_auprc": metrics[adaptive]["auprc"],
        "fixed_pauli_auprc": metrics[f"{prefix}_fixed_pauli"]["auprc"],
        "matched_mlp_auprc": metrics[f"{prefix}_matched_mlp"]["auprc"],
        "retained_vqc_auprc": metrics["reference_vqc_train_cdf"]["auprc"],
        "adaptive_fusion_auprc": metrics[adaptive_fusion]["auprc"],
        "classical_system_ceiling_auprc": CLASSICAL_CEILING_AUPRC,
        "q4_max_absolute_error_vs_reference": q4_max_error,
        "promotion_gates": {
            "retained_gain_at_least_0_005": retained_delta["mean"] >= 0.005,
            "retained_ci_above_zero": retained_delta["ci95_low"] > 0.0,
            "beats_fixed_pauli_with_positive_ci": fixed_delta["ci95_low"] > 0.0,
            "beats_matched_mlp_with_positive_ci": mlp_delta["ci95_low"] > 0.0,
            "fusion_beats_classical_ceiling": (
                metrics[adaptive_fusion]["auprc"] > CLASSICAL_CEILING_AUPRC
            ),
        },
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    verdict["promote_to_confirmation"] = all(verdict["promotion_gates"].values())
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)


def run(args):
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    feature_frame = pd.read_csv(args.features).set_index("ecg_id")
    approved = load_feature_manifest(args.manifest, feature_frame.columns)[
        "approved_features"
    ]
    joined = metadata.join(
        feature_frame[approved], how="inner", validate="one_to_one"
    )
    joined = joined[
        joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")
    ].copy()
    guard_fold_access(joined.strat_fold.to_numpy(int), purpose="tuning")
    paths = _validate_representations(args.representations, joined)
    reference = _load_reference(args.reference, joined)
    adaptive = AdaptiveObservableQuantumClassifier(args.locality)
    target_parameters = sum(
        value.numel() for value in adaptive.parameters() if value.requires_grad
    )
    matched = ParameterMatchedQ4MLP(target_parameters)
    preflight = {
        "experiment": "adaptive_nonlocal_observable_q4",
        "locality": int(args.locality),
        "records": int(len(joined)),
        "patients": int(joined.patient_id.nunique()),
        "folds": sorted(joined.strat_fold.unique().astype(int).tolist()),
        "seed": int(args.seed),
        "restarts": int(args.restarts),
        "epochs": int(args.epochs),
        "adaptive_parameters": int(target_parameters),
        "matched_mlp_parameters": int(
            sum(value.numel() for value in matched.parameters())
        ),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
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
    parser.add_argument("--locality", type=int, choices=(2, 3), required=True)
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
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
