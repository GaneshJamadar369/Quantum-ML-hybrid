"""Final bounded difficulty-aware KD screen for the retained q4 VQC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from aquire_preprocessing.concept_distillation import (
    assistant_reliability_weights,
    bernoulli_js_per_record,
)
from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.difficulty_distillation import (
    FROZEN_DIFFICULTY_SCREEN,
    difficulty_weights,
)
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier
from run_advanced_quantum_fusion_screen import _fit_q4, _seed
from run_concept_rationale_distillation_screen import (
    _assistant_model,
    _oof_q4_assistant,
)
from run_divergence_distillation_screen import (
    _metrics,
    _oof_calibrated_teacher,
    _safe_logit,
    _validate_representations,
)
from run_independent_dual_route_screen import _cross_fitted_fusion
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _json_default, _patient_unique_sample


def _train_candidate(
    train_q: np.ndarray,
    train_y: np.ndarray,
    teacher_probability: np.ndarray,
    calibration_reliability: np.ndarray,
    train_hard_negative: np.ndarray,
    val_q: np.ndarray,
    spec,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    device,
) -> tuple[np.ndarray, dict]:
    import torch
    from torch.nn import functional as functional

    _seed(seed)
    model = TorchStatevectorQuantumClassifier(
        4, n_layers=2, topology="ring"
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    x = torch.from_numpy(train_q).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    teacher = torch.from_numpy(teacher_probability.astype(np.float32)).to(device)
    reliability = torch.from_numpy(calibration_reliability.astype(np.float32)).to(device)
    hard_negative = torch.from_numpy(train_hard_negative.astype(np.float32)).to(device)
    indices = np.arange(len(train_q))
    rng = np.random.default_rng(seed)
    losses, gradients, epoch_weights = [], [], []
    for epoch in range(epochs):
        rng.shuffle(indices)
        running, seen, epoch_gradients, recorded_weights = 0.0, 0, [], []
        model.train()
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch])
            hard = functional.binary_cross_entropy_with_logits(logits, y[batch])
            if spec.mode == "hard":
                loss = hard
                weights = torch.ones_like(logits)
            else:
                weights = difficulty_weights(
                    logits,
                    teacher[batch],
                    y[batch],
                    reliability[batch],
                    hard_negative[batch],
                    spec,
                    epoch=epoch,
                    epochs=epochs,
                )
                soft = (
                    bernoulli_js_per_record(logits, teacher[batch]) * weights
                ).mean()
                loss = (1.0 - spec.alpha) * hard + spec.alpha * soft
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch)
            seen += len(batch)
            epoch_gradients.append(float(gradient))
            recorded_weights.append(weights.detach().cpu().numpy())
        losses.append(running / max(seen, 1))
        gradients.append(float(np.median(epoch_gradients)))
        merged_weights = np.concatenate(recorded_weights)
        epoch_weights.append(
            {
                "epoch": epoch + 1,
                "min": float(merged_weights.min()),
                "mean": float(merged_weights.mean()),
                "max": float(merged_weights.max()),
            }
        )
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(val_q).to(device)).cpu().numpy()
    if not np.isfinite(logits).all():
        raise RuntimeError(f"Non-finite VQC output for {spec.name}")
    return logits, {
        "spec": spec.to_dict(),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "gradient_norm_min": float(np.min(gradients)),
        "gradient_norm_max": float(np.max(gradients)),
        "weight_first_epoch": epoch_weights[0],
        "weight_last_epoch": epoch_weights[-1],
        "epochs": int(epochs),
        "simulator": "exact_torch_statevector",
        "device": str(device),
    }


def run(args) -> None:
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    feature_frame = pd.read_csv(args.features).set_index("ecg_id")
    manifest = load_feature_manifest(args.manifest, feature_frame.columns)
    approved = manifest["approved_features"]
    joined = metadata.join(feature_frame[approved], how="inner", validate="one_to_one")
    joined = joined[
        joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")
    ].copy()
    folds = joined.strat_fold.to_numpy(int)
    guard_fold_access(folds, purpose="tuning")
    labels = joined.mi_label.to_numpy(int)
    patients = joined.patient_id.to_numpy(int)
    hard = joined.hard_negative.to_numpy(bool)
    record_ids = joined.index.to_numpy(int)
    clinical = joined[approved].to_numpy(np.float32)
    row_for_id = {int(ecg_id): row for row, ecg_id in enumerate(record_ids)}
    representation_paths = _validate_representations(args.representations, joined)
    preflight = {
        "records": int(len(joined)),
        "patients": int(joined.patient_id.nunique()),
        "folds": sorted(np.unique(folds).astype(int).tolist()),
        "arms": [spec.to_dict() for spec in FROZEN_DIFFICULTY_SCREEN],
        "hard_label_records_removed": 0,
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2))
    if args.preflight_only:
        print(json.dumps(preflight, indent=2), flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = {spec.name: spec for spec in FROZEN_DIFFICULTY_SCREEN}
    predictions = {
        name: np.full(len(joined), np.nan, dtype=float)
        for name in [f"vqc_{name}" for name in specs]
        + ["q4_assistant_full", "q4_logistic", "q4_mlp", "clinical_teacher"]
    }
    assistant_audits, clinical_teacher_audits, quantum_audits = [], [], []

    for held_out in sorted(np.unique(folds).astype(int)):
        print(f"fold {held_out}", flush=True)
        with np.load(representation_paths[held_out], allow_pickle=False) as rep:
            train_ids = rep["train_record_ids"].astype(int)
            val_ids = rep["val_record_ids"].astype(int)
            train_h = rep["train_embeddings"].astype(np.float32)
            val_h = rep["val_embeddings"].astype(np.float32)
            train_y = rep["train_labels"].astype(int)
            train_patients = rep["train_patient_ids"].astype(int)
        train_rows = np.asarray([row_for_id[int(ecg_id)] for ecg_id in train_ids])
        val_rows = np.asarray([row_for_id[int(ecg_id)] for ecg_id in val_ids])
        if not np.array_equal(train_y, labels[train_rows]):
            raise ValueError(f"Fold {held_out} label alignment failed")
        train_q, val_q, correlations = _fit_q4(
            train_h, train_y, val_h, args.seed + held_out
        )
        assistant_train, assistant_val, reliability, assistant_audit = (
            _oof_q4_assistant(
                train_q,
                train_y,
                folds[train_rows],
                val_q,
                args.seed + held_out * 10,
            )
        )
        # Recompute explicitly for audit parity and guard against accidental
        # changes to the imported assistant implementation.
        expected_reliability = assistant_reliability_weights(assistant_train, train_y)
        if not np.allclose(reliability, expected_reliability):
            raise RuntimeError("Assistant reliability implementation drift")
        predictions["q4_assistant_full"][val_rows] = assistant_val
        assistant_audits.append({"fold": held_out, **assistant_audit})

        _, clinical_val, clinical_audit = _oof_calibrated_teacher(
            clinical[train_rows],
            train_y,
            folds[train_rows],
            clinical[val_rows],
            args.seed + held_out * 1000,
        )
        predictions["clinical_teacher"][val_rows] = clinical_val
        clinical_teacher_audits.append({"fold": held_out, **clinical_audit})

        sample = _patient_unique_sample(
            train_y,
            hard[train_rows],
            train_patients,
            per_class=args.per_class,
            seed=args.seed + held_out,
        )
        q_sample, y_sample = train_q[sample], train_y[sample]
        hard_sample = hard[train_rows][sample]
        for name, spec in specs.items():
            logits, audit = _train_candidate(
                q_sample,
                y_sample,
                assistant_train[sample],
                reliability[sample],
                hard_sample,
                val_q,
                spec,
                epochs=args.epochs,
                batch_size=args.batch_size,
                seed=args.seed + held_out * 100,
                device=device,
            )
            predictions[f"vqc_{name}"][val_rows] = expit(logits)
            quantum_audits.append(
                {"fold": held_out, "q4_label_correlations": correlations, **audit}
            )
            print(
                f"  {name}: {average_precision_score(labels[val_rows], expit(logits)):.4f}",
                flush=True,
            )

        logistic = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000
        ).fit(q_sample, y_sample)
        predictions["q4_logistic"][val_rows] = logistic.predict_proba(val_q)[:, 1]
        mlp = _assistant_model(args.seed + held_out * 100 + 1).fit(q_sample, y_sample)
        predictions["q4_mlp"][val_rows] = mlp.predict_proba(val_q)[:, 1]

    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("Incomplete OOF predictions")

    fusion_predictions = {}
    clinical_logit = _safe_logit(predictions["clinical_teacher"])
    fusion_coefficients = {}
    for name in specs:
        fused_logit, coefficients = _cross_fitted_fusion(
            clinical_logit,
            _safe_logit(predictions[f"vqc_{name}"]),
            labels,
            folds,
        )
        fusion_predictions[f"fusion_{name}"] = expit(fused_logit)
        fusion_coefficients[f"fusion_{name}"] = coefficients
    classical_logit, coefficients = _cross_fitted_fusion(
        clinical_logit, _safe_logit(predictions["q4_mlp"]), labels, folds
    )
    fusion_predictions["fusion_q4_mlp"] = expit(classical_logit)
    fusion_coefficients["fusion_q4_mlp"] = coefficients
    all_predictions = {**predictions, **fusion_predictions}
    metrics = {name: _metrics(labels, value) for name, value in all_predictions.items()}

    comparisons = {}
    for name in specs:
        if name == "hard":
            continue
        for control in ("vqc_hard", "q4_logistic", "q4_mlp"):
            key = f"vqc_{name}_minus_{control}"
            comparisons[key] = _paired_patient_bootstrap(
                labels,
                patients,
                predictions[f"vqc_{name}"],
                predictions[control],
                iterations=args.bootstrap_iterations,
                seed=args.seed,
                comparison=key,
            )
        for control in ("fusion_hard", "fusion_q4_mlp"):
            key = f"fusion_{name}_minus_{control}"
            comparisons[key] = _paired_patient_bootstrap(
                labels,
                patients,
                fusion_predictions[f"fusion_{name}"],
                fusion_predictions[control],
                iterations=args.bootstrap_iterations,
                seed=args.seed,
                comparison=key,
            )

    candidates = [name for name in specs if name not in {"hard", "uniform_js"}]
    winner = max(
        candidates,
        key=lambda name: (
            metrics[f"vqc_{name}"]["auprc"],
            -metrics[f"vqc_{name}"]["brier"],
        ),
    )
    hard_delta = comparisons[f"vqc_{winner}_minus_vqc_hard"]["delta_auprc"]
    best_control = max(
        ("q4_logistic", "q4_mlp"), key=lambda name: metrics[name]["auprc"]
    )
    control_delta = comparisons[f"vqc_{winner}_minus_{best_control}"]["delta_auprc"]
    fusion_delta = comparisons[f"fusion_{winner}_minus_fusion_q4_mlp"]["delta_auprc"]
    verdict = {
        "difficulty_winner": winner,
        "winner_auprc": metrics[f"vqc_{winner}"]["auprc"],
        "uniform_js_auprc": metrics["vqc_uniform_js"]["auprc"],
        "hard_control_auprc": metrics["vqc_hard"]["auprc"],
        "best_identical_q4_control": best_control,
        "best_identical_q4_control_auprc": metrics[best_control]["auprc"],
        "winner_fusion_auprc": metrics[f"fusion_{winner}"]["auprc"],
        "all_classical_fusion_auprc": metrics["fusion_q4_mlp"]["auprc"],
        "advance_to_five_seeds": bool(
            metrics[f"vqc_{winner}"]["auprc"] - metrics["vqc_hard"]["auprc"]
            >= 0.005
            and hard_delta["ci95_low"] > 0.0
        ),
        "standalone_quantum_predictive_advantage": bool(
            control_delta["ci95_low"] > 0.0
        ),
        "system_quantum_value_gate": bool(fusion_delta["ci95_low"] > 0.0),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }

    pd.DataFrame(
        {
            "ecg_id": record_ids,
            "patient_id": patients,
            "strat_fold": folds,
            "y_true": labels,
            **all_predictions,
        }
    ).to_csv(args.output / "oof_predictions.csv", index=False)
    pd.DataFrame(
        [{"model": name, **values} for name, values in metrics.items()]
    ).sort_values("auprc", ascending=False).to_csv(
        args.output / "metrics.csv", index=False
    )
    for filename, payload in (
        ("assistant_audits.json", assistant_audits),
        ("clinical_teacher_audits.json", clinical_teacher_audits),
        ("quantum_audits.json", quantum_audits),
        ("fusion_coefficients.json", fusion_coefficients),
        ("bootstrap.json", comparisons),
        ("verdict.json", verdict),
    ):
        (args.output / filename).write_text(
            json.dumps(payload, indent=2, default=_json_default)
        )
    (args.output / "experiment_config.json").write_text(
        json.dumps(
            {
                "specs": [spec.to_dict() for spec in FROZEN_DIFFICULTY_SCREEN],
                "per_class": args.per_class,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "hard_loss_reweighted": False,
                "records_filtered": False,
                "distillation_teacher": "inner-official-fold OOF q4 MLP",
                "fusion_expert": "inner-official-fold OOF clinical HistGB",
            },
            indent=2,
        )
    )
    print(json.dumps({"metrics": metrics, "verdict": verdict}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
