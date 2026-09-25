"""Conditional-information gate and clinically structured q8/4-qubit screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from aquire_preprocessing.concept_distillation import bernoulli_js_per_record
from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.difficulty_distillation import (
    FROZEN_DIFFICULTY_SCREEN,
    difficulty_weights,
)
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_quantum import (
    TorchStatevectorQuantumClassifier,
    TorchStructuredReuploadingQuantumClassifier,
)
from aquire_preprocessing.structured_reuploading import (
    FoldLocalConceptEncoder,
    load_structured_reuploading_config,
)
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


def _hgb(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=15,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=int(seed),
    )


def _logistic() -> LogisticRegression:
    return LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)


def _fit_probability(model, x_train, y_train, x_val, *, balanced_weight=False):
    if balanced_weight:
        model.fit(x_train, y_train, sample_weight=compute_sample_weight("balanced", y_train))
    else:
        model.fit(x_train, y_train)
    return model.predict_proba(x_val)[:, 1]


def _fit_calibrated_rbf(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_folds: np.ndarray,
    val_x: np.ndarray,
) -> np.ndarray:
    oof_score = np.full(len(train_y), np.nan, dtype=float)
    for inner_fold in sorted(np.unique(train_folds).astype(int)):
        inner_val = train_folds == inner_fold
        model = SVC(C=1.0, kernel="rbf", class_weight="balanced")
        model.fit(train_x[~inner_val], train_y[~inner_val])
        oof_score[inner_val] = model.decision_function(train_x[inner_val])
    if not np.isfinite(oof_score).all():
        raise RuntimeError("Incomplete inner-fold RBF calibration scores")
    calibrator = LogisticRegression(C=1.0, max_iter=2000).fit(
        oof_score.reshape(-1, 1), train_y
    )
    full = SVC(C=1.0, kernel="rbf", class_weight="balanced").fit(train_x, train_y)
    return calibrator.predict_proba(full.decision_function(val_x).reshape(-1, 1))[:, 1]


def _train_quantum(
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    *,
    teacher_probability: np.ndarray,
    reliability: np.ndarray,
    hard_negative: np.ndarray,
    spec,
    epochs: int,
    batch_size: int,
    seed: int,
    device,
    disable_entanglement: bool = False,
) -> tuple[np.ndarray, dict]:
    import torch
    from torch.nn import functional

    _seed(seed)
    model = model.to(device)
    if disable_entanglement:
        with torch.no_grad():
            model.interactions.zero_()
        model.interactions.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    x = torch.from_numpy(train_x.astype(np.float32)).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    teacher = torch.from_numpy(teacher_probability.astype(np.float32)).to(device)
    reliability_tensor = torch.from_numpy(reliability.astype(np.float32)).to(device)
    hard_tensor = torch.from_numpy(hard_negative.astype(np.float32)).to(device)
    order = np.arange(len(train_x))
    rng = np.random.default_rng(seed)
    losses, gradients = [], []
    for epoch in range(epochs):
        rng.shuffle(order)
        model.train()
        running, seen, epoch_gradients = 0.0, 0, []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch])
            hard_loss = functional.binary_cross_entropy_with_logits(logits, y[batch])
            if spec.mode == "hard":
                loss = hard_loss
            else:
                weights = difficulty_weights(
                    logits,
                    teacher[batch],
                    y[batch],
                    reliability_tensor[batch],
                    hard_tensor[batch],
                    spec,
                    epoch=epoch,
                    epochs=epochs,
                )
                soft = (bernoulli_js_per_record(logits, teacher[batch]) * weights).mean()
                loss = (1.0 - spec.alpha) * hard_loss + spec.alpha * soft
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite structured quantum loss")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch)
            seen += len(batch)
            epoch_gradients.append(float(gradient))
        losses.append(running / max(seen, 1))
        gradients.append(float(np.median(epoch_gradients)))
    model.eval()
    with torch.inference_mode():
        logits = model(torch.from_numpy(val_x.astype(np.float32)).to(device)).cpu().numpy()
    if not np.isfinite(logits).all():
        raise RuntimeError("Non-finite structured quantum predictions")
    return logits, {
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "gradient_norm_min": float(np.min(gradients)),
        "gradient_norm_max": float(np.max(gradients)),
        "epochs": int(epochs),
        "spec": spec.to_dict(),
        "entanglement": not disable_entanglement,
        "simulator": "exact_torch_statevector",
        "device": str(device),
    }


def _simple_metrics(y: np.ndarray, probability: np.ndarray) -> dict:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1.0 - 1e-7)
    return {
        "auprc": float(average_precision_score(y, probability)),
        "auroc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "logloss": float(log_loss(y, probability)),
    }


def run(args) -> None:
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    feature_frame = pd.read_csv(args.features).set_index("ecg_id")
    manifest = load_feature_manifest(args.manifest, feature_frame.columns)
    approved = manifest["approved_features"]
    config, groups = load_structured_reuploading_config(args.config, approved)
    joined = metadata.join(feature_frame[approved], how="inner", validate="one_to_one")
    joined = joined[
        joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")
    ].copy()
    folds = joined.strat_fold.to_numpy(int)
    guard_fold_access(folds, purpose="tuning")
    if joined.groupby("patient_id").strat_fold.nunique().max() != 1:
        raise RuntimeError("Patient crosses development folds")
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
        "approved_feature_count": len(approved),
        "concept_groups": [
            {"name": group.name, "feature_count": len(group.features)} for group in groups
        ],
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2))
    if args.preflight_only:
        print(json.dumps(preflight, indent=2), flush=True)
        return

    phase_a_names = [
        "q4_hgb", "q4_c4_hgb", "q4_logistic", "q4_c4_logistic",
        *[f"q4_{group.name}_hgb" for group in groups],
    ]
    phase_a = {name: np.full(len(joined), np.nan) for name in phase_a_names}
    fold_cache, encoder_audits = [], []
    for held_out in sorted(np.unique(folds).astype(int)):
        print(f"Stage A fold {held_out}", flush=True)
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
        train_q4, val_q4, q4_correlations = _fit_q4(
            train_h, train_y, val_h, args.seed + held_out
        )
        encoder = FoldLocalConceptEncoder(groups, seed=args.seed + held_out)
        train_c4 = encoder.fit_transform(clinical[train_rows], approved)
        val_c4 = encoder.transform(clinical[val_rows])
        train_q8, val_q8 = np.c_[train_q4, train_c4], np.c_[val_q4, val_c4]
        phase_a["q4_hgb"][val_rows] = _fit_probability(
            _hgb(args.seed + held_out), train_q4, train_y, val_q4,
            balanced_weight=True,
        )
        phase_a["q4_c4_hgb"][val_rows] = _fit_probability(
            _hgb(args.seed + held_out), train_q8, train_y, val_q8,
            balanced_weight=True,
        )
        phase_a["q4_logistic"][val_rows] = _fit_probability(
            _logistic(), train_q4, train_y, val_q4
        )
        phase_a["q4_c4_logistic"][val_rows] = _fit_probability(
            _logistic(), train_q8, train_y, val_q8
        )
        for group_index, group in enumerate(groups):
            phase_a[f"q4_{group.name}_hgb"][val_rows] = _fit_probability(
                _hgb(args.seed + held_out),
                np.c_[train_q4, train_c4[:, group_index]],
                train_y,
                np.c_[val_q4, val_c4[:, group_index]],
                balanced_weight=True,
            )
        encoder_audits.append(
            {"fold": held_out, "q4_label_correlations": q4_correlations, "groups": encoder.audit()}
        )
        fold_cache.append(
            {
                "fold": held_out,
                "train_rows": train_rows,
                "val_rows": val_rows,
                "train_q4": train_q4,
                "val_q4": val_q4,
                "train_c4": train_c4,
                "val_c4": val_c4,
                "train_y": train_y,
                "train_patients": train_patients,
            }
        )
    if any(not np.isfinite(value).all() for value in phase_a.values()):
        raise RuntimeError("Incomplete Stage A OOF predictions")
    phase_a_metrics = {name: _simple_metrics(labels, value) for name, value in phase_a.items()}
    phase_a_bootstrap = {}
    for augmented in ["q4_c4_hgb", *[f"q4_{group.name}_hgb" for group in groups]]:
        key = f"{augmented}_minus_q4_hgb"
        phase_a_bootstrap[key] = _paired_patient_bootstrap(
            labels, patients, phase_a[augmented], phase_a["q4_hgb"],
            iterations=args.bootstrap_iterations, seed=args.seed, comparison=key,
        )
    linear_key = "q4_c4_logistic_minus_q4_logistic"
    phase_a_bootstrap[linear_key] = _paired_patient_bootstrap(
        labels, patients, phase_a["q4_c4_logistic"], phase_a["q4_logistic"],
        iterations=args.bootstrap_iterations, seed=args.seed, comparison=linear_key,
    )
    primary = phase_a_bootstrap["q4_c4_hgb_minus_q4_hgb"]["delta_auprc"]
    minimum = float(config["gate"]["minimum_delta_auprc"])
    observed_delta = (
        phase_a_metrics["q4_c4_hgb"]["auprc"]
        - phase_a_metrics["q4_hgb"]["auprc"]
    )
    stage_a_pass = bool(observed_delta >= minimum and primary["ci95_low"] > 0.0)
    stage_a_gate = {
        "pass": stage_a_pass,
        "primary_comparison": "q4_c4_hgb_minus_q4_hgb",
        "minimum_delta_auprc": minimum,
        "observed_delta_auprc": observed_delta,
        "observed": primary,
        "decision": "RUN_QUANTUM_SCREEN" if stage_a_pass else "STOP_BEFORE_QUANTUM",
    }
    pd.DataFrame({"ecg_id": record_ids, "patient_id": patients, "strat_fold": folds,
                  "y_true": labels, **phase_a}).to_csv(
        args.output / "stage_a_oof_predictions.csv", index=False
    )
    (args.output / "stage_a_metrics.json").write_text(json.dumps(phase_a_metrics, indent=2))
    (args.output / "stage_a_bootstrap.json").write_text(
        json.dumps(phase_a_bootstrap, indent=2, default=_json_default)
    )
    (args.output / "stage_a_gate.json").write_text(json.dumps(stage_a_gate, indent=2))
    (args.output / "concept_encoder_audits.json").write_text(
        json.dumps(encoder_audits, indent=2, default=_json_default)
    )
    print(json.dumps(stage_a_gate, indent=2), flush=True)
    if args.phase_a_only or not stage_a_pass:
        return

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    specs = {spec.name: spec for spec in FROZEN_DIFFICULTY_SCREEN}
    hard_spec, curriculum_spec = specs["hard"], specs["curriculum_difficulty_js"]
    prediction_names = (
        "q4_curriculum_vqc", "structured_hard_vqc", "structured_curriculum_vqc",
        "structured_no_entanglement_vqc", "concept_curriculum_vqc",
        "q4_mlp", "structured_logistic", "structured_mlp", "structured_rbf",
        "clinical_teacher",
    )
    predictions = {name: np.full(len(joined), np.nan) for name in prediction_names}
    quantum_audits = []
    for cache in fold_cache:
        held_out = cache["fold"]
        print(f"Stage B fold {held_out}", flush=True)
        train_rows, val_rows = cache["train_rows"], cache["val_rows"]
        train_q4, val_q4 = cache["train_q4"], cache["val_q4"]
        train_c4, val_c4 = cache["train_c4"], cache["val_c4"]
        train_q8, val_q8 = np.c_[train_q4, train_c4], np.c_[val_q4, val_c4]
        train_y, train_patients = cache["train_y"], cache["train_patients"]
        train_folds = folds[train_rows]
        sample = _patient_unique_sample(
            train_y, hard[train_rows], train_patients,
            per_class=args.per_class, seed=args.seed + held_out,
        )
        q4_teacher, q4_val_teacher, q4_reliability, _ = _oof_q4_assistant(
            train_q4, train_y, train_folds, val_q4, args.seed + 10 * held_out
        )
        q8_teacher, _, q8_reliability, _ = _oof_q4_assistant(
            train_q8, train_y, train_folds, val_q8, args.seed + 20 * held_out
        )
        c4_teacher, _, c4_reliability, _ = _oof_q4_assistant(
            train_c4, train_y, train_folds, val_c4, args.seed + 30 * held_out
        )
        arms = [
            ("q4_curriculum_vqc", TorchStatevectorQuantumClassifier(4, n_layers=2, topology="ring"),
             train_q4, val_q4, q4_teacher, q4_reliability, curriculum_spec, False),
            ("structured_hard_vqc", TorchStructuredReuploadingQuantumClassifier(),
             train_q8, val_q8, q8_teacher, q8_reliability, hard_spec, False),
            ("structured_curriculum_vqc", TorchStructuredReuploadingQuantumClassifier(),
             train_q8, val_q8, q8_teacher, q8_reliability, curriculum_spec, False),
            ("structured_no_entanglement_vqc", TorchStructuredReuploadingQuantumClassifier(),
             train_q8, val_q8, q8_teacher, q8_reliability, curriculum_spec, True),
            ("concept_curriculum_vqc", TorchStatevectorQuantumClassifier(4, n_layers=2, topology="ring"),
             train_c4, val_c4, c4_teacher, c4_reliability, curriculum_spec, False),
        ]
        for arm_index, (name, model, train_x, val_x, teacher, reliability, spec, no_entangle) in enumerate(arms):
            logits, audit = _train_quantum(
                model, train_x[sample], train_y[sample], val_x,
                teacher_probability=teacher[sample], reliability=reliability[sample],
                hard_negative=hard[train_rows][sample], spec=spec,
                epochs=args.epochs, batch_size=args.batch_size,
                seed=args.seed + held_out * 100 + arm_index, device=device,
                disable_entanglement=no_entangle,
            )
            predictions[name][val_rows] = expit(logits)
            quantum_audits.append({"fold": held_out, "arm": name, **audit})
        q8_sample, y_sample = train_q8[sample], train_y[sample]
        predictions["q4_mlp"][val_rows] = _assistant_model(
            args.seed + held_out * 100 + 1
        ).fit(train_q4[sample], y_sample).predict_proba(val_q4)[:, 1]
        predictions["structured_logistic"][val_rows] = _logistic().fit(
            q8_sample, y_sample
        ).predict_proba(val_q8)[:, 1]
        predictions["structured_mlp"][val_rows] = MLPClassifier(
            hidden_layer_sizes=(11,), activation="tanh", alpha=1e-3,
            learning_rate_init=2e-3, max_iter=600,
            random_state=args.seed + held_out * 1000 + 1,
        ).fit(q8_sample, y_sample).predict_proba(val_q8)[:, 1]
        predictions["structured_rbf"][val_rows] = _fit_calibrated_rbf(
            q8_sample, y_sample, train_folds[sample], val_q8
        )
        _, clinical_val, _ = _oof_calibrated_teacher(
            clinical[train_rows], train_y, train_folds,
            clinical[val_rows], args.seed + held_out * 1000,
        )
        predictions["clinical_teacher"][val_rows] = clinical_val
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("Incomplete Stage B OOF predictions")

    fusion = {}
    fusion_audits = {}
    clinical_logit = _safe_logit(predictions["clinical_teacher"])
    for name in ("q4_curriculum_vqc", "structured_curriculum_vqc", "structured_mlp"):
        logits, audit = _cross_fitted_fusion(
            clinical_logit, _safe_logit(predictions[name]), labels, folds
        )
        fusion[f"fusion_{name}"] = expit(logits)
        fusion_audits[f"fusion_{name}"] = audit
    all_predictions = {**predictions, **fusion}
    metrics = {name: _metrics(labels, value) for name, value in all_predictions.items()}
    comparisons = {}
    comparison_pairs = [
        ("structured_curriculum_vqc", "q4_curriculum_vqc"),
        ("structured_curriculum_vqc", "structured_mlp"),
        ("structured_curriculum_vqc", "structured_logistic"),
        ("structured_curriculum_vqc", "structured_rbf"),
        ("structured_curriculum_vqc", "structured_no_entanglement_vqc"),
        ("fusion_structured_curriculum_vqc", "fusion_structured_mlp"),
    ]
    for left, right in comparison_pairs:
        key = f"{left}_minus_{right}"
        comparisons[key] = _paired_patient_bootstrap(
            labels, patients, all_predictions[left], all_predictions[right],
            iterations=args.bootstrap_iterations, seed=args.seed, comparison=key,
        )
    best_control = max(
        ("structured_logistic", "structured_mlp", "structured_rbf"),
        key=lambda name: metrics[name]["auprc"],
    )
    representation_delta = comparisons[
        "structured_curriculum_vqc_minus_q4_curriculum_vqc"
    ]["delta_auprc"]
    quantum_key = f"structured_curriculum_vqc_minus_{best_control}"
    if quantum_key not in comparisons:
        comparisons[quantum_key] = _paired_patient_bootstrap(
            labels, patients, predictions["structured_curriculum_vqc"],
            predictions[best_control], iterations=args.bootstrap_iterations,
            seed=args.seed, comparison=quantum_key,
        )
    quantum_delta = comparisons[quantum_key]["delta_auprc"]
    system_delta = comparisons[
        "fusion_structured_curriculum_vqc_minus_fusion_structured_mlp"
    ]["delta_auprc"]
    verdict = {
        "stage_a_pass": True,
        "representation_gate": bool(
            representation_delta["mean"] >= 0.005 and representation_delta["ci95_low"] > 0.0
        ),
        "best_identical_z8_classical_control": best_control,
        "quantum_gate": bool(quantum_delta["ci95_low"] > 0.0),
        "system_gate": bool(
            system_delta["ci95_low"] > 0.0
            and metrics["fusion_structured_curriculum_vqc"]["auprc"] > 0.83802
        ),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    pd.DataFrame({"ecg_id": record_ids, "patient_id": patients, "strat_fold": folds,
                  "y_true": labels, **all_predictions}).to_csv(
        args.output / "stage_b_oof_predictions.csv", index=False
    )
    (args.output / "stage_b_metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "stage_b_bootstrap.json").write_text(
        json.dumps(comparisons, indent=2, default=_json_default)
    )
    (args.output / "stage_b_quantum_audits.json").write_text(
        json.dumps(quantum_audits, indent=2, default=_json_default)
    )
    (args.output / "stage_b_fusion_audits.json").write_text(
        json.dumps(fusion_audits, indent=2, default=_json_default)
    )
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"metrics": metrics, "verdict": verdict}, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--phase-a-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
