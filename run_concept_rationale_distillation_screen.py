"""Patient-safe concept-rationale and teacher-assistant screen for q4 VQC.

The MI prediction path is unchanged: Transformer h128 -> fold-local PLS q4 ->
four-qubit VQC -> linear quantum-observable readout.  During training only, an
auxiliary head predicts deployable ECG concepts from the same Z/ZZ quantum
observables.  An inner-OOF q4 MLP supplies answer-level teacher probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier

from aquire_preprocessing.concept_distillation import (
    CONCEPT_COLUMNS,
    FROZEN_CONCEPT_SCREEN,
    FoldLocalConceptTransform,
    assistant_reliability_weights,
    bernoulli_js_per_record,
    masked_concept_loss,
)
from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier
from run_advanced_quantum_fusion_screen import _fit_q4, _seed
from run_divergence_distillation_screen import (
    _metrics,
    _safe_logit,
    _validate_representations,
)
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _json_default, _patient_unique_sample


def _assistant_model(seed: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(7,),
        activation="tanh",
        alpha=1e-3,
        learning_rate_init=2e-3,
        max_iter=600,
        early_stopping=False,
        random_state=int(seed),
    )


def _oof_q4_assistant(
    train_q: np.ndarray,
    train_y: np.ndarray,
    train_folds: np.ndarray,
    val_q: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Generate outer-training OOF soft targets and outer-validation scores."""

    raw_oof = np.full(len(train_y), np.nan, dtype=float)
    unique_folds = sorted(np.unique(train_folds).astype(int))
    if len(unique_folds) < 3:
        raise ValueError("Assistant requires at least three inner folds")
    for inner_fold in unique_folds:
        inner_val = train_folds == inner_fold
        inner_fit = ~inner_val
        model = _assistant_model(seed + inner_fold)
        model.fit(train_q[inner_fit], train_y[inner_fit])
        raw_oof[inner_val] = model.predict_proba(train_q[inner_val])[:, 1]
    if not np.isfinite(raw_oof).all():
        raise RuntimeError("Incomplete inner-OOF assistant predictions")

    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(_safe_logit(raw_oof).reshape(-1, 1), train_y)
    train_probability = calibrator.predict_proba(
        _safe_logit(raw_oof).reshape(-1, 1)
    )[:, 1]
    full = _assistant_model(seed)
    full.fit(train_q, train_y)
    val_raw = full.predict_proba(val_q)[:, 1]
    val_probability = calibrator.predict_proba(
        _safe_logit(val_raw).reshape(-1, 1)
    )[:, 1]
    reliability = assistant_reliability_weights(train_probability, train_y)
    audit = {
        "inner_folds": unique_folds,
        "oof_auprc": float(average_precision_score(train_y, train_probability)),
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0, 0]),
        "reliability_mean": float(reliability.mean()),
        "reliability_nonzero_fraction": float((reliability > 0).mean()),
    }
    return (
        train_probability.astype(np.float32),
        val_probability.astype(np.float32),
        reliability,
        audit,
    )


def _train_candidate(
    train_q: np.ndarray,
    train_y: np.ndarray,
    assistant_probability: np.ndarray,
    reliability: np.ndarray,
    concept_target: np.ndarray,
    concept_observed: np.ndarray,
    val_q: np.ndarray,
    spec,
    *,
    batch_size: int,
    seed: int,
    device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class ConceptStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.quantum = TorchStatevectorQuantumClassifier(
                4, n_layers=2, topology="ring"
            )
            self.concept_head = nn.Linear(8, len(CONCEPT_COLUMNS))

        def forward(self, values):
            observables = self.quantum.quantum_observables(values)
            mi_logit = self.quantum.readout(observables).squeeze(-1)
            concepts = torch.tanh(self.concept_head(observables))
            return mi_logit, concepts

    _seed(seed)
    model = ConceptStudent().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    x = torch.from_numpy(train_q).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    assistant = torch.from_numpy(assistant_probability.astype(np.float32)).to(device)
    confidence = torch.from_numpy(reliability.astype(np.float32)).to(device)
    concepts = torch.from_numpy(concept_target.astype(np.float32)).to(device)
    observed = torch.from_numpy(concept_observed.astype(np.float32)).to(device)
    indices = np.arange(len(train_q))
    rng = np.random.default_rng(seed)
    losses, gradients = [], []
    for epoch in range(spec.stage1_epochs + spec.stage2_epochs):
        rng.shuffle(indices)
        running, seen, epoch_gradients = 0.0, 0, []
        model.train()
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_concepts = model(x[batch])
            hard = functional.binary_cross_entropy_with_logits(logits, y[batch])
            if epoch < spec.stage1_epochs:
                js = bernoulli_js_per_record(logits, assistant[batch])
                if spec.reliability_weighted:
                    weights = confidence[batch]
                    soft = (js * weights).sum() / weights.sum().clamp_min(1.0)
                else:
                    soft = js.mean()
                concept = masked_concept_loss(
                    predicted_concepts, concepts[batch], observed[batch]
                )
                loss = (
                    spec.hard_weight * hard
                    + spec.assistant_weight * soft
                    + spec.concept_weight * concept
                )
            else:
                loss = hard
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
        val_logits, val_concepts = model(torch.from_numpy(val_q).to(device))
        val_logits = val_logits.cpu().numpy()
        val_concepts = val_concepts.cpu().numpy()
    if not np.isfinite(val_logits).all() or not np.isfinite(val_concepts).all():
        raise RuntimeError(f"Non-finite output for {spec.name}")
    quantum_parameters = sum(
        parameter.numel() for parameter in model.quantum.parameters()
    )
    return val_logits, val_concepts, {
        "spec": spec.to_dict(),
        "quantum_inference_parameters": int(quantum_parameters),
        "training_only_concept_parameters": int(
            sum(parameter.numel() for parameter in model.concept_head.parameters())
        ),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "gradient_norm_min": float(np.min(gradients)),
        "gradient_norm_max": float(np.max(gradients)),
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
    missing_concepts = set(CONCEPT_COLUMNS) - set(approved)
    if missing_concepts:
        raise ValueError(f"Concepts are not approved deployable features: {missing_concepts}")
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
    raw_concepts = joined[list(CONCEPT_COLUMNS)].to_numpy(np.float32)
    row_for_id = {int(ecg_id): row for row, ecg_id in enumerate(record_ids)}
    representation_paths = _validate_representations(args.representations, joined)
    preflight = {
        "records": int(len(joined)),
        "patients": int(joined.patient_id.nunique()),
        "concepts": list(CONCEPT_COLUMNS),
        "concept_count": len(CONCEPT_COLUMNS),
        "folds": sorted(np.unique(folds).astype(int).tolist()),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2))
    if args.preflight_only:
        print(json.dumps(preflight, indent=2), flush=True)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    specs = {spec.name: spec for spec in FROZEN_CONCEPT_SCREEN}
    predictions = {
        name: np.full(len(joined), np.nan, dtype=float)
        for name in [f"vqc_{name}" for name in specs]
        + ["q4_assistant_full", "q4_logistic", "q4_mlp"]
    }
    concept_predictions = {
        name: np.full((len(joined), len(CONCEPT_COLUMNS)), np.nan, dtype=float)
        for name in specs
    }
    transformed_concepts = np.full_like(raw_concepts, np.nan, dtype=np.float32)
    observed_concepts = np.zeros_like(raw_concepts, dtype=np.float32)
    assistant_audits, quantum_audits = [], []

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
        predictions["q4_assistant_full"][val_rows] = assistant_val
        assistant_audits.append({"fold": held_out, **assistant_audit})

        concept_transform = FoldLocalConceptTransform()
        train_concepts, train_observed = concept_transform.fit_transform(
            raw_concepts[train_rows]
        )
        val_concepts, val_observed = concept_transform.transform(raw_concepts[val_rows])
        transformed_concepts[val_rows] = val_concepts
        observed_concepts[val_rows] = val_observed

        sample = _patient_unique_sample(
            train_y,
            hard[train_rows],
            train_patients,
            per_class=args.per_class,
            seed=args.seed + held_out,
        )
        q_sample, y_sample = train_q[sample], train_y[sample]
        for name, spec in specs.items():
            logits, predicted_concepts, audit = _train_candidate(
                q_sample,
                y_sample,
                assistant_train[sample],
                reliability[sample],
                train_concepts[sample],
                train_observed[sample],
                val_q,
                spec,
                batch_size=args.batch_size,
                seed=args.seed + held_out * 100,
                device=device,
            )
            predictions[f"vqc_{name}"][val_rows] = expit(logits)
            concept_predictions[name][val_rows] = predicted_concepts
            quantum_audits.append(
                {"fold": held_out, "q4_label_correlations": correlations, **audit}
            )
            print(
                f"  {name}: {average_precision_score(labels[val_rows], expit(logits)):.4f}",
                flush=True,
            )

        logistic_model = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000
        ).fit(q_sample, y_sample)
        predictions["q4_logistic"][val_rows] = logistic_model.predict_proba(val_q)[:, 1]
        mlp_model = _assistant_model(args.seed + held_out * 100 + 1).fit(
            q_sample, y_sample
        )
        predictions["q4_mlp"][val_rows] = mlp_model.predict_proba(val_q)[:, 1]

    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError("Incomplete OOF predictions")
    metrics = {name: _metrics(labels, value) for name, value in predictions.items()}
    concept_metrics = {}
    for name, prediction in concept_predictions.items():
        observed = observed_concepts.astype(bool)
        absolute_error = np.abs(prediction - transformed_concepts)
        per_concept = {}
        for index, concept_name in enumerate(CONCEPT_COLUMNS):
            selected = observed[:, index]
            correlation = np.corrcoef(
                prediction[selected, index], transformed_concepts[selected, index]
            )[0, 1] if selected.sum() > 2 else np.nan
            per_concept[concept_name] = {
                "mae": float(absolute_error[selected, index].mean()),
                "correlation": float(correlation) if np.isfinite(correlation) else None,
                "coverage": float(selected.mean()),
            }
        concept_metrics[name] = per_concept

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
    candidates = [name for name in specs if name != "hard"]
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
    verdict = {
        "screen_winner": winner,
        "winner_auprc": metrics[f"vqc_{winner}"]["auprc"],
        "hard_control_auprc": metrics["vqc_hard"]["auprc"],
        "best_identical_q4_control": best_control,
        "best_identical_q4_control_auprc": metrics[best_control]["auprc"],
        "advance_to_five_seeds": bool(
            metrics[f"vqc_{winner}"]["auprc"] - metrics["vqc_hard"]["auprc"]
            >= 0.005
            and hard_delta["ci95_low"] > 0.0
        ),
        "quantum_predictive_advantage_gate": bool(control_delta["ci95_low"] > 0.0),
        "claim_boundary": (
            "Concept distillation can improve training and interpretability; it cannot "
            "by itself establish computational quantum advantage."
        ),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }

    pd.DataFrame(
        {
            "ecg_id": record_ids,
            "patient_id": patients,
            "strat_fold": folds,
            "y_true": labels,
            **predictions,
        }
    ).to_csv(args.output / "oof_predictions.csv", index=False)
    pd.DataFrame(
        [{"model": name, **values} for name, values in metrics.items()]
    ).sort_values("auprc", ascending=False).to_csv(
        args.output / "metrics.csv", index=False
    )
    (args.output / "assistant_audits.json").write_text(
        json.dumps(assistant_audits, indent=2, default=_json_default)
    )
    (args.output / "quantum_audits.json").write_text(
        json.dumps(quantum_audits, indent=2, default=_json_default)
    )
    (args.output / "concept_metrics.json").write_text(
        json.dumps(concept_metrics, indent=2, default=_json_default)
    )
    (args.output / "bootstrap.json").write_text(
        json.dumps(comparisons, indent=2, default=_json_default)
    )
    (args.output / "experiment_config.json").write_text(
        json.dumps(
            {
                "specs": [spec.to_dict() for spec in FROZEN_CONCEPT_SCREEN],
                "concepts": list(CONCEPT_COLUMNS),
                "per_class": args.per_class,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "assistant": "inner-official-fold OOF q4 MLP with sigmoid calibration",
            },
            indent=2,
        )
    )
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"metrics": metrics, "verdict": verdict}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--preflight-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
