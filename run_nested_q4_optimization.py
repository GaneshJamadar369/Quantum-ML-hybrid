"""Nested, multi-seed optimization of the retained four-qubit ECG VQC."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from aquire_preprocessing.concept_distillation import bernoulli_js_per_record
from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier
from run_advanced_quantum_fusion_screen import _fit_q4, _seed
from run_divergence_distillation_screen import (
    _metrics,
    _oof_calibrated_teacher,
    _safe_logit,
    _validate_representations,
)
from run_independent_dual_route_screen import _cross_fitted_fusion
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _json_default, _patient_unique_sample


@dataclass(frozen=True)
class Candidate:
    name: str
    learning_rate: float
    bandwidth: float
    js_weight: float
    ranking_weight: float

    def __post_init__(self):
        if not 0.0 < self.bandwidth < 2.0:
            raise ValueError("Bandwidth must lie in the trainable circuit range (0,2)")
        if self.js_weight < 0 or self.ranking_weight < 0:
            raise ValueError("Loss weights cannot be negative")
        if self.js_weight + self.ranking_weight >= 1.0:
            raise ValueError("BCE must retain positive weight")


CANDIDATES = (
    Candidate("baseline", 5e-3, 1.0, 0.0, 0.0),
    Candidate("low_lr", 2e-3, 1.0, 0.0, 0.0),
    Candidate("narrow_js", 5e-3, 0.5, 0.35, 0.0),
    Candidate("wide_js", 2e-3, 1.5, 0.35, 0.0),
    Candidate("rank15", 5e-3, 1.0, 0.0, 0.15),
    Candidate("js_rank15", 2e-3, 1.0, 0.35, 0.15),
)


def _fingerprint(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def _model(candidate: Candidate, device, *, entanglement: bool = True):
    import torch

    model = TorchStatevectorQuantumClassifier(4, n_layers=2, topology="ring").to(device)
    initial = float(logit(candidate.bandwidth / 2.0))
    with torch.no_grad():
        model.feature_scales.fill_(initial)
        if not entanglement:
            model.interactions.zero_()
    if not entanglement:
        model.interactions.requires_grad_(False)
    return model


def _ranking_loss(logits, labels):
    import torch
    from torch.nn import functional

    positive = logits[labels > 0.5]
    negative = logits[labels <= 0.5]
    if len(positive) == 0 or len(negative) == 0:
        return torch.zeros((), dtype=logits.dtype, device=logits.device)
    return functional.softplus(-(positive[:, None] - negative[None, :])).mean()


def _combined_loss(logits, labels, teacher, candidate: Candidate):
    from torch.nn import functional

    bce = functional.binary_cross_entropy_with_logits(logits, labels)
    js = bernoulli_js_per_record(logits, teacher).mean()
    ranking = _ranking_loss(logits, labels)
    bce_weight = 1.0 - candidate.js_weight - candidate.ranking_weight
    return bce_weight * bce + candidate.js_weight * js + candidate.ranking_weight * ranking


def _search_candidate(
    train_x,
    train_y,
    train_teacher,
    val_x,
    val_y,
    candidate: Candidate,
    *,
    max_epochs: int,
    batch_size: int,
    seed: int,
    device,
) -> dict:
    import torch

    _seed(seed)
    model = _model(candidate, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate.learning_rate, weight_decay=1e-4)
    x = torch.from_numpy(train_x.astype(np.float32)).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    teacher = torch.from_numpy(train_teacher.astype(np.float32)).to(device)
    val_tensor = torch.from_numpy(val_x.astype(np.float32)).to(device)
    order, rng = np.arange(len(train_x)), np.random.default_rng(seed)
    history, gradient_history = [], []
    for epoch in range(1, max_epochs + 1):
        rng.shuffle(order)
        model.train()
        gradients = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch])
            loss = _combined_loss(logits, y[batch], teacher[batch], candidate)
            loss.backward()
            gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            probability = torch.sigmoid(model(val_tensor)).cpu().numpy()
        score = float(average_precision_score(val_y, probability))
        history.append({"epoch": epoch, "inner_auprc": score})
        gradient_history.append(float(np.median(gradients)))
    eligible = [row for row in history if row["epoch"] >= 5]
    winner = max(eligible, key=lambda row: (row["inner_auprc"], -row["epoch"]))
    return {
        "candidate": asdict(candidate),
        "best_epoch": int(winner["epoch"]),
        "best_inner_auprc": float(winner["inner_auprc"]),
        "gradient_min": float(np.min(gradient_history)),
        "gradient_max": float(np.max(gradient_history)),
        "history": history,
    }


def _fit_final_quantum(
    train_x,
    train_y,
    train_teacher,
    val_x,
    candidate: Candidate,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    device,
    entanglement: bool,
):
    import torch

    _seed(seed)
    model = _model(candidate, device, entanglement=entanglement)
    optimizer = torch.optim.Adam(model.parameters(), lr=candidate.learning_rate, weight_decay=1e-4)
    x = torch.from_numpy(train_x.astype(np.float32)).to(device)
    y = torch.from_numpy(train_y.astype(np.float32)).to(device)
    teacher = torch.from_numpy(train_teacher.astype(np.float32)).to(device)
    order, rng = np.arange(len(train_x)), np.random.default_rng(seed)
    losses, gradients = [], []
    for _ in range(epochs):
        rng.shuffle(order)
        model.train()
        running, seen, epoch_gradients = 0.0, 0, []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch])
            loss = _combined_loss(logits, y[batch], teacher[batch], candidate)
            loss.backward()
            epoch_gradients.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)))
            optimizer.step()
            running += float(loss.detach()) * len(batch)
            seen += len(batch)
        losses.append(running / max(seen, 1))
        gradients.append(float(np.median(epoch_gradients)))
    model.eval()
    with torch.inference_mode():
        val_tensor = torch.from_numpy(val_x.astype(np.float32)).to(device)
        logits = model(val_tensor).cpu().numpy()
        observables = model.quantum_observables(val_tensor).cpu().numpy()
    return expit(logits), observables, {
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "trainable_parameters": int(sum(value.numel() for value in model.parameters() if value.requires_grad)),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "gradient_min": float(np.min(gradients)),
        "gradient_max": float(np.max(gradients)),
        "entanglement": bool(entanglement),
    }


def _select_logistic(train_x, train_y, val_x, val_y):
    records = []
    for c_value in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=c_value, class_weight="balanced", max_iter=2000).fit(train_x, train_y)
        score = average_precision_score(val_y, model.predict_proba(val_x)[:, 1])
        records.append((float(score), c_value))
    return max(records)[1], [{"C": value, "inner_auprc": score} for score, value in records]


def _mlp(spec, seed):
    return MLPClassifier(
        hidden_layer_sizes=(7,), activation="tanh", alpha=spec[0],
        learning_rate_init=spec[1], max_iter=600, early_stopping=False,
        random_state=int(seed),
    )


def _select_mlp(train_x, train_y, val_x, val_y, seed):
    records = []
    for alpha in (1e-4, 1e-3):
        for learning_rate in (1e-3, 2e-3):
            spec = (alpha, learning_rate)
            model = _mlp(spec, seed).fit(train_x, train_y)
            score = average_precision_score(val_y, model.predict_proba(val_x)[:, 1])
            records.append((float(score), spec))
    winner = max(records)[1]
    return winner, [
        {"alpha": spec[0], "learning_rate": spec[1], "inner_auprc": score}
        for score, spec in records
    ]


def _select_rbf(train_x, train_y, val_x, val_y):
    records = []
    for c_value in (0.1, 1.0, 10.0):
        for gamma in ("scale", "auto"):
            model = SVC(C=c_value, gamma=gamma, kernel="rbf", class_weight="balanced").fit(train_x, train_y)
            score = average_precision_score(val_y, model.decision_function(val_x))
            records.append((float(score), (c_value, gamma)))
    return max(records)[1], [
        {"C": spec[0], "gamma": spec[1], "inner_auprc": score}
        for score, spec in records
    ]


def _fit_calibrated_rbf(train_x, train_y, train_folds, val_x, spec):
    oof = np.full(len(train_y), np.nan)
    for held_out in sorted(np.unique(train_folds).astype(int)):
        selected = train_folds == held_out
        model = SVC(C=spec[0], gamma=spec[1], kernel="rbf", class_weight="balanced").fit(
            train_x[~selected], train_y[~selected]
        )
        oof[selected] = model.decision_function(train_x[selected])
    calibrator = LogisticRegression(C=1.0, max_iter=2000).fit(oof.reshape(-1, 1), train_y)
    model = SVC(C=spec[0], gamma=spec[1], kernel="rbf", class_weight="balanced").fit(train_x, train_y)
    return calibrator.predict_proba(model.decision_function(val_x).reshape(-1, 1))[:, 1]


def _operating_point(y, probability, hard_negative):
    fpr, tpr, thresholds = roc_curve(y, probability)
    index = int(np.argmin(np.abs(fpr - 0.10)))
    threshold = thresholds[index]
    negative_hard = (y == 0) & hard_negative
    return {
        "threshold_descriptive": float(threshold),
        "sensitivity_at_90_specificity": float(tpr[index]),
        "specificity": float(1.0 - fpr[index]),
        "hard_negative_fpr": float(np.mean(probability[negative_hard] >= threshold)) if negative_hard.any() else None,
    }


def _run_seed(seed: int, joined, clinical, approved, paths, args, device):
    labels = joined.mi_label.to_numpy(int)
    folds = joined.strat_fold.to_numpy(int)
    patients = joined.patient_id.to_numpy(int)
    hard = joined.hard_negative.to_numpy(bool)
    record_ids = joined.index.to_numpy(int)
    row_for_id = {int(value): row for row, value in enumerate(record_ids)}
    names = ("vqc", "vqc_no_entanglement", "q4_logistic", "q4_mlp", "q4_rbf", "clinical_teacher")
    predictions = {name: np.full(len(joined), np.nan) for name in names}
    q4_oof = np.full((len(joined), 4), np.nan, dtype=np.float32)
    observable_oof = np.full((len(joined), 8), np.nan, dtype=np.float32)
    selections, fold_audits = [], []
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
        inner_fold = outer % 8 + 1
        inner_val = train_folds == inner_fold
        inner_fit = ~inner_val
        inner_train_q, inner_val_q, _ = _fit_q4(
            train_h[inner_fit], train_y[inner_fit], train_h[inner_val], seed + outer
        )
        inner_rows = train_rows[inner_fit]
        teacher_inner, _, _ = _oof_calibrated_teacher(
            clinical[inner_rows], train_y[inner_fit], folds[inner_rows],
            clinical[train_rows[inner_val]], seed + outer * 10,
        )
        inner_sample = _patient_unique_sample(
            train_y[inner_fit], hard[inner_rows], train_patients[inner_fit],
            per_class=args.search_per_class, seed=seed + outer,
        )
        candidate_audits = []
        for candidate_index, candidate in enumerate(CANDIDATES):
            candidate_audits.append(
                _search_candidate(
                    inner_train_q[inner_sample], train_y[inner_fit][inner_sample],
                    teacher_inner[inner_sample], inner_val_q, train_y[inner_val], candidate,
                    max_epochs=args.max_epochs, batch_size=args.batch_size,
                    seed=seed + outer * 100 + candidate_index, device=device,
                )
            )
        selected = max(
            candidate_audits,
            key=lambda item: (item["best_inner_auprc"], -item["best_epoch"]),
        )
        candidate = next(value for value in CANDIDATES if value.name == selected["candidate"]["name"])

        train_q, val_q, q_correlations = _fit_q4(train_h, train_y, val_h, seed + outer)
        q4_oof[val_rows] = val_q
        teacher_train, teacher_val, teacher_audit = _oof_calibrated_teacher(
            clinical[train_rows], train_y, train_folds, clinical[val_rows], seed + outer * 10
        )
        predictions["clinical_teacher"][val_rows] = teacher_val
        final_sample = _patient_unique_sample(
            train_y, hard[train_rows], train_patients,
            per_class=args.final_per_class, seed=seed + outer,
        )
        vqc_probability, observables, vqc_audit = _fit_final_quantum(
            train_q[final_sample], train_y[final_sample], teacher_train[final_sample], val_q,
            candidate, epochs=selected["best_epoch"], batch_size=args.batch_size,
            seed=seed + outer * 1000, device=device, entanglement=True,
        )
        no_ent_probability, _, no_ent_audit = _fit_final_quantum(
            train_q[final_sample], train_y[final_sample], teacher_train[final_sample], val_q,
            candidate, epochs=selected["best_epoch"], batch_size=args.batch_size,
            seed=seed + outer * 1000, device=device, entanglement=False,
        )
        predictions["vqc"][val_rows] = vqc_probability
        predictions["vqc_no_entanglement"][val_rows] = no_ent_probability
        observable_oof[val_rows] = observables

        inner_control_sample = _patient_unique_sample(
            train_y[inner_fit], hard[inner_rows], train_patients[inner_fit],
            per_class=args.search_per_class, seed=seed + outer,
        )
        control_x, control_y = inner_train_q[inner_control_sample], train_y[inner_fit][inner_control_sample]
        logistic_c, logistic_search = _select_logistic(control_x, control_y, inner_val_q, train_y[inner_val])
        mlp_spec, mlp_search = _select_mlp(control_x, control_y, inner_val_q, train_y[inner_val], seed + outer)
        rbf_spec, rbf_search = _select_rbf(control_x, control_y, inner_val_q, train_y[inner_val])
        sample_q, sample_y = train_q[final_sample], train_y[final_sample]
        predictions["q4_logistic"][val_rows] = LogisticRegression(
            C=logistic_c, class_weight="balanced", max_iter=2000
        ).fit(sample_q, sample_y).predict_proba(val_q)[:, 1]
        predictions["q4_mlp"][val_rows] = _mlp(mlp_spec, seed + outer * 1000 + 1).fit(
            sample_q, sample_y
        ).predict_proba(val_q)[:, 1]
        predictions["q4_rbf"][val_rows] = _fit_calibrated_rbf(
            sample_q, sample_y, train_folds[final_sample], val_q, rbf_spec
        )
        selections.append({
            "outer_fold": outer, "inner_validation_fold": inner_fold,
            "selected_candidate": selected, "all_candidates": candidate_audits,
            "logistic_search": logistic_search, "mlp_search": mlp_search,
            "rbf_search": rbf_search,
        })
        fold_audits.append({
            "outer_fold": outer, "seed": seed,
            "inner_fit_folds": sorted(np.unique(train_folds[inner_fit]).astype(int).tolist()),
            "inner_validation_fold": inner_fold,
            "outer_validation_fold": outer,
            "final_sample_records": int(len(final_sample)),
            "final_sample_patients": int(len(np.unique(train_patients[final_sample]))),
            "sample_hash": _fingerprint(train_ids[final_sample]),
            "q4_label_correlations": q_correlations,
            "teacher": teacher_audit, "vqc": vqc_audit,
            "vqc_no_entanglement": no_ent_audit,
        })
    if any(not np.isfinite(value).all() for value in predictions.values()):
        raise RuntimeError(f"Incomplete predictions for seed {seed}")

    fusion_vqc, fusion_vqc_audit = _cross_fitted_fusion(
        _safe_logit(predictions["clinical_teacher"]), _safe_logit(predictions["vqc"]), labels, folds
    )
    fusion_mlp, fusion_mlp_audit = _cross_fitted_fusion(
        _safe_logit(predictions["clinical_teacher"]), _safe_logit(predictions["q4_mlp"]), labels, folds
    )
    predictions["fusion_vqc"] = expit(fusion_vqc)
    predictions["fusion_q4_mlp"] = expit(fusion_mlp)
    metrics = {name: _metrics(labels, value) for name, value in predictions.items()}
    operating = {name: _operating_point(labels, value, hard) for name, value in predictions.items()}
    best_control = max(("q4_logistic", "q4_mlp", "q4_rbf"), key=lambda name: metrics[name]["auprc"])
    comparisons = {}
    for left, right in (
        ("vqc", best_control),
        ("vqc", "vqc_no_entanglement"),
        ("fusion_vqc", "fusion_q4_mlp"),
    ):
        key = f"{left}_minus_{right}"
        comparisons[key] = _paired_patient_bootstrap(
            labels, patients, predictions[left], predictions[right],
            iterations=args.bootstrap_iterations, seed=seed, comparison=key,
        )
    seed_dir = args.output / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "ecg_id": record_ids, "patient_id": patients, "strat_fold": folds,
        "y_true": labels, "hard_negative": hard, **predictions,
        **{f"q4_{index}": q4_oof[:, index] for index in range(4)},
        **{f"quantum_observable_{index}": observable_oof[:, index] for index in range(8)},
    })
    frame.to_csv(seed_dir / "oof_predictions_and_observables.csv", index=False)
    (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (seed_dir / "operating_points.json").write_text(json.dumps(operating, indent=2))
    (seed_dir / "bootstrap.json").write_text(json.dumps(comparisons, indent=2, default=_json_default))
    (seed_dir / "selections.json").write_text(json.dumps(selections, indent=2, default=_json_default))
    (seed_dir / "fold_audits.json").write_text(json.dumps(fold_audits, indent=2, default=_json_default))
    (seed_dir / "fusion_audits.json").write_text(json.dumps({
        "fusion_vqc": fusion_vqc_audit, "fusion_q4_mlp": fusion_mlp_audit,
    }, indent=2, default=_json_default))
    verdict = {
        "seed": seed,
        "best_identical_q4_control": best_control,
        "vqc_auprc": metrics["vqc"]["auprc"],
        "control_auprc": metrics[best_control]["auprc"],
        "fusion_vqc_auprc": metrics["fusion_vqc"]["auprc"],
        "fusion_classical_auprc": metrics["fusion_q4_mlp"]["auprc"],
        "fold_9_accessed": False, "fold_10_accessed": False,
    }
    (seed_dir / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2), flush=True)


def run(args):
    import torch

    args.output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    feature_frame = pd.read_csv(args.features).set_index("ecg_id")
    manifest = load_feature_manifest(args.manifest, feature_frame.columns)
    approved = manifest["approved_features"]
    joined = metadata.join(feature_frame[approved], how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")].copy()
    folds = joined.strat_fold.to_numpy(int)
    guard_fold_access(folds, purpose="tuning")
    paths = _validate_representations(args.representations, joined)
    preflight = {
        "records": int(len(joined)), "patients": int(joined.patient_id.nunique()),
        "folds": sorted(np.unique(folds).astype(int).tolist()),
        "seeds": args.seeds, "candidates": [asdict(value) for value in CANDIDATES],
        "fold_9_accessed": False, "fold_10_accessed": False,
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
        _run_seed(int(seed), joined, clinical, approved, paths, args, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--search-per-class", type=int, default=750)
    parser.add_argument("--final-per-class", type=int, default=2000)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
