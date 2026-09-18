"""Official-fold OOF classical baselines for Gate G6."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import DEV_FOLDS
from .manifest import guard_fold_access
from .features import FoldLocalTabularTransformer
from .feature_conditioning import FoldLocalOutlierClipper
from .feature_interactions import ClinicalInteractionEngineer
from .imbalance import fold_safe_smote_enn, hard_negative_weights


@dataclass
class ModelMetrics:
    model: str
    auprc: float
    auroc: float
    sensitivity_at_90_specificity: float
    specificity_at_05: float
    sensitivity_at_05: float
    f1_at_05: float
    log_loss: float
    brier: float
    calibration_error: float
    inference_ms_per_record: float
    hard_neg_fpr: float
    normal_specificity: float
    mi_vs_hard_neg_auprc: float
    mi_vs_hard_neg_auroc: float


def _require_sklearn():
    try:
        from sklearn import metrics  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required; install requirements.lock") from exc


def _models(seed: int = 42) -> Dict[str, object]:
    _require_sklearn()
    from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC

    models: Dict[str, object] = {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        "rbf_svc": make_pipeline(
            SimpleImputer(strategy="median"), RobustScaler(),
            CalibratedClassifierCV(
                SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=seed),
                method="sigmoid", cv=3, ensemble=False,
            ),
        ),
        "random_forest": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", min_samples_leaf=2, n_jobs=-1, random_state=seed)),
        "hist_gradient_boosting": make_pipeline(SimpleImputer(strategy="median"), HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, l2_regularization=1.0, random_state=seed)),
        "mlp": make_pipeline(SimpleImputer(strategy="median"), RobustScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=400, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore
        models["xgboost"] = make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=seed, n_jobs=-1,
            ),
        )
    except ImportError as exc:
        raise RuntimeError(
            "XGBoost is required for the declared Phase 6 comparison; install the research extra"
        ) from exc
    return models


def _sensitivity_at_specificity(y: np.ndarray, p: np.ndarray, target_specificity: float = 0.90) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    valid = np.where((1.0 - fpr) >= target_specificity)[0]
    return float(np.max(tpr[valid])) if len(valid) else 0.0


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & (p < upper if upper < 1.0 else p <= upper)
        if mask.any():
            value += float(mask.sum() / total) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return value if total else float("nan")


class FoldLocalPlattCalibrator:
    """OOF-safe Platt scaling.

    For each held-out fold k, the calibrator is trained on the OOF logits
    of all OTHER folds (i.e. folds 1..k-1, k+1..8) and then applied to
    fold k.  This is a strict leave-one-fold-out procedure — the calibrator
    never sees any logit from the fold it is calibrating.
    """

    def __init__(self) -> None:
        self.calibrators_: dict = {}

    def fit_from_oof_logits(
        self, logits: np.ndarray, labels: np.ndarray, folds: np.ndarray
    ) -> "FoldLocalPlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        self.calibrators_ = {}
        for held_out in np.unique(folds):
            calib_train = folds != held_out          # all OTHER folds
            if calib_train.sum() < 10:
                continue
            lr = LogisticRegression(solver="lbfgs", max_iter=500)
            lr.fit(logits[calib_train].reshape(-1, 1), labels[calib_train])
            self.calibrators_[int(held_out)] = lr
        return self

    def transform(self, logits: np.ndarray, folds: np.ndarray) -> np.ndarray:
        p_cal = np.full_like(logits, fill_value=np.nan, dtype=float)
        for held_out, lr in self.calibrators_.items():
            mask = folds == held_out
            if mask.any():
                p_cal[mask] = lr.predict_proba(logits[mask].reshape(-1, 1))[:, 1]
        # Fallback: if any fold has no calibrator, use sigmoid of raw logit
        no_cal = np.isnan(p_cal)
        if no_cal.any():
            p_cal[no_cal] = 1.0 / (1.0 + np.exp(-logits[no_cal]))
        return p_cal


def evaluate_probabilities(y: np.ndarray, p: np.ndarray, model: str, latency_ms: float, hard_negative: np.ndarray = None) -> ModelMetrics:
    from sklearn.metrics import (
        average_precision_score, roc_auc_score, confusion_matrix,
        f1_score, log_loss, brier_score_loss,
    )
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    
    # Subgroup metrics
    hard_neg_fpr = 0.0
    normal_specificity = 0.0
    mi_vs_hard_neg_auprc = 0.0
    mi_vs_hard_neg_auroc = 0.0
    
    if hard_negative is not None:
        # hard_neg_fpr: FPR on hard_negative == True (where y == 0)
        hard_neg_mask = (y == 0) & hard_negative
        if hard_neg_mask.any():
            hard_neg_fpr = float(pred[hard_neg_mask].mean())
            
        # normal_specificity: Specificity on truly normal ECGs (where y == 0 and not hard_negative)
        normal_mask = (y == 0) & (~hard_negative)
        if normal_mask.any():
            normal_specificity = float((1 - pred[normal_mask]).mean())
            
        # mi_vs_hard_neg: restricted to y == 1 OR (y == 0 and hard_negative)
        mi_vs_hn_mask = (y == 1) | ((y == 0) & hard_negative)
        if mi_vs_hn_mask.any() and y[mi_vs_hn_mask].sum() > 0 and (y[mi_vs_hn_mask] == 0).sum() > 0:
            mi_vs_hard_neg_auprc = float(average_precision_score(y[mi_vs_hn_mask], p[mi_vs_hn_mask]))
            mi_vs_hard_neg_auroc = float(roc_auc_score(y[mi_vs_hn_mask], p[mi_vs_hn_mask]))

    return ModelMetrics(
        model=model,
        auprc=float(average_precision_score(y, p)),
        auroc=float(roc_auc_score(y, p)),
        sensitivity_at_90_specificity=_sensitivity_at_specificity(y, p),
        specificity_at_05=float(tn / max(tn + fp, 1)),
        sensitivity_at_05=float(tp / max(tp + fn, 1)),
        f1_at_05=float(f1_score(y, pred)),
        log_loss=float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        brier=float(brier_score_loss(y, p)),
        calibration_error=_expected_calibration_error(y, p),
        inference_ms_per_record=float(latency_ms),
        hard_neg_fpr=hard_neg_fpr,
        normal_specificity=normal_specificity,
        mi_vs_hard_neg_auprc=mi_vs_hard_neg_auprc,
        mi_vs_hard_neg_auroc=mi_vs_hard_neg_auroc,
    )


def run_oof_baselines(
    features: pd.DataFrame,
    labels: Sequence[int],
    folds: Sequence[int],
    patient_ids: Sequence[int],
    output_dir: Path,
    seed: int = 42,
    record_ids: Optional[Sequence[int]] = None,
    qc_groups: Optional[Sequence[str]] = None,
    hard_negative: Optional[Sequence[bool]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate patient-safe OOF predictions using official folds 1–8 only."""
    folds = np.asarray(folds, dtype=int)
    labels = np.asarray(labels, dtype=int)
    patient_ids = np.asarray(patient_ids)
    record_ids = np.asarray(features.index if record_ids is None else record_ids)
    qc_groups = np.asarray(["unknown"] * len(labels) if qc_groups is None else qc_groups)
    hard_negative = np.asarray([False] * len(labels) if hard_negative is None else hard_negative, dtype=bool)
    for name, values in {
        "record_ids": record_ids, "qc_groups": qc_groups, "hard_negative": hard_negative,
    }.items():
        if len(values) != len(labels):
            raise ValueError(f"{name} length differs from labels")
    guard_fold_access(folds, purpose="tuning")
    if not set(np.unique(folds)).issubset(set(DEV_FOLDS)):
        raise ValueError("G6 baselines accept development folds 1–8 only")
    if len(features) != len(labels) or len(labels) != len(folds):
        raise ValueError("Feature, label and fold lengths differ")
    if pd.Index(patient_ids).duplicated().any():
        # Multiple ECGs per patient are allowed, but the official fold for a
        # patient must remain unique.
        patient_fold_counts = pd.DataFrame({"patient": patient_ids, "fold": folds}).groupby("patient").fold.nunique()
        if (patient_fold_counts > 1).any():
            raise ValueError("Patient crosses OOF folds")

    x = features.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    fold_matrices = {}
    selection_report = {}
    
    # Pre-compute pre-modelling conditioning
    for held_out in sorted(np.unique(folds)):
        allowed = [int(value) for value in sorted(np.unique(folds)) if value != held_out]
        train_mask = np.isin(folds, allowed)
        
        # 1. Outlier clipping
        clipper = FoldLocalOutlierClipper()
        clipper.fit(x.loc[train_mask])
        x_clipped = clipper.transform(x)
        
        # 2. Interactions
        interactor = ClinicalInteractionEngineer()
        interactor.fit(x_clipped.loc[train_mask], labels[train_mask])
        x_inter = interactor.transform(x_clipped)
        
        # 3. Features Transformer (Yeo-Johnson, variance, ANOVA+MI, etc.)
        transformer = FoldLocalTabularTransformer().fit(
            x_inter, folds, allowed, labels=labels, max_features=256
        )
        fold_matrices[int(held_out)] = transformer.transform(x_inter)
        selection_report[str(int(held_out))] = {
            "training_folds": allowed,
            "input_columns": transformer.columns_,
            "selected_columns": transformer.selected_columns_,
            "clip_report": clipper.audit_report(),
            "interaction_correlations": interactor.audit_report()
        }
        
    predictions = []
    metric_rows = []
    
    for model_name, template in _models(seed).items():
        probability = np.full(len(x), np.nan, dtype=float)
        total_latency = 0.0
        total_predicted = 0
        for held_out in sorted(np.unique(folds)):
            train = folds != held_out
            valid = folds == held_out
            model = template
            transformed = fold_matrices[int(held_out)]
            
            X_train_f = transformed[train]
            y_train_f = labels[train]
            
            # Apply SMOTE-ENN only for specific models on training fold
            if model_name in ["logistic", "rbf_svc", "mlp"]:
                X_train_f, y_train_f = fold_safe_smote_enn(X_train_f, y_train_f, random_state=seed)
                
            # Compute sample weights (for hard negatives)
            sample_weight = hard_negative_weights(y_train_f, hard_negative[train] if hard_negative is not None else np.zeros_like(y_train_f, dtype=bool))
            
            # Fit model
            if model_name in ["logistic", "random_forest", "hist_gradient_boosting", "xgboost"]:
                # SVM CalibratedClassifierCV and MLP don't natively support sample_weight well in all paths
                try:
                    model.fit(X_train_f, y_train_f, **{f"{model.steps[-1][0]}__sample_weight": sample_weight} if hasattr(model, 'steps') else {'sample_weight': sample_weight})
                except TypeError:
                    model.fit(X_train_f, y_train_f)
            else:
                model.fit(X_train_f, y_train_f)
                
            start = time.perf_counter()
            probability[valid] = model.predict_proba(transformed[valid])[:, 1]
            total_latency += time.perf_counter() - start
            total_predicted += int(valid.sum())
            
        if not np.isfinite(probability).all():
            raise RuntimeError(f"{model_name} did not produce complete OOF probabilities")
            
        latency_ms = total_latency * 1000.0 / max(total_predicted, 1)
        
        # Stage 8: Platt Calibration
        logits = np.log(np.clip(probability, 1e-7, 1 - 1e-7) / np.clip(1 - probability, 1e-7, 1))
        calibrator = FoldLocalPlattCalibrator()
        calibrator.fit_from_oof_logits(logits, labels, folds)
        calibrated_probability = calibrator.transform(logits, folds)
        
        metrics = evaluate_probabilities(labels, calibrated_probability, model_name, latency_ms, hard_negative=hard_negative)
        metric_rows.append(asdict(metrics))
        predictions.append(pd.DataFrame({
            "row_index": np.arange(len(x)), "ecg_id": record_ids, "patient_id": patient_ids,
            "fold": folds, "label": labels, "model": model_name,
            "logit": logits,
            "probability": calibrated_probability, "qc_group": qc_groups,
            "hard_negative": hard_negative,
        }))

    metrics_df = pd.DataFrame(metric_rows).sort_values(["auprc", "brier"], ascending=[False, True])
    predictions_df = pd.concat(predictions, ignore_index=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "classical_oof_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "classical_oof_predictions.csv", index=False)
    # Separate subgroup CSV for hard-negative sensitivity analysis
    subgroup_cols = ["model", "hard_neg_fpr", "normal_specificity",
                     "mi_vs_hard_neg_auprc", "mi_vs_hard_neg_auroc"]
    metrics_df[subgroup_cols].to_csv(output_dir / "subgroup_metrics.csv", index=False)
    # Within 0.005 AUPRC, prefer Brier then latency.
    near = metrics_df[metrics_df.auprc >= float(metrics_df.auprc.max()) - 0.005]
    winner = near.sort_values(["brier", "inference_ms_per_record"]).iloc[0].to_dict()
    (output_dir / "provisional_classical_champion.json").write_text(json.dumps(winner, indent=2))
    (output_dir / "fold_local_feature_selection.json").write_text(
        json.dumps(selection_report, indent=2)
    )
    return metrics_df, predictions_df


def build_resnet1d(n_leads: int = 12):
    """Build the full-waveform baseline lazily so tabular use needs no PyTorch."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for the waveform baseline") from exc

    class ResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int):
            super().__init__()
            padding = dilation
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm1d(channels), nn.ReLU(),
                nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation, bias=False),
                nn.BatchNorm1d(channels),
            )
            self.activation = nn.ReLU()
        def forward(self, x):
            return self.activation(x + self.net(x))

    class AquireResNet1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(n_leads, 64, 9, padding=4, bias=False), nn.BatchNorm1d(64), nn.ReLU())
            self.blocks = nn.Sequential(*[ResidualBlock(64, d) for d in (1, 2, 4, 8)])
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.head = nn.Linear(64, 1)
        def forward(self, x):
            embedding = self.pool(self.blocks(self.stem(x))).squeeze(-1)
            return self.head(embedding).squeeze(-1), embedding
    return AquireResNet1D()
