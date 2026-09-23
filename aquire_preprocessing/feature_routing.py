"""Evidence tools for classical/quantum feature routing.

The functions in this module do not select a production feature set.  They
turn cross-fitted development predictions into testable routing hypotheses:

* ``CLASSICAL_CORE``: stable deployable measurements with reproducible main
  effects;
* ``QUANTUM_RESIDUAL_CANDIDATE``: measurements associated with errors left by
  the classical expert and with patient-level loss differences between the
  quantum and classical experts;
* ``SHARED_ABLATION``: evidence for both roles, requiring a nested ablation;
* ``REVIEW``: insufficient evidence for branch assignment.

Final routing must be repeated inside every outer training fold.  A table
produced from pooled OOF predictions is an audit and hypothesis generator,
not a valid input to the final held-out model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .feature_evidence import benjamini_hochberg


@dataclass(frozen=True)
class RoutingThresholds:
    """Prespecified thresholds for exploratory routing hypotheses."""

    fdr_alpha: float = 0.05
    minimum_abs_rho: float = 0.03
    minimum_fold_sign_consistency: float = 0.75
    minimum_marginal_auc_distance: float = 0.05
    minimum_effect_size: float = 0.10


def feature_family(name: str) -> str:
    """Map a deployable feature name to a clinically readable family."""

    value = str(name).lower()
    if value.startswith("clinical__"):
        if "st_" in value or "st_reciprocity" in value or "st_contrast" in value:
            return "territorial_st_pattern"
        if "t_inversion" in value:
            return "territorial_t_pattern"
        if "r_" in value or "rs_" in value or "transition" in value:
            return "territorial_qrs_pattern"
        if "axis" in value:
            return "axis_pattern"
        return "clinical_composite"
    if value in {"heart_rate_bpm", "rr_median_ms", "rr_iqr_ms", "rr_cv"}:
        return "rhythm"
    if value.endswith("__st60_mv"):
        return "lead_st"
    if value.endswith("__t_polarity"):
        return "lead_t_polarity"
    if value.endswith(("__r_amp_mv", "__s_amp_mv", "__rs_ratio", "__range_mv")):
        return "lead_qrs_morphology"
    if "interval" in value or value.startswith("qtc_"):
        return "interval"
    return "other"


def binary_log_loss_per_record(labels: np.ndarray, probability: np.ndarray) -> np.ndarray:
    """Return finite per-record binary cross entropy."""

    y = np.asarray(labels, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    if y.shape != p.shape:
        raise ValueError("Labels and probabilities must have identical shape")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Labels and probabilities must be finite")
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 20 or np.unique(x[valid]).size < 2:
        return np.nan, np.nan, int(valid.sum())
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic), float(result.pvalue), int(valid.sum())


def _fold_association(
    values: np.ndarray,
    target: np.ndarray,
    folds: np.ndarray,
) -> tuple[float, float, float]:
    correlations = []
    for fold in sorted(np.unique(folds)):
        mask = folds == fold
        rho, _, _ = _safe_spearman(values[mask], target[mask])
        if np.isfinite(rho):
            correlations.append(rho)
    if not correlations:
        return np.nan, np.nan, np.nan
    correlations = np.asarray(correlations, dtype=float)
    nonzero = correlations[np.abs(correlations) > 1e-12]
    if not len(nonzero):
        consistency = 1.0
    else:
        positive = np.mean(nonzero > 0)
        consistency = float(max(positive, 1.0 - positive))
    return float(np.median(correlations)), consistency, float(np.min(np.abs(correlations)))


def route_feature_hypotheses(
    features: pd.DataFrame,
    labels: Sequence[int],
    folds: Sequence[int],
    classical_probability: Sequence[float],
    quantum_probability: Sequence[float],
    *,
    approved_features: Sequence[str],
    excluded_features: Mapping[str, str] | None = None,
    statistical_evidence: pd.DataFrame | None = None,
    thresholds: RoutingThresholds = RoutingThresholds(),
) -> pd.DataFrame:
    """Build a pooled-OOF routing audit from deployable feature measurements.

    ``classical_probability`` and ``quantum_probability`` must be honest OOF
    scores for every row.  Their per-record log-loss difference is positive
    when the quantum model performs better.  Association with that difference
    identifies *patient strata* where the current quantum model behaves
    differently; it does not prove that feeding the named feature to a circuit
    will improve the circuit.  That causal question belongs to nested ablation.
    """

    y = np.asarray(labels, dtype=int)
    fold_array = np.asarray(folds, dtype=int)
    classical = np.asarray(classical_probability, dtype=float)
    quantum = np.asarray(quantum_probability, dtype=float)
    n = len(features)
    if any(len(value) != n for value in (y, fold_array, classical, quantum)):
        raise ValueError("Routing inputs must have the same number of rows")
    if not set(np.unique(fold_array)) <= set(range(1, 9)):
        raise ValueError("Routing audit may only use development folds 1-8")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Routing labels must be binary")

    classical_residual = y.astype(float) - classical
    quantum_advantage = (
        binary_log_loss_per_record(y, classical)
        - binary_log_loss_per_record(y, quantum)
    )
    approved = set(approved_features)
    excluded = dict(excluded_features or {})
    evidence_lookup = {}
    if statistical_evidence is not None and len(statistical_evidence):
        evidence_lookup = statistical_evidence.set_index("feature").to_dict("index")

    rows: list[dict] = []
    candidates = list(dict.fromkeys([*features.columns, *excluded.keys()]))
    for name in candidates:
        if name in excluded:
            rows.append({
                "feature": name,
                "family": feature_family(name),
                "approved": False,
                "route": "EXCLUDE_MEASUREMENT_INVALID",
                "route_reason": excluded[name],
            })
            continue
        if name not in approved or name not in features:
            continue
        values = pd.to_numeric(features[name], errors="coerce").to_numpy(dtype=float)
        residual_rho, residual_p, observed = _safe_spearman(values, classical_residual)
        advantage_rho, advantage_p, _ = _safe_spearman(values, quantum_advantage)
        residual_fold_median, residual_consistency, residual_fold_min = _fold_association(
            values, classical_residual, fold_array
        )
        advantage_fold_median, advantage_consistency, advantage_fold_min = _fold_association(
            values, quantum_advantage, fold_array
        )
        evidence = evidence_lookup.get(name, {})
        rows.append({
            "feature": name,
            "family": feature_family(name),
            "approved": True,
            "observed": observed,
            "coverage": float(np.mean(np.isfinite(values))),
            "classical_residual_rho": residual_rho,
            "classical_residual_p": residual_p,
            "classical_residual_fold_median_rho": residual_fold_median,
            "classical_residual_fold_sign_consistency": residual_consistency,
            "classical_residual_fold_min_abs_rho": residual_fold_min,
            "quantum_advantage_rho": advantage_rho,
            "quantum_advantage_p": advantage_p,
            "quantum_advantage_fold_median_rho": advantage_fold_median,
            "quantum_advantage_fold_sign_consistency": advantage_consistency,
            "quantum_advantage_fold_min_abs_rho": advantage_fold_min,
            "univariate_auc_strength": evidence.get("univariate_auroc_strength", np.nan),
            "cliffs_delta": evidence.get("cliffs_delta", np.nan),
            "marginal_direction_consistency": evidence.get("direction_consistency", np.nan),
            "mutual_information": evidence.get("mutual_information", np.nan),
        })

    output = pd.DataFrame(rows)
    approved_mask = output.get("approved", pd.Series(False, index=output.index)).eq(True)
    for prefix in ("classical_residual", "quantum_advantage"):
        output[f"{prefix}_fdr_q"] = np.nan
        if approved_mask.any():
            output.loc[approved_mask, f"{prefix}_fdr_q"] = benjamini_hochberg(
                output.loc[approved_mask, f"{prefix}_p"].to_numpy(float)
            )

    for index in output.index[approved_mask]:
        row = output.loc[index]
        marginal = (
            np.isfinite(row.univariate_auc_strength)
            and abs(float(row.univariate_auc_strength) - 0.5)
            >= thresholds.minimum_marginal_auc_distance
            and np.isfinite(row.cliffs_delta)
            and abs(float(row.cliffs_delta)) >= thresholds.minimum_effect_size
            and float(row.marginal_direction_consistency) >= thresholds.minimum_fold_sign_consistency
        )
        residual = (
            float(row.classical_residual_fdr_q) <= thresholds.fdr_alpha
            and abs(float(row.classical_residual_rho)) >= thresholds.minimum_abs_rho
            and float(row.classical_residual_fold_sign_consistency)
            >= thresholds.minimum_fold_sign_consistency
        )
        advantage = (
            float(row.quantum_advantage_fdr_q) <= thresholds.fdr_alpha
            and abs(float(row.quantum_advantage_rho)) >= thresholds.minimum_abs_rho
            and float(row.quantum_advantage_fold_sign_consistency)
            >= thresholds.minimum_fold_sign_consistency
        )
        quantum_candidate = residual and advantage
        if marginal and quantum_candidate:
            route = "SHARED_ABLATION"
            reason = "stable main effect and stable association with residual/quantum loss difference"
        elif quantum_candidate:
            route = "QUANTUM_RESIDUAL_CANDIDATE"
            reason = "stable residual and quantum-loss-difference associations"
        elif marginal:
            route = "CLASSICAL_CORE"
            reason = "stable deployable marginal signal"
        else:
            route = "REVIEW_NESTED_GROUP_ABLATION"
            reason = "approved measurement without sufficient pooled routing evidence"
        output.loc[index, "route"] = route
        output.loc[index, "route_reason"] = reason
    return output.sort_values(["route", "family", "feature"]).reset_index(drop=True)
