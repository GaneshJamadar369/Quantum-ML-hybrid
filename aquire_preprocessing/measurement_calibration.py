"""Prespecified calibration gate for locally extracted ECG measurements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _bootstrap_intervals(
    local: np.ndarray,
    reference: np.ndarray,
    seed: int,
    iterations: int = 500,
) -> dict:
    rng = np.random.default_rng(seed)
    correlations, median_errors = [], []
    for _ in range(iterations):
        index = rng.integers(0, len(local), len(local))
        left, right = local[index], reference[index]
        correlations.append(np.corrcoef(left, right)[0, 1] if np.std(left) and np.std(right) else np.nan)
        median_errors.append(np.median(np.abs(left - right)))
    return {
        "correlation_ci_low": float(np.nanpercentile(correlations, 2.5)),
        "correlation_ci_high": float(np.nanpercentile(correlations, 97.5)),
        "median_absolute_error_ci_low": float(np.nanpercentile(median_errors, 2.5)),
        "median_absolute_error_ci_high": float(np.nanpercentile(median_errors, 97.5)),
    }


def run_measurement_calibration(
    local_features: Path,
    reference_features: Path,
    feature_pairs: Mapping[str, str],
    thresholds: Mapping[str, Mapping[str, float | str]],
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Compare a patient-stratified local sample with ECGDeli references.

    ECGDeli is a measurement reference, not ground truth. Passing this gate
    permits full re-extraction; it does not establish clinical validity.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local = pd.read_csv(local_features)
    if not local.ecg_id.is_unique:
        raise ValueError("Local calibration ecg_id must be unique")
    if "strat_fold" not in local or not set(local.strat_fold.unique()).issubset(set(range(1, 9))):
        raise PermissionError("Measurement calibration may use folds 1-8 only")
    reference_columns = {name.removeprefix("ref_ecgdeli__") for name in feature_pairs.values()}
    reference = pd.read_csv(
        reference_features,
        usecols=lambda column: column == "ecg_id" or column in reference_columns,
    )
    if not reference.ecg_id.is_unique:
        raise ValueError("ECGDeli reference ecg_id must be unique")
    reference = reference.rename(
        columns={column: f"ref_ecgdeli__{column}" for column in reference.columns if column != "ecg_id"}
    )
    joined = local.merge(reference, on="ecg_id", how="left", validate="one_to_one")

    rows = []
    for ordinal, (local_name, reference_name) in enumerate(feature_pairs.items()):
        rule = thresholds[local_name]
        pair = joined[[local_name, reference_name]].apply(pd.to_numeric, errors="coerce").dropna()
        x, y = pair[local_name].to_numpy(float), pair[reference_name].to_numpy(float)
        if len(pair) >= 3:
            pearson = float(np.corrcoef(x, y)[0, 1])
            spearman = float(spearmanr(x, y).statistic)
            error = x - y
            bootstrap = _bootstrap_intervals(x, y, seed + ordinal)
        else:
            pearson = spearman = np.nan
            error = np.array([], dtype=float)
            bootstrap = {key: np.nan for key in [
                "correlation_ci_low", "correlation_ci_high",
                "median_absolute_error_ci_low", "median_absolute_error_ci_high",
            ]}
        coverage = len(pair) / max(len(joined), 1)
        median_error = float(np.median(np.abs(error))) if len(error) else np.nan
        passed = bool(
            coverage >= float(rule["minimum_coverage"])
            and pearson >= float(rule["minimum_correlation"])
            and median_error <= float(rule["maximum_median_absolute_error"])
        )
        bias = float(np.mean(error)) if len(error) else np.nan
        error_sd = float(np.std(error, ddof=1)) if len(error) > 1 else np.nan
        rows.append({
            "local": local_name,
            "reference": reference_name,
            "group": rule["group"],
            "n": int(len(pair)),
            "coverage": float(coverage),
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
            "mean_bias": bias,
            "median_absolute_error": median_error,
            "bland_altman_lower": bias - 1.96 * error_sd,
            "bland_altman_upper": bias + 1.96 * error_sd,
            **bootstrap,
            "minimum_coverage": float(rule["minimum_coverage"]),
            "minimum_correlation": float(rule["minimum_correlation"]),
            "maximum_median_absolute_error": float(rule["maximum_median_absolute_error"]),
            "point_estimate_pass": passed,
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "measurement_calibration_metrics.csv", index=False)
    timing = metrics[metrics.group.eq("timing")]
    amplitude = metrics[metrics.group.eq("amplitude")]
    timing_passes = int(timing.point_estimate_pass.sum())
    amplitude_passes = int(amplitude.point_estimate_pass.sum())
    decision = "PASS_FULL_REEXTRACTION" if (
        timing_passes == len(timing) and amplitude_passes >= 6
    ) else "STOP_REPAIR_EXTRACTOR"
    conclusion = {
        "decision": decision,
        "records": int(len(joined)),
        "unique_patients": int(joined.patient_id.nunique()) if "patient_id" in joined else None,
        "folds": sorted(joined.strat_fold.astype(int).unique().tolist()),
        "timing_pairs_passed": timing_passes,
        "timing_pairs_required": int(len(timing)),
        "amplitude_pairs_passed": amplitude_passes,
        "amplitude_pairs_required": 6,
        "all_thresholds_prespecified": True,
        "reference_role": "ECGDeli agreement reference; not clinical ground truth",
        "next_action": (
            "re-extract all accepted development ECGs with v0.3"
            if decision == "PASS_FULL_REEXTRACTION"
            else "inspect failed pairs and repair or exclude the affected measurements"
        ),
    }
    path = output_dir / "measurement_calibration_conclusion.json"
    path.write_text(json.dumps(conclusion, indent=2))
    return conclusion

