"""Development-fold-only calibration of signal-quality detectors."""

from __future__ import annotations

import json
import ast
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import DEV_FOLDS
from .manifest import guard_fold_access

TECHNICAL_ANNOTATION_COLUMNS = (
    "baseline_drift", "static_noise", "burst_noise", "electrodes_problems",
)


def annotation_present(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "none", "nan", "[]", "{}"}


def annotation_present_for_lead(value: object, lead: object) -> bool:
    """Interpret PTB-XL technical annotations without labelling every lead."""
    if not annotation_present(value):
        return False
    if not isinstance(value, str):
        return bool(value)
    text = value.strip()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        parsed = text
    if isinstance(parsed, dict):
        named = {str(key).lower() for key, flag in parsed.items() if bool(flag)}
        return str(lead).lower() in named or bool(named & {"all", "global"})
    if isinstance(parsed, (list, tuple, set)):
        named = {str(item).lower() for item in parsed}
        return str(lead).lower() in named or bool(named & {"all", "global"})
    if str(parsed).strip().lower() in {"1", "true", "yes"}:
        return True
    return str(lead).lower() in str(parsed).lower().replace(",", " ").split()


def _operating_points(y: np.ndarray, score: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        prediction = score >= threshold
        tp = int(np.sum(prediction & y)); fn = int(np.sum(~prediction & y))
        tn = int(np.sum(~prediction & ~y)); fp = int(np.sum(prediction & ~y))
        rows.append({
            "threshold": float(threshold),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "false_positive_count": fp,
            "positive_count": int(y.sum()),
            "prevalence": float(y.mean()),
        })
    return pd.DataFrame(rows)


def calibrate_qc_threshold(
    frame: pd.DataFrame,
    score_column: str,
    annotation_columns: Sequence[str],
    minimum_sensitivity: float = 0.90,
) -> tuple[dict, pd.DataFrame]:
    """Choose a detector threshold without looking at model performance."""
    required = {"strat_fold", score_column, *annotation_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"QC calibration columns missing: {sorted(missing)}")
    guard_fold_access(frame.strat_fold, purpose="qc_calibration")
    if not set(frame.strat_fold.unique()).issubset(set(DEV_FOLDS)):
        raise ValueError("QC calibration accepts development folds 1–8 only")
    if "lead" in frame.columns:
        y = np.asarray([
            any(annotation_present_for_lead(row[column], row["lead"]) for column in annotation_columns)
            for _, row in frame.iterrows()
        ], dtype=bool)
    else:
        annotations = frame[list(annotation_columns)]
        mapped = annotations.map(annotation_present) if hasattr(annotations, "map") else annotations.applymap(annotation_present)
        y = mapped.any(axis=1).to_numpy(bool)
    score = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(float)
    keep = np.isfinite(score)
    y, score = y[keep], score[keep]
    if len(score) == 0 or y.sum() == 0 or (~y).sum() == 0:
        raise ValueError("QC calibration requires finite scores and both annotation classes")
    thresholds = np.unique(np.quantile(score, np.linspace(0, 1, 201)))
    table = _operating_points(y, score, thresholds)
    eligible = table[table.sensitivity >= minimum_sensitivity]
    selected = (eligible if len(eligible) else table).sort_values(
        ["specificity", "sensitivity", "threshold"], ascending=[False, False, False]
    ).iloc[0].to_dict()
    selected.update({
        "score_column": score_column,
        "annotation_columns": list(annotation_columns),
        "selection_rule": f"highest specificity with sensitivity >= {minimum_sensitivity:.2f}",
        "n_records": int(len(score)),
        "folds": sorted(int(v) for v in frame.strat_fold.unique()),
        "downstream_model_metric_used": False,
    })
    return selected, table


def write_qc_calibration_report(
    frame: pd.DataFrame,
    detector_specs: Iterable[dict],
    output_dir: Path,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"detectors": {}}
    for spec in detector_specs:
        name = spec["name"]
        selected, curve = calibrate_qc_threshold(
            frame, spec["score"], spec["annotations"], spec.get("minimum_sensitivity", 0.90)
        )
        curve.to_csv(output_dir / f"{name}_operating_points.csv", index=False)
        report["detectors"][name] = selected
    (output_dir / "qc_threshold_validation.json").write_text(json.dumps(report, indent=2))
    return report
