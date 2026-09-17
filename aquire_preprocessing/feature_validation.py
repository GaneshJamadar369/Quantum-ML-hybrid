"""Reference/deployable feature alignment and comparison reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def coverage_report(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    group_columns = [c for c in ["strat_fold", "mi_label", "hard_negative", "sex"] if c in frame]
    working = frame.copy()
    if "age" in working:
        working["age_group"] = pd.cut(working.age, [-np.inf, 39, 59, 79, np.inf], labels=["<40", "40-59", "60-79", "80+"])
        group_columns.append("age_group")
    rows = []
    for group in group_columns:
        for value, part in working.groupby(group, dropna=False, observed=False):
            for column in feature_columns:
                rows.append({
                    "group": group, "value": str(value), "feature": column,
                    "records": int(len(part)), "coverage": float(part[column].notna().mean()),
                    "missingness": float(part[column].isna().mean()),
                })
    return pd.DataFrame(rows)


def compare_local_to_reference(
    joined: pd.DataFrame,
    feature_pairs: Mapping[str, str],
    output_dir: Path,
) -> pd.DataFrame:
    """Compare deployable measurements to explicitly mapped reference fields."""
    rows = []
    for local, reference in feature_pairs.items():
        if local not in joined or reference not in joined:
            rows.append({"local": local, "reference": reference, "failure": "column_missing"})
            continue
        pair = joined[[local, reference]].apply(pd.to_numeric, errors="coerce").dropna()
        error = pair[local] - pair[reference] if len(pair) else pd.Series(dtype=float)
        rows.append({
            "local": local, "reference": reference,
            "n": int(len(pair)), "coverage": float(len(pair) / max(len(joined), 1)),
            "correlation": float(pair.corr().iloc[0, 1]) if len(pair) > 2 else np.nan,
            "mae": float(error.abs().mean()) if len(error) else np.nan,
            "median_absolute_error": float(error.abs().median()) if len(error) else np.nan,
            "failure_rate": float(1 - len(pair) / max(len(joined), 1)),
        })
    result = pd.DataFrame(rows)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "deployable_vs_reference.csv", index=False)
    (output_dir / "feature_pairs.json").write_text(json.dumps(dict(feature_pairs), indent=2))
    return result
