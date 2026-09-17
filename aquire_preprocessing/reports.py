"""Cohort and artifact reports used by Gates G0–G5."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_cohort_report(manifest: pd.DataFrame, output: Path) -> dict:
    report = {
        "records": int(len(manifest)),
        "patients": int(manifest.patient_id.nunique()),
        "mi_records": int(manifest.mi_label.sum()),
        "non_mi_records": int((manifest.mi_label == 0).sum()),
        "hard_negative_records": int(manifest.hard_negative.sum()),
        "by_fold": manifest.groupby("strat_fold").agg(
            records=("patient_id", "size"), patients=("patient_id", "nunique"),
            mi=("mi_label", "sum"), hard_negative=("hard_negative", "sum"),
        ).reset_index().to_dict("records"),
        "label_quality": manifest.label_quality_group.value_counts(dropna=False).to_dict(),
    }
    group_counts = {}
    for values in manifest.hard_negative_groups:
        for group in values:
            group_counts[group] = group_counts.get(group, 0) + 1
    report["hard_negative_groups"] = group_counts
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def write_uncertain_mi_manifest(manifest: pd.DataFrame, output: Path) -> pd.DataFrame:
    uncertain = manifest[
        manifest.mi_label.eq(1) & manifest.label_quality_group.ne("high")
    ][[
        "patient_id", "strat_fold", "mi_scp_codes", "annotation_likelihood_max",
        "annotation_likelihood_known", "label_quality_group", "second_opinion_flag",
        "validated_by_human_flag", "overlapping_superclasses",
    ]].copy()
    uncertain.index.name = "ecg_id"
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    uncertain.to_csv(output)
    return uncertain
