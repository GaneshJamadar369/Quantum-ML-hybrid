"""Development-fold visual audit for representative ECG and QC cases."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import CANONICAL_LEAD_ORDER, DEV_FOLDS
from .pipeline import process_record


def _select_records(manifest: pd.DataFrame, per_group: int) -> Dict[str, List[int]]:
    dev = manifest[manifest.strat_fold.isin(DEV_FOLDS)]
    normal = dev[(dev.mi_label == 0) & ~dev.hard_negative]
    return {
        "MI": dev[dev.mi_label == 1].index[:per_group].astype(int).tolist(),
        "non-MI": normal.index[:per_group].astype(int).tolist(),
        "hard negative": dev[dev.hard_negative].index[:per_group].astype(int).tolist(),
    }


def create_visual_audit(manifest: pd.DataFrame, output: Path, per_group: int = 2) -> pd.DataFrame:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --visual-audit") from exc
    selections = _select_records(manifest, per_group)
    records = []
    for group, ecg_ids in selections.items():
        for ecg_id in ecg_ids:
            processed = process_record(ecg_id, manifest.loc[ecg_id], sampling_rate=100)
            if processed is not None:
                records.append((group, processed))
    # Scan a bounded development subset for actual QC failures.
    for ecg_id, row in manifest[manifest.strat_fold.isin(DEV_FOLDS)].iloc[:3000].iterrows():
        if sum(group == "QC FAIL" for group, _ in records) >= per_group:
            break
        processed = process_record(int(ecg_id), row, sampling_rate=100)
        if processed is not None and processed.qc_result.qc_status == "FAIL":
            records.append(("QC FAIL", processed))
    if not records:
        raise RuntimeError("No records available for visual audit")

    figure, axes = plt.subplots(len(records), 1, figsize=(16, 2.3 * len(records)), squeeze=False)
    time = np.arange(records[0][1].accepted_signal.shape[1]) / 100.0
    rows = []
    for axis, (group, record) in zip(axes[:, 0], records):
        offsets = np.arange(12)[::-1] * 2.5
        for lead_index, lead in enumerate(CANONICAL_LEAD_ORDER):
            axis.plot(time, record.accepted_signal[lead_index] + offsets[lead_index], lw=0.55)
            axis.text(-0.15, offsets[lead_index], lead, fontsize=7, va="center")
        axis.set_title(f"{group} | ECG {record.ecg_id} | QC {record.qc_result.qc_status}", loc="left", fontsize=9)
        axis.set_xlim(0, 10); axis.set_yticks([]); axis.grid(alpha=0.15)
        rows.append({
            "group": group, "ecg_id": record.ecg_id, "patient_id": record.patient_id,
            "mi_label": record.mi_label, "qc_status": record.qc_result.qc_status,
            "eligibility": record.eligibility.value,
        })
    figure.tight_layout()
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    table = pd.DataFrame(rows)
    table.to_csv(output.with_suffix(".csv"), index=False)
    return table
