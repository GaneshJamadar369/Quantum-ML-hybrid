"""Restartable, patient-stratified feature extraction from accepted HDF5 ECGs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import DEV_FOLDS
from .features import extract_deployable_features
from .manifest import guard_fold_access


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def _index_frame(handle) -> pd.DataFrame:
    required = {
        "accepted_signal", "sample_mask", "lead_mask", "ecg_id", "patient_id",
        "mi_label", "strat_fold", "hard_negative", "eligibility",
    }
    missing = sorted(required.difference(handle.keys()))
    if missing:
        raise ValueError(f"HDF5 feature extraction is missing datasets: {missing}")
    frame = pd.DataFrame({
        "array_index": np.arange(len(handle["ecg_id"]), dtype=int),
        "ecg_id": handle["ecg_id"][:].astype(int),
        "patient_id": handle["patient_id"][:].astype(int),
        "mi_label": handle["mi_label"][:].astype(int),
        "strat_fold": handle["strat_fold"][:].astype(int),
        "hard_negative": handle["hard_negative"][:].astype(bool),
        "eligibility": _decode(handle["eligibility"][:]),
    })
    if not frame.ecg_id.is_unique:
        raise ValueError("HDF5 ecg_id values must be unique")
    if not set(frame.strat_fold.unique()).issubset(set(DEV_FOLDS)):
        raise PermissionError("Feature repair may access development folds 1-8 only")
    guard_fold_access(frame.strat_fold.to_numpy(), purpose="feature_selection")
    if not frame.eligibility.eq("PRIMARY").all():
        raise ValueError("Primary feature extraction HDF5 contains non-primary records")
    return frame


def select_patient_stratified_sample(
    frame: pd.DataFrame,
    sample_size: Optional[int],
    seed: int = 42,
) -> pd.DataFrame:
    """Select at most one ECG per patient, balanced across fold and cohort."""

    if sample_size is None or sample_size >= len(frame):
        return frame.sort_values("array_index").reset_index(drop=True)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    rng = np.random.default_rng(seed)
    candidates = (
        frame.assign(_random=rng.random(len(frame)))
        .sort_values("_random")
        .drop_duplicates("patient_id")
        .drop(columns="_random")
    )
    candidates = candidates.assign(
        cohort=np.where(
            candidates.mi_label.eq(1), "MI",
            np.where(candidates.hard_negative, "HARD_NON_MI", "OTHER_NON_MI"),
        )
    )
    groups = [part for _, part in candidates.groupby(["strat_fold", "cohort"], sort=True)]
    selected: list[pd.DataFrame] = []
    remaining = int(min(sample_size, len(candidates)))
    while remaining and groups:
        next_groups = []
        for part in groups:
            if remaining == 0:
                break
            row = part.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
            selected.append(row)
            part = part.drop(row.index)
            remaining -= 1
            if len(part):
                next_groups.append(part)
        groups = next_groups
    result = pd.concat(selected, ignore_index=True).drop(columns="cohort")
    return result.sort_values("array_index").reset_index(drop=True)


def reextract_features_from_hdf5(
    hdf5_path: Path,
    output_csv: Path,
    sample_size: Optional[int] = None,
    seed: int = 42,
    shard_size: int = 128,
    require_delineation: bool = True,
) -> dict:
    """Extract v0.3 features without rerunning waveform preprocessing.

    Shards are committed atomically. An interrupted job resumes from completed
    ECG IDs and creates the final CSV only after every selected record exists.
    """

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("h5py is required for feature re-extraction") from exc

    hdf5_path, output_csv = Path(hdf5_path), Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    part_dir = output_csv.parent / f"{output_csv.stem}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as handle:
        population = _index_frame(handle)
        selected = select_patient_stratified_sample(population, sample_size, seed)
        selected_ids = set(selected.ecg_id.astype(int))
        completed_ids: set[int] = set()
        for path in sorted(part_dir.glob("part-*.csv")):
            completed_ids.update(pd.read_csv(path, usecols=["ecg_id"]).ecg_id.astype(int))
        unexpected = completed_ids.difference(selected_ids)
        if unexpected:
            raise ValueError(
                "Existing extraction shards were created for a different selection; "
                f"first unexpected ecg_id={min(unexpected)}"
            )

        pending = selected[~selected.ecg_id.isin(completed_ids)]
        next_part = len(list(part_dir.glob("part-*.csv")))
        for start in range(0, len(pending), shard_size):
            rows = []
            for record in pending.iloc[start:start + shard_size].itertuples(index=False):
                index = int(record.array_index)
                bundle = extract_deployable_features(
                    handle["accepted_signal"][index],
                    100,
                    int(record.ecg_id),
                    sample_mask=handle["sample_mask"][index],
                    lead_mask=handle["lead_mask"][index],
                    require_delineation=require_delineation,
                )
                rows.append({
                    "ecg_id": int(record.ecg_id),
                    "patient_id": int(record.patient_id),
                    "mi_label": int(record.mi_label),
                    "strat_fold": int(record.strat_fold),
                    "hard_negative": bool(record.hard_negative),
                    "extractor_version": bundle.extractor,
                    "extractor_failures": json.dumps(bundle.failures),
                    **bundle.values,
                })
            part = part_dir / f"part-{next_part:05d}.csv"
            temporary = part.with_suffix(".csv.tmp")
            pd.DataFrame(rows).to_csv(temporary, index=False)
            temporary.replace(part)
            next_part += 1
            print(f"feature extraction: {min(start + shard_size, len(pending))}/{len(pending)} pending records")

    parts = [pd.read_csv(path) for path in sorted(part_dir.glob("part-*.csv"))]
    combined = pd.concat(parts, ignore_index=True)
    if not combined.ecg_id.is_unique:
        raise ValueError("Duplicate ecg_id found across extraction shards")
    combined = combined.set_index("ecg_id").loc[selected.ecg_id.astype(int)].reset_index()
    if len(combined) != len(selected):
        raise RuntimeError("Feature extraction did not produce every selected ECG")
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(output_csv)

    failure_counts: dict[str, int] = {}
    for raw in combined.extractor_failures:
        for failure in json.loads(raw):
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
    manifest = {
        "status": "COMPLETE",
        "extractor_version": "aquire-local-v0.3.0",
        "source_hdf5": str(hdf5_path),
        "records": int(len(combined)),
        "unique_patients": int(combined.patient_id.nunique()),
        "folds": sorted(combined.strat_fold.astype(int).unique().tolist()),
        "patient_stratified_sample": sample_size is not None,
        "requested_sample_size": sample_size,
        "require_delineation": require_delineation,
        "failure_counts": dict(sorted(failure_counts.items())),
    }
    manifest_path = output_csv.with_suffix(".manifest.json")
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, indent=2))
    tmp_manifest.replace(manifest_path)
    return manifest

