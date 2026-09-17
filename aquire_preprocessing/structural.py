"""WFDB ingestion with strict identity checks and lossless validity masks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import numpy as np
import wfdb

from .config import (
    CANONICAL_LEAD_ORDER,
    LEAD_NAME_ALIASES,
    NUM_LEADS,
    PATHS,
    QC,
    SAMPLES_100HZ,
    SAMPLES_500HZ,
)
from .contracts import ValidatedECG


@dataclass
class StructuralValidation(ValidatedECG):
    """Backward-compatible name enriched with research provenance."""

    original_lead_order: List[str] = field(default_factory=list)
    canonical_lead_order: List[str] = field(default_factory=list)
    lead_reorder_applied: bool = False
    gain_applied: bool = False
    duration_seconds: float = 0.0
    n_samples: int = 0
    n_nonfinite: int = 0


def _resolve_record_path(
    ecg_id: int,
    sampling_rate: int = 100,
    metadata_row: Optional[Mapping[str, object]] = None,
) -> str:
    field_name = "filename_lr" if sampling_rate == 100 else "filename_hr"
    if metadata_row is not None and field_name in metadata_row:
        filename = metadata_row[field_name]
        if filename is not None and str(filename) not in {"", "nan"}:
            return str(PATHS.ptbxl_root / str(filename))
    folder = f"{(int(ecg_id) // 1000) * 1000:05d}"
    suffix = "lr" if sampling_rate == 100 else "hr"
    directory = PATHS.records100_dir if sampling_rate == 100 else PATHS.records500_dir
    return str(directory / folder / f"{int(ecg_id):05d}_{suffix}")


def _canonicalize_lead_names(raw_names: List[str]) -> Tuple[List[str], List[str]]:
    canonical, unknown = [], []
    for name in raw_names:
        clean = str(name).strip()
        resolved = LEAD_NAME_ALIASES.get(clean)
        if resolved is None:
            unknown.append(clean)
            canonical.append(clean)
        else:
            canonical.append(resolved)
    return canonical, unknown


def _compute_reorder_indices(current_order: List[str], target_order: List[str]) -> Optional[List[int]]:
    if len(current_order) != len(target_order) or set(current_order) != set(target_order):
        return None
    return [current_order.index(lead) for lead in target_order]


def _source_checksum(record_base: str) -> Tuple[str, List[str]]:
    paths = [Path(record_base + ".hea"), Path(record_base + ".dat")]
    digest = hashlib.sha256()
    present = []
    for path in paths:
        if path.exists():
            present.append(str(path.resolve()))
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return (digest.hexdigest() if present else "", present)


def validate_record(
    ecg_id: int,
    sampling_rate: int = 100,
    record_path: Optional[str] = None,
    metadata_row: Optional[Mapping[str, object]] = None,
) -> StructuralValidation:
    result = StructuralValidation(ecg_id=int(ecg_id))
    if sampling_rate not in {100, 500}:
        result.errors.append(f"Unsupported sampling rate: {sampling_rate}")
        return result

    path = record_path or _resolve_record_path(ecg_id, sampling_rate, metadata_row)
    result.source_checksum, result.source_paths = _source_checksum(path)
    try:
        record = wfdb.rdrecord(path)
    except FileNotFoundError:
        result.errors.append(f"WFDB file not found: {path}")
        return result
    except Exception as exc:
        result.errors.append(f"WFDB read error: {exc}")
        return result

    actual_fs = int(round(float(record.fs)))
    result.sampling_rate = actual_fs
    if actual_fs != sampling_rate:
        result.errors.append(f"Sampling rate mismatch: expected {sampling_rate}, got {actual_fs}")
        return result
    if int(record.n_sig) != NUM_LEADS:
        result.errors.append(f"Expected {NUM_LEADS} leads, got {record.n_sig}")
        return result

    raw_names = list(record.sig_name)
    result.original_lead_order = raw_names
    canonical, unknown = _canonicalize_lead_names(raw_names)
    if unknown:
        result.errors.append(f"Unknown lead names: {unknown}")
        return result
    if len(set(canonical)) != NUM_LEADS:
        result.errors.append(f"Duplicate lead names after canonicalization: {canonical}")
        return result
    reorder = _compute_reorder_indices(canonical, CANONICAL_LEAD_ORDER)
    if reorder is None:
        result.errors.append("Canonical lead set does not match required 12-lead set")
        return result

    raw = record.p_signal
    if raw is None or raw.ndim != 2:
        result.errors.append("WFDB signal is missing or not two-dimensional")
        return result
    result.original_shape = tuple(int(v) for v in raw.shape)
    result.n_samples = int(raw.shape[0])
    result.duration_seconds = result.n_samples / actual_fs
    if raw.shape[1] != NUM_LEADS:
        result.errors.append(f"Signal shape lead mismatch: {raw.shape}")
        return result

    expected = SAMPLES_100HZ if sampling_rate == 100 else SAMPLES_500HZ
    max_adjust = max(1, int(round(QC.max_length_adjustment_ms * sampling_rate / 1000.0)))
    delta = int(raw.shape[0]) - expected
    if abs(delta) > max_adjust:
        result.errors.append(
            f"Sample count mismatch exceeds {QC.max_length_adjustment_ms:g} ms tolerance: "
            f"expected {expected}, got {raw.shape[0]}"
        )
        return result

    valid = np.isfinite(raw)
    result.n_nonfinite = int((~valid).sum())
    signal = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    if delta > 0:
        signal = signal[:expected]
        valid = valid[:expected]
        result.transformations.append(f"cropped_tail_samples:{delta}")
    elif delta < 0:
        pad = -delta
        signal = np.pad(signal, ((0, pad), (0, 0)), constant_values=0.0)
        valid = np.pad(valid, ((0, pad), (0, 0)), constant_values=False)
        result.transformations.append(f"padded_tail_samples:{pad}")
    if result.n_nonfinite:
        result.transformations.append(f"nonfinite_replaced:{result.n_nonfinite}")

    signal = signal[:, reorder]
    valid = valid[:, reorder]
    result.lead_reorder_applied = reorder != list(range(NUM_LEADS))
    if result.lead_reorder_applied:
        result.transformations.append("canonical_lead_reorder")

    result.signal_mv = signal.T.astype(np.float32)
    result.sample_mask = valid.T.astype(bool)
    result.lead_mask = result.sample_mask.any(axis=1)
    result.lead_order = list(CANONICAL_LEAD_ORDER)
    result.canonical_lead_order = list(CANONICAL_LEAD_ORDER)
    result.physical_units = "mV"
    result.gain_applied = True
    result.is_valid = True
    return result


def validate_batch(
    ecg_ids: List[int],
    sampling_rate: int = 100,
    max_errors: int = 10,
) -> Tuple[Dict[int, StructuralValidation], Dict[str, int]]:
    results: Dict[int, StructuralValidation] = {}
    invalid = warnings = 0
    for ecg_id in ecg_ids:
        result = validate_record(ecg_id, sampling_rate)
        results[int(ecg_id)] = result
        invalid += int(not result.is_valid)
        warnings += int(bool(result.warnings or result.transformations))
        if invalid >= max_errors:
            break
    return results, {"total": len(results), "valid": len(results) - invalid, "invalid": invalid, "warnings": warnings}
