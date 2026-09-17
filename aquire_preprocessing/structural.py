"""
WFDB structural validation for 12-lead ECG records.

Implements plan phase 2:
- .hea/.dat file pairing
- Lead count, name, and order validation
- Sampling rate and duration verification
- Physical-unit conversion (ADC → mV)
- Canonical lead reordering
- NaN/Inf detection
- Shape validation → float32[12, 1000]
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import wfdb

from .config import (
    CANONICAL_LEAD_ORDER,
    LEAD_NAME_ALIASES,
    NUM_LEADS,
    SAMPLES_100HZ,
    SAMPLES_500HZ,
    SAMPLING_RATE_100HZ,
    SAMPLING_RATE_500HZ,
    TARGET_SHAPE_100HZ,
    PATHS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation result container
# ---------------------------------------------------------------------------

@dataclass
class StructuralValidation:
    """Result of structural validation for a single ECG record."""

    ecg_id: int
    is_valid: bool = True
    signal: Optional[np.ndarray] = None          # float32[12, samples]
    sampling_rate: int = 0
    original_lead_order: List[str] = field(default_factory=list)
    canonical_lead_order: List[str] = field(default_factory=list)
    lead_reorder_applied: bool = False
    physical_units: str = "mV"
    gain_applied: bool = False
    duration_seconds: float = 0.0
    n_samples: int = 0
    n_nonfinite: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_checksum: str = ""


# ---------------------------------------------------------------------------
# Core validation functions
# ---------------------------------------------------------------------------

def _resolve_record_path(ecg_id: int, sampling_rate: int = 100) -> str:
    """Resolve the WFDB record path for an ecg_id.

    PTB-XL stores records in subdirectories grouped by ecg_id ranges:
        records100/00000/00001_lr
        records100/21000/21001_lr
    """
    folder = f"{(ecg_id // 1000) * 1000:05d}"
    suffix = "lr" if sampling_rate == 100 else "hr"
    record_name = f"{ecg_id:05d}_{suffix}"

    if sampling_rate == 100:
        base_dir = PATHS.records100_dir
    else:
        base_dir = PATHS.records500_dir

    return str(base_dir / folder / record_name)


def _canonicalize_lead_names(
    lead_names: List[str],
) -> Tuple[List[str], List[str]]:
    """Map raw WFDB lead names to canonical names.

    Returns
    -------
    canonical_names : list of str
        Canonical lead names (e.g., ['I', 'II', ...]).
    unknown : list of str
        Any lead names that could not be resolved.
    """
    canonical = []
    unknown = []
    for name in lead_names:
        clean = name.strip()
        resolved = LEAD_NAME_ALIASES.get(clean)
        if resolved is None:
            unknown.append(clean)
            canonical.append(clean)  # keep original for error reporting
        else:
            canonical.append(resolved)
    return canonical, unknown


def _compute_reorder_indices(
    current_order: List[str],
    target_order: List[str],
) -> Optional[List[int]]:
    """Compute the index permutation to reorder leads.

    Returns None if the sets don't match.
    """
    if set(current_order) != set(target_order):
        return None
    return [current_order.index(lead) for lead in target_order]


def validate_record(
    ecg_id: int,
    sampling_rate: int = 100,
    record_path: Optional[str] = None,
) -> StructuralValidation:
    """Perform full structural validation on a single ECG record.

    Parameters
    ----------
    ecg_id : int
        The ecg_id from ptbxl_database.csv.
    sampling_rate : int
        Target sampling rate (100 or 500).
    record_path : str, optional
        Override the auto-resolved WFDB record path.

    Returns
    -------
    StructuralValidation
        Contains the validated signal tensor or error details.
    """
    result = StructuralValidation(ecg_id=ecg_id)

    # --- 1. Resolve and load WFDB record ---
    path = record_path or _resolve_record_path(ecg_id, sampling_rate)

    try:
        record = wfdb.rdrecord(path)
    except FileNotFoundError:
        result.is_valid = False
        result.errors.append(f"WFDB file not found: {path}")
        return result
    except Exception as e:
        result.is_valid = False
        result.errors.append(f"WFDB read error: {e}")
        return result

    # --- 2. Validate sampling rate ---
    actual_fs = record.fs
    result.sampling_rate = actual_fs

    expected_samples = SAMPLES_100HZ if sampling_rate == 100 else SAMPLES_500HZ
    if actual_fs != sampling_rate:
        result.warnings.append(
            f"Sampling rate mismatch: expected {sampling_rate} Hz, got {actual_fs} Hz"
        )

    # --- 3. Validate lead count ---
    n_sig = record.n_sig
    if n_sig != NUM_LEADS:
        result.is_valid = False
        result.errors.append(
            f"Expected {NUM_LEADS} leads, got {n_sig}"
        )
        return result

    # --- 4. Validate and canonicalize lead names ---
    raw_lead_names = list(record.sig_name)
    result.original_lead_order = raw_lead_names

    canonical_names, unknown = _canonicalize_lead_names(raw_lead_names)
    if unknown:
        result.is_valid = False
        result.errors.append(
            f"Unknown lead names (cannot canonicalize): {unknown}"
        )
        return result

    # Check for duplicate canonical names
    if len(set(canonical_names)) != NUM_LEADS:
        result.is_valid = False
        result.errors.append(
            f"Duplicate lead names after canonicalization: {canonical_names}"
        )
        return result

    # --- 5. Extract signal and convert to physical units ---
    # wfdb.rdrecord with physical=True already converts to mV
    # record.p_signal is shape (n_samples, n_leads) in physical units
    signal = record.p_signal  # (samples, leads) in mV

    if signal is None:
        result.is_valid = False
        result.errors.append("Signal data is None after WFDB read")
        return result

    # --- 6. Validate shape ---
    n_samples, n_leads = signal.shape
    result.n_samples = n_samples
    result.duration_seconds = n_samples / actual_fs

    if n_leads != NUM_LEADS:
        result.is_valid = False
        result.errors.append(
            f"Signal shape lead mismatch: {signal.shape}"
        )
        return result

    if n_samples != expected_samples:
        result.warnings.append(
            f"Sample count: expected {expected_samples}, got {n_samples}"
        )
        # Truncate or pad if close
        if n_samples > expected_samples:
            signal = signal[:expected_samples, :]
            result.warnings.append(f"Truncated to {expected_samples} samples")
        elif n_samples < expected_samples:
            pad_width = expected_samples - n_samples
            signal = np.pad(
                signal, ((0, pad_width), (0, 0)),
                mode="constant", constant_values=0.0,
            )
            result.warnings.append(
                f"Padded {pad_width} samples with zeros"
            )
        n_samples = expected_samples

    # --- 7. Check for nonfinite values ---
    nonfinite_mask = ~np.isfinite(signal)
    n_nonfinite = int(nonfinite_mask.sum())
    result.n_nonfinite = n_nonfinite

    if n_nonfinite > 0:
        result.warnings.append(
            f"Found {n_nonfinite} non-finite samples "
            f"({n_nonfinite / signal.size * 100:.2f}%)"
        )
        # Replace NaN/Inf with 0.0 and record in sample_mask later
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    # --- 8. Reorder leads to canonical order ---
    reorder_idx = _compute_reorder_indices(canonical_names, CANONICAL_LEAD_ORDER)
    if reorder_idx is None:
        result.is_valid = False
        result.errors.append(
            f"Cannot reorder leads: canonical set {set(canonical_names)} "
            f"does not match target {set(CANONICAL_LEAD_ORDER)}"
        )
        return result

    if reorder_idx != list(range(NUM_LEADS)):
        signal = signal[:, reorder_idx]
        result.lead_reorder_applied = True
        result.warnings.append(
            f"Leads reordered from {canonical_names} to {CANONICAL_LEAD_ORDER}"
        )

    result.canonical_lead_order = list(CANONICAL_LEAD_ORDER)

    # --- 9. Transpose to (leads, samples) and cast to float32 ---
    signal = signal.T.astype(np.float32)  # (12, 1000) or (12, 5000)

    assert signal.shape == (NUM_LEADS, n_samples), (
        f"Final shape {signal.shape} != expected ({NUM_LEADS}, {n_samples})"
    )

    result.signal = signal
    result.physical_units = "mV"
    result.gain_applied = True

    return result


def validate_batch(
    ecg_ids: List[int],
    sampling_rate: int = 100,
    max_errors: int = 10,
) -> Tuple[Dict[int, StructuralValidation], Dict[str, int]]:
    """Validate a batch of ECG records.

    Returns
    -------
    results : dict
        ecg_id → StructuralValidation
    summary : dict
        Counts of valid, invalid, warnings.
    """
    results = {}
    n_valid = 0
    n_invalid = 0
    n_warnings = 0
    error_count = 0

    for ecg_id in ecg_ids:
        result = validate_record(ecg_id, sampling_rate)
        results[ecg_id] = result

        if result.is_valid:
            n_valid += 1
            if result.warnings:
                n_warnings += 1
        else:
            n_invalid += 1
            error_count += 1
            if error_count <= max_errors:
                logger.error(
                    "ecg_id %d INVALID: %s",
                    ecg_id, "; ".join(result.errors),
                )

    summary = {
        "total": len(ecg_ids),
        "valid": n_valid,
        "invalid": n_invalid,
        "with_warnings": n_warnings,
    }

    logger.info(
        "Batch validation: %d total, %d valid, %d invalid, %d with warnings",
        summary["total"], summary["valid"],
        summary["invalid"], summary["with_warnings"],
    )

    return results, summary
