"""Anti-aliased high-resolution conversion and morphology validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import resample_poly

from .contracts import GateState
from .morphology_gate import GateResult, evaluate_gate


@dataclass
class ResamplingResult:
    signal_100hz: np.ndarray
    sample_mask_100hz: np.ndarray
    lead_mask: np.ndarray
    gate: Optional[GateResult]
    accepted: bool
    method: str = "scipy-resample-poly-kaiser"


def resample_500_to_100(
    signal_500hz: np.ndarray,
    sample_mask: Optional[np.ndarray] = None,
    lead_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a 10 s 500 Hz ECG to 100 Hz using a polyphase anti-alias filter.

    A 100 Hz output sample is marked valid only when all five contributing
    500 Hz samples were valid. This intentionally avoids inventing provenance.
    """
    signal = np.asarray(signal_500hz, dtype=np.float32)
    if signal.shape != (12, 5000):
        raise ValueError(f"Expected (12, 5000), got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError("Resampling input must be finite; preserve missingness in sample_mask")
    mask = np.ones_like(signal, dtype=bool) if sample_mask is None else np.asarray(sample_mask, dtype=bool)
    if mask.shape != signal.shape:
        raise ValueError("sample_mask shape differs from signal")
    leads = mask.any(axis=1) if lead_mask is None else np.asarray(lead_mask, dtype=bool)
    if leads.shape != (12,):
        raise ValueError("lead_mask must have shape (12,)")

    # resample_poly includes the required low-pass anti-alias filter before
    # decimation. Kaiser beta 8.6 gives strong stop-band attenuation.
    output = resample_poly(signal, up=1, down=5, axis=1, window=("kaiser", 8.6)).astype(np.float32)
    grouped_mask = mask.reshape(12, 1000, 5).all(axis=2)
    grouped_mask &= leads[:, None]
    return output, grouped_mask, leads


def resample_corrected_pair(
    minimal_500hz: np.ndarray,
    corrected_500hz: np.ndarray,
    ecg_id: int,
    sample_mask: Optional[np.ndarray] = None,
    lead_mask: Optional[np.ndarray] = None,
) -> ResamplingResult:
    """Resample both views and accept the corrected view only after its gate."""
    minimal, mask, leads = resample_500_to_100(minimal_500hz, sample_mask, lead_mask)
    corrected, corrected_mask, corrected_leads = resample_500_to_100(
        corrected_500hz, sample_mask, lead_mask
    )
    mask &= corrected_mask
    leads &= corrected_leads
    gate = evaluate_gate(minimal, corrected, int(ecg_id), fs=100, lead_mask=leads)
    accepted = gate.state == GateState.PASS.value
    return ResamplingResult(corrected if accepted else minimal, mask, leads, gate, accepted)
