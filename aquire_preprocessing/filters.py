"""
Morphology-preserving signal filters.

Implements plan phase 7:
- Baseline wander correction (zero-phase, preserves ST segment)
- Powerline interference removal (50/60 Hz notch)
- Short-gap interpolation with sample mask
- All filters are zero-phase (scipy filtfilt) to avoid phase distortion
- No universal aggressive high-pass filter
"""

import logging
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

from .config import QC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline wander correction
# ---------------------------------------------------------------------------

def correct_baseline_wander(
    lead_signal: np.ndarray,
    fs: int = 100,
    cutoff_hz: float = 0.05,
    order: int = 2,
) -> np.ndarray:
    """Remove baseline wander using a zero-phase high-pass filter.

    Uses an extremely conservative cutoff (0.05 Hz) — well below
    the 0.5 Hz danger zone cited by AHA/ACCF/HRS for ST-segment
    distortion. This removes only very slow electrode drift while
    fully preserving repolarization morphology.

    Parameters
    ----------
    lead_signal : np.ndarray
        Single-lead signal, shape (samples,).
    fs : int
        Sampling rate in Hz.
    cutoff_hz : float
        High-pass cutoff frequency in Hz. Default 0.05 Hz.
    order : int
        Butterworth filter order. Default 2.

    Returns
    -------
    np.ndarray
        Baseline-corrected signal, same shape as input.
    """
    # Nyquist frequency
    nyquist = fs / 2.0

    if cutoff_hz >= nyquist:
        logger.warning(
            "Cutoff %.3f Hz >= Nyquist %.1f Hz, skipping baseline correction",
            cutoff_hz, nyquist,
        )
        return lead_signal.copy()

    # Design zero-phase Butterworth high-pass
    b, a = butter(order, cutoff_hz / nyquist, btype="high")

    # Apply zero-phase filtering (no phase distortion)
    try:
        corrected = filtfilt(b, a, lead_signal, padtype="odd", padlen=3 * max(len(a), len(b)))
    except ValueError:
        # Signal too short for filter, return unchanged
        logger.warning("Signal too short for baseline correction, skipping")
        return lead_signal.copy()

    return corrected.astype(np.float32)


def correct_baseline_wander_median(
    lead_signal: np.ndarray,
    fs: int = 100,
    window_seconds: float = 0.6,
) -> np.ndarray:
    """Alternative baseline correction using median filter subtraction.

    This method estimates the baseline using two cascaded median filters
    (600 ms and 1200 ms windows), following the approach used in
    several clinical ECG processing papers.

    Parameters
    ----------
    lead_signal : np.ndarray
        Single-lead signal, shape (samples,).
    fs : int
        Sampling rate in Hz.
    window_seconds : float
        First median filter window in seconds.

    Returns
    -------
    np.ndarray
        Baseline-corrected signal.
    """
    from scipy.ndimage import median_filter

    win1 = int(window_seconds * fs)
    win2 = int(window_seconds * 2 * fs)

    # Ensure odd window sizes
    win1 = win1 if win1 % 2 == 1 else win1 + 1
    win2 = win2 if win2 % 2 == 1 else win2 + 1

    # Two-pass median filter to estimate baseline
    baseline = median_filter(lead_signal, size=win1)
    baseline = median_filter(baseline, size=win2)

    corrected = lead_signal - baseline
    return corrected.astype(np.float32)


# ---------------------------------------------------------------------------
# Powerline interference removal
# ---------------------------------------------------------------------------

def remove_powerline(
    lead_signal: np.ndarray,
    fs: int = 100,
    freq_hz: float = 50.0,
    quality_factor: float = 30.0,
) -> np.ndarray:
    """Remove powerline interference using a zero-phase IIR notch filter.

    Parameters
    ----------
    lead_signal : np.ndarray
        Single-lead signal, shape (samples,).
    fs : int
        Sampling rate in Hz.
    freq_hz : float
        Power-line frequency (50 Hz or 60 Hz).
    quality_factor : float
        Q factor of the notch filter. Higher = narrower notch.

    Returns
    -------
    np.ndarray
        Signal with powerline interference removed.
    """
    nyquist = fs / 2.0

    if freq_hz >= nyquist:
        # Cannot notch a frequency above Nyquist
        logger.debug(
            "Powerline freq %.0f Hz >= Nyquist %.0f Hz, skipping",
            freq_hz, nyquist,
        )
        return lead_signal.copy()

    # Design notch filter
    b, a = iirnotch(freq_hz, quality_factor, fs)

    # Apply zero-phase filtering
    try:
        filtered = filtfilt(b, a, lead_signal)
    except ValueError:
        logger.warning("Signal too short for notch filter, skipping")
        return lead_signal.copy()

    return filtered.astype(np.float32)


# ---------------------------------------------------------------------------
# Short-gap interpolation
# ---------------------------------------------------------------------------

def repair_short_gaps(
    lead_signal: np.ndarray,
    sample_mask: np.ndarray,
    fs: int = 100,
    max_gap_ms: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate short missing-sample gaps.

    Gaps longer than max_gap_ms are left unrepaired and retained
    in the sample_mask as False.

    Parameters
    ----------
    lead_signal : np.ndarray
        Single-lead signal, shape (samples,).
    sample_mask : np.ndarray
        Boolean mask, True = valid sample, shape (samples,).
    fs : int
        Sampling rate.
    max_gap_ms : float
        Maximum interpolatable gap in milliseconds. Default from config.

    Returns
    -------
    repaired_signal : np.ndarray
        Signal with short gaps filled.
    updated_mask : np.ndarray
        Updated mask (repaired samples set to True).
    """
    if max_gap_ms is None:
        max_gap_ms = QC.max_interpolatable_gap_ms

    max_gap_samples = int(max_gap_ms / 1000.0 * fs)
    repaired = lead_signal.copy()
    mask = sample_mask.copy()

    # Find contiguous missing regions
    missing = ~mask
    if not np.any(missing):
        return repaired, mask

    # Identify gap boundaries
    diff = np.diff(missing.astype(int))
    gap_starts = np.where(diff == 1)[0] + 1
    gap_ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if missing[0]:
        gap_starts = np.concatenate(([0], gap_starts))
    if missing[-1]:
        gap_ends = np.concatenate((gap_ends, [len(missing)]))

    for start, end in zip(gap_starts, gap_ends):
        gap_len = end - start

        if gap_len <= max_gap_samples:
            # Short gap: linear interpolation
            left_val = repaired[start - 1] if start > 0 else 0.0
            right_val = repaired[end] if end < len(repaired) else 0.0

            interp_vals = np.linspace(left_val, right_val, gap_len + 2)[1:-1]
            repaired[start:end] = interp_vals
            mask[start:end] = True

            logger.debug(
                "Repaired gap at samples %d-%d (%d samples)",
                start, end, gap_len,
            )
        else:
            # Long gap: leave unrepaired, mask stays False
            logger.debug(
                "Gap at samples %d-%d (%d samples) exceeds max (%d), not repaired",
                start, end, gap_len, max_gap_samples,
            )

    return repaired.astype(np.float32), mask


# ---------------------------------------------------------------------------
# Convenience: apply all indicated corrections to a lead
# ---------------------------------------------------------------------------

def apply_corrections(
    lead_signal: np.ndarray,
    actions: list,
    fs: int = 100,
    sample_mask: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the router-indicated corrections to a single lead.

    Parameters
    ----------
    lead_signal : np.ndarray
        Single-lead signal (samples,).
    actions : list of str
        Action names from router (e.g., ['baseline_correction', 'powerline_removal']).
    fs : int
        Sampling rate.
    sample_mask : np.ndarray
        Boolean mask for the lead.

    Returns
    -------
    corrected : np.ndarray
        Corrected signal.
    updated_mask : np.ndarray
        Updated sample mask.
    """
    corrected = lead_signal.copy()
    if sample_mask is None:
        sample_mask = np.ones(len(lead_signal), dtype=bool)
    mask = sample_mask.copy()

    for action in actions:
        if action == "baseline_correction":
            corrected = correct_baseline_wander(corrected, fs)
        elif action == "powerline_removal":
            corrected = remove_powerline(corrected, fs)
        elif action == "short_gap_repair":
            corrected, mask = repair_short_gaps(corrected, mask, fs)
        elif action in ("minimal", "lead_mask", "fail"):
            pass  # No correction needed
        else:
            logger.warning("Unknown action '%s', skipping", action)

    return corrected, mask
