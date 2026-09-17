"""
Morphology-preservation utility gate.

Implements plan phase 9:
- Compare corrected signal (View B) against minimal reference (View A)
- Measure R-peak shift, QRS width change, ST-level deviation, T-wave polarity
- Compute U_P = artifact_reduction − λ_m·distortion − λ_f·failure − λ_t·time
- Reject correction if U_P < 0 → fall back to View A
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from .config import MORPHOLOGY, NUM_LEADS, CANONICAL_LEAD_ORDER

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Morphology measurement results
# ---------------------------------------------------------------------------

@dataclass
class MorphologyMetrics:
    """Per-lead morphology comparison between View A and View B."""
    lead_name: str
    rpeak_shift_samples: float = 0.0
    qrs_width_change_ms: float = 0.0
    qrs_amplitude_change_mv: float = 0.0
    st_level_shift_mv: float = 0.0
    twave_polarity_preserved: bool = True
    passed: bool = True
    violations: List[str] = field(default_factory=list)


@dataclass
class GateResult:
    """Result of the morphology-preservation gate for a record."""
    ecg_id: int
    gate_passed: bool = True
    utility_score: float = 0.0
    artifact_reduction: float = 0.0
    morphology_distortion: float = 0.0
    per_lead_metrics: dict = field(default_factory=dict)  # lead → MorphologyMetrics
    processing_time_ms: float = 0.0
    reason: str = ""


# ---------------------------------------------------------------------------
# R-peak detection (simple amplitude-based for QC purposes)
# ---------------------------------------------------------------------------

def _detect_r_peaks(
    lead_signal: np.ndarray,
    fs: int = 100,
) -> np.ndarray:
    """Detect R-peak locations in a single lead.

    Uses scipy's find_peaks with physiological constraints.
    """
    # Minimum distance between peaks: ~200 ms (300 BPM max)
    min_distance = int(0.2 * fs)

    # Height threshold: use adaptive threshold based on signal
    signal_std = np.std(lead_signal)
    height_threshold = 0.5 * signal_std  # half a standard deviation

    peaks, _ = find_peaks(
        lead_signal,
        distance=min_distance,
        height=height_threshold,
    )

    return peaks


def _measure_qrs_width(
    lead_signal: np.ndarray,
    peak_idx: int,
    fs: int = 100,
) -> float:
    """Estimate QRS width around a detected R-peak.

    Returns width in milliseconds.
    """
    # Search window: ±100 ms around the peak
    window_samples = int(0.1 * fs)
    start = max(0, peak_idx - window_samples)
    end = min(len(lead_signal), peak_idx + window_samples)

    segment = lead_signal[start:end]
    peak_in_segment = peak_idx - start

    # Find QRS onset and offset as the nearest local minima
    # Left of peak
    left_segment = segment[:peak_in_segment]
    if len(left_segment) > 0:
        left_min_idx = np.argmin(np.abs(left_segment - np.mean(left_segment)))
        qrs_onset = start + left_min_idx
    else:
        qrs_onset = start

    # Right of peak
    right_segment = segment[peak_in_segment:]
    if len(right_segment) > 0:
        right_min_idx = np.argmin(np.abs(right_segment - np.mean(right_segment)))
        qrs_offset = start + peak_in_segment + right_min_idx
    else:
        qrs_offset = end

    width_samples = qrs_offset - qrs_onset
    width_ms = width_samples / fs * 1000.0

    return width_ms


def _measure_st_level(
    lead_signal: np.ndarray,
    peak_idx: int,
    fs: int = 100,
) -> float:
    """Measure ST-segment level relative to baseline.

    Samples the signal at J-point + 60 ms after the R-peak.
    """
    # J-point approximation: ~80 ms after R-peak
    j_point = peak_idx + int(0.08 * fs)
    # ST measurement point: J + 60 ms
    st_point = j_point + int(0.06 * fs)

    if st_point >= len(lead_signal):
        return 0.0

    # ST level = average of a small window around the ST point
    window = int(0.02 * fs)  # 20 ms window
    st_start = max(0, st_point - window)
    st_end = min(len(lead_signal), st_point + window)

    return float(np.mean(lead_signal[st_start:st_end]))


def _measure_twave_polarity(
    lead_signal: np.ndarray,
    peak_idx: int,
    fs: int = 100,
) -> float:
    """Determine T-wave polarity (positive value = upright).

    Looks at the signal ~200-350 ms after the R-peak.
    """
    t_start = peak_idx + int(0.20 * fs)
    t_end = peak_idx + int(0.35 * fs)

    if t_end >= len(lead_signal):
        return 0.0

    t_segment = lead_signal[t_start:t_end]
    return float(np.mean(t_segment))


# ---------------------------------------------------------------------------
# Per-lead morphology comparison
# ---------------------------------------------------------------------------

def compare_lead_morphology(
    minimal: np.ndarray,
    corrected: np.ndarray,
    lead_name: str,
    fs: int = 100,
) -> MorphologyMetrics:
    """Compare morphological features between View A and View B for one lead.

    Parameters
    ----------
    minimal : np.ndarray
        View A (minimal) single-lead signal.
    corrected : np.ndarray
        View B (corrected) single-lead signal.
    lead_name : str
        Lead name for reporting.
    fs : int
        Sampling rate.

    Returns
    -------
    MorphologyMetrics
        Comparison results with pass/fail status.
    """
    metrics = MorphologyMetrics(lead_name=lead_name)

    # --- 1. R-peak location shift ---
    peaks_a = _detect_r_peaks(minimal, fs)
    peaks_b = _detect_r_peaks(corrected, fs)

    if len(peaks_a) > 0 and len(peaks_b) > 0:
        # Match nearest peaks
        n_match = min(len(peaks_a), len(peaks_b))
        shifts = []
        for i in range(n_match):
            # Find closest peak in B for each peak in A
            diffs = np.abs(peaks_b - peaks_a[i])
            min_shift = np.min(diffs)
            shifts.append(min_shift)

        max_shift = max(shifts) if shifts else 0
        metrics.rpeak_shift_samples = float(max_shift)

        if max_shift > MORPHOLOGY.max_rpeak_shift_samples:
            metrics.passed = False
            metrics.violations.append(
                f"R-peak shifted by {max_shift} samples "
                f"(max: {MORPHOLOGY.max_rpeak_shift_samples})"
            )

        # --- 2. QRS width change ---
        if len(peaks_a) > 0 and len(peaks_b) > 0:
            width_a = _measure_qrs_width(minimal, peaks_a[0], fs)
            width_b = _measure_qrs_width(corrected, peaks_b[0], fs)
            width_change = abs(width_b - width_a)
            metrics.qrs_width_change_ms = width_change

            if width_change > MORPHOLOGY.max_qrs_width_change_ms:
                metrics.passed = False
                metrics.violations.append(
                    f"QRS width changed by {width_change:.1f} ms "
                    f"(max: {MORPHOLOGY.max_qrs_width_change_ms} ms)"
                )

        # --- 3. QRS amplitude change ---
        amp_a = float(np.max(minimal[peaks_a[0] - 2:peaks_a[0] + 3])
                       if peaks_a[0] >= 2 else np.max(minimal))
        amp_b = float(np.max(corrected[peaks_b[0] - 2:peaks_b[0] + 3])
                       if peaks_b[0] >= 2 else np.max(corrected))
        amp_change = abs(amp_b - amp_a)
        metrics.qrs_amplitude_change_mv = amp_change

        if amp_change > MORPHOLOGY.max_qrs_amplitude_change_mv:
            metrics.passed = False
            metrics.violations.append(
                f"QRS amplitude changed by {amp_change:.3f} mV "
                f"(max: {MORPHOLOGY.max_qrs_amplitude_change_mv} mV)"
            )

        # --- 4. ST-level shift ---
        st_a = _measure_st_level(minimal, peaks_a[0], fs)
        st_b = _measure_st_level(corrected, peaks_b[0], fs)
        st_shift = abs(st_b - st_a)
        metrics.st_level_shift_mv = st_shift

        if st_shift > MORPHOLOGY.max_st_level_shift_mv:
            metrics.passed = False
            metrics.violations.append(
                f"ST-level shifted by {st_shift:.4f} mV "
                f"(max: {MORPHOLOGY.max_st_level_shift_mv} mV)"
            )

        # --- 5. T-wave polarity ---
        twave_a = _measure_twave_polarity(minimal, peaks_a[0], fs)
        twave_b = _measure_twave_polarity(corrected, peaks_b[0], fs)

        if MORPHOLOGY.twave_polarity_must_match:
            polarity_preserved = (twave_a * twave_b) >= 0  # same sign
            metrics.twave_polarity_preserved = polarity_preserved

            if not polarity_preserved:
                metrics.passed = False
                metrics.violations.append(
                    f"T-wave polarity inverted "
                    f"(A: {twave_a:.4f}, B: {twave_b:.4f})"
                )

    else:
        # Cannot detect peaks: skip morphology checks but flag it
        metrics.violations.append(
            f"Could not detect R-peaks (A: {len(peaks_a)}, B: {len(peaks_b)})"
        )

    return metrics


# ---------------------------------------------------------------------------
# Full morphology gate
# ---------------------------------------------------------------------------

def evaluate_gate(
    signal_minimal: np.ndarray,
    signal_corrected: np.ndarray,
    ecg_id: int,
    fs: int = 100,
) -> GateResult:
    """Evaluate the morphology-preservation gate for a complete 12-lead ECG.

    If the gate fails (U_P < 0), the correction should be rejected
    and the system falls back to View A (minimal).

    Parameters
    ----------
    signal_minimal : np.ndarray
        View A, float32[12, samples].
    signal_corrected : np.ndarray
        View B, float32[12, samples].
    ecg_id : int
        Record identifier.
    fs : int
        Sampling rate.

    Returns
    -------
    GateResult
        Contains the utility score and per-lead metrics.
    """
    start_time = time.time()
    gate = GateResult(ecg_id=ecg_id)

    # If signals are identical, gate trivially passes
    if np.allclose(signal_minimal, signal_corrected, atol=1e-7):
        gate.gate_passed = True
        gate.utility_score = 0.0
        gate.reason = "No correction applied (View B == View A)"
        gate.processing_time_ms = (time.time() - start_time) * 1000
        return gate

    # --- Per-lead morphology comparison ---
    total_distortion = 0.0
    n_violations = 0

    for i, lead_name in enumerate(CANONICAL_LEAD_ORDER):
        metrics = compare_lead_morphology(
            signal_minimal[i],
            signal_corrected[i],
            lead_name,
            fs,
        )
        gate.per_lead_metrics[lead_name] = metrics

        if not metrics.passed:
            n_violations += 1
            total_distortion += (
                metrics.st_level_shift_mv
                + metrics.rpeak_shift_samples * 0.1
                + metrics.qrs_width_change_ms * 0.01
                + metrics.qrs_amplitude_change_mv
            )

    gate.morphology_distortion = total_distortion

    # --- Artifact reduction estimate ---
    # Compare noise levels before and after correction
    diff = signal_corrected - signal_minimal
    gate.artifact_reduction = float(np.std(diff))

    # --- Processing time ---
    elapsed_ms = (time.time() - start_time) * 1000
    gate.processing_time_ms = elapsed_ms

    # --- Compute utility score ---
    # U_P = artifact_reduction − λ_m·distortion − λ_f·failure_rate − λ_t·time
    failure_rate = n_violations / max(NUM_LEADS, 1)
    gate.utility_score = (
        gate.artifact_reduction
        - MORPHOLOGY.lambda_m * total_distortion
        - MORPHOLOGY.lambda_f * failure_rate
        - MORPHOLOGY.lambda_t * (elapsed_ms / 1000.0)
    )

    # --- Gate decision ---
    if gate.utility_score < 0 or n_violations > 0:
        gate.gate_passed = False
        gate.reason = (
            f"REJECTED: U_P={gate.utility_score:.4f}, "
            f"{n_violations} lead(s) with morphology violations"
        )
        logger.warning(
            "ecg_id %d: Morphology gate REJECTED correction — %s",
            ecg_id, gate.reason,
        )
    else:
        gate.gate_passed = True
        gate.reason = f"ACCEPTED: U_P={gate.utility_score:.4f}"

    return gate
