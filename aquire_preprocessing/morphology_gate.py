"""Beat-matched, morphology-preservation gate for offline corrections."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, welch

from .config import CANONICAL_LEAD_ORDER, MORPHOLOGY, NUM_LEADS
from .contracts import GateState


@dataclass
class MorphologyMetrics:
    lead_name: str
    rpeak_shift_samples: float = 0.0
    qrs_width_change_ms: float = 0.0
    qrs_amplitude_change_mv: float = 0.0
    st_level_shift_mv: float = 0.0
    twave_polarity_preserved: bool = True
    waveform_correlation_min: float = 1.0
    beat_count_a: int = 0
    beat_count_b: int = 0
    matched_beats: int = 0
    state: str = GateState.PASS.value
    passed: bool = True
    violations: List[str] = field(default_factory=list)


@dataclass
class GateResult:
    ecg_id: int
    state: str = GateState.PASS.value
    gate_passed: bool = True
    utility_score: float = 0.0
    artifact_reduction: float = 0.0
    morphology_distortion: float = 0.0
    per_lead_metrics: Dict[str, MorphologyMetrics] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    reason: str = ""


def _robust_scale(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return max(1.4826 * mad, float(np.std(x)) * 0.25, 1e-6)


def _detect_r_peaks(lead_signal: np.ndarray, fs: int = 100) -> np.ndarray:
    """Polarity-aware detector used for QC, not a diagnostic delineator."""
    x = np.asarray(lead_signal, dtype=float)
    if x.size < max(20, fs) or not np.all(np.isfinite(x)):
        return np.array([], dtype=int)
    centered = x - np.median(x)
    envelope = np.abs(centered)
    prominence = max(0.8 * _robust_scale(centered), 0.02)
    peaks, props = find_peaks(
        envelope,
        distance=max(1, int(0.25 * fs)),
        prominence=prominence,
    )
    if len(peaks) > 1:
        # Reject small T-wave candidates relative to the robust peak amplitude.
        peak_amp = envelope[peaks]
        threshold = max(np.percentile(peak_amp, 25) * 0.6, prominence)
        peaks = peaks[peak_amp >= threshold]
    return peaks.astype(int)


def _multilead_r_peaks(signal: np.ndarray, lead_mask: np.ndarray, fs: int) -> np.ndarray:
    """Use lead II when reliable, otherwise temporal consensus across leads."""
    lead_ii = CANONICAL_LEAD_ORDER.index("II")
    if lead_mask[lead_ii]:
        preferred = _detect_r_peaks(signal[lead_ii], fs)
        if len(preferred) >= 3:
            return preferred
    detected = [
        _detect_r_peaks(signal[index], fs)
        for index in range(NUM_LEADS) if lead_mask[index]
    ]
    events = sorted((int(peak), lead) for lead, peaks in enumerate(detected) for peak in peaks)
    tolerance = max(1, int(round(0.08 * fs)))
    clusters: List[List[Tuple[int, int]]] = []
    for peak, lead in events:
        if not clusters or peak - int(np.median([p for p, _ in clusters[-1]])) > tolerance:
            clusters.append([(peak, lead)])
        else:
            clusters[-1].append((peak, lead))
    consensus = [
        int(round(np.median([peak for peak, _ in cluster])))
        for cluster in clusters if len({lead for _, lead in cluster}) >= 2
    ]
    return np.asarray(consensus, dtype=int)


def _match_peaks(a: np.ndarray, b: np.ndarray, fs: int) -> List[Tuple[int, int]]:
    tolerance = max(1, int(round(0.10 * fs)))
    available = set(range(len(b)))
    pairs: List[Tuple[int, int]] = []
    for peak_a in a:
        if not available:
            break
        index = min(available, key=lambda j: abs(int(b[j]) - int(peak_a)))
        if abs(int(b[index]) - int(peak_a)) <= tolerance:
            pairs.append((int(peak_a), int(b[index])))
            available.remove(index)
    return pairs


def _qrs_bounds(signal: np.ndarray, peak: int, fs: int) -> Tuple[int, int]:
    """Estimate QRS limits from local derivative energy around a peak."""
    radius = max(2, int(round(0.12 * fs)))
    start, stop = max(0, peak - radius), min(len(signal), peak + radius + 1)
    segment = np.asarray(signal[start:stop], dtype=float)
    if segment.size < 5:
        return start, stop - 1
    derivative = np.abs(np.gradient(segment))
    local_peak = peak - start
    threshold = max(np.percentile(derivative, 35), derivative.max() * 0.08)
    left = local_peak
    while left > 1 and derivative[left] > threshold:
        left -= 1
    right = local_peak
    while right < len(derivative) - 2 and derivative[right] > threshold:
        right += 1
    return start + left, start + right


def _measure_qrs_width(signal: np.ndarray, peak_idx: int, fs: int = 100) -> float:
    onset, offset = _qrs_bounds(signal, peak_idx, fs)
    return (offset - onset) * 1000.0 / fs


def _baseline_level(signal: np.ndarray, onset: int, fs: int) -> float:
    start = max(0, onset - int(round(0.20 * fs)))
    stop = max(start + 1, onset - int(round(0.08 * fs)))
    return float(np.median(signal[start:stop]))


def _measure_st_level(signal: np.ndarray, peak_idx: int, fs: int = 100) -> float:
    onset, offset = _qrs_bounds(signal, peak_idx, fs)
    point = offset + int(round(0.06 * fs))
    half_window = max(1, int(round(0.01 * fs)))
    if point + half_window >= len(signal):
        return float("nan")
    baseline = _baseline_level(signal, onset, fs)
    return float(np.mean(signal[point - half_window:point + half_window + 1]) - baseline)


def _measure_twave_polarity(signal: np.ndarray, peak_idx: int, fs: int = 100) -> float:
    onset, offset = _qrs_bounds(signal, peak_idx, fs)
    start = offset + int(round(0.10 * fs))
    stop = min(len(signal), offset + int(round(0.40 * fs)))
    if stop <= start:
        return float("nan")
    return float(np.mean(signal[start:stop]) - _baseline_level(signal, onset, fs))


def _beat_correlation(a: np.ndarray, b: np.ndarray, pa: int, pb: int, fs: int) -> float:
    pre, post = int(round(0.20 * fs)), int(round(0.40 * fs))
    if pa - pre < 0 or pb - pre < 0 or pa + post >= len(a) or pb + post >= len(b):
        return 1.0
    wa, wb = a[pa - pre:pa + post], b[pb - pre:pb + post]
    if np.std(wa) < 1e-8 or np.std(wb) < 1e-8:
        return 1.0 if np.allclose(wa, wb, atol=1e-6) else 0.0
    return float(np.corrcoef(wa, wb)[0, 1])


def compare_lead_morphology(
    minimal: np.ndarray,
    corrected: np.ndarray,
    lead_name: str,
    fs: int = 100,
    peaks_minimal: Optional[np.ndarray] = None,
    peaks_corrected: Optional[np.ndarray] = None,
) -> MorphologyMetrics:
    metrics = MorphologyMetrics(lead_name=lead_name)
    peaks_a = _detect_r_peaks(minimal, fs) if peaks_minimal is None else np.asarray(peaks_minimal, dtype=int)
    peaks_b = _detect_r_peaks(corrected, fs) if peaks_corrected is None else np.asarray(peaks_corrected, dtype=int)
    metrics.beat_count_a, metrics.beat_count_b = len(peaks_a), len(peaks_b)
    pairs = _match_peaks(peaks_a, peaks_b, fs)
    metrics.matched_beats = len(pairs)
    if not pairs:
        metrics.state = GateState.INDETERMINATE.value
        metrics.passed = False
        metrics.violations.append("No matched QRS complexes")
        return metrics

    shifts, widths, amplitudes, st_shifts, correlations, polarity = [], [], [], [], [], []
    for pa, pb in pairs:
        shifts.append(abs(pb - pa))
        widths.append(abs(_measure_qrs_width(minimal, pa, fs) - _measure_qrs_width(corrected, pb, fs)))
        oa, xa = _qrs_bounds(minimal, pa, fs)
        ob, xb = _qrs_bounds(corrected, pb, fs)
        amp_a = float(np.max(np.abs(minimal[oa:xa + 1] - _baseline_level(minimal, oa, fs))))
        amp_b = float(np.max(np.abs(corrected[ob:xb + 1] - _baseline_level(corrected, ob, fs))))
        amplitudes.append(abs(amp_b - amp_a))
        st_a, st_b = _measure_st_level(minimal, pa, fs), _measure_st_level(corrected, pb, fs)
        if np.isfinite(st_a) and np.isfinite(st_b):
            st_shifts.append(abs(st_b - st_a))
        t_a, t_b = _measure_twave_polarity(minimal, pa, fs), _measure_twave_polarity(corrected, pb, fs)
        if np.isfinite(t_a) and np.isfinite(t_b) and abs(t_a) > 0.01 and abs(t_b) > 0.01:
            polarity.append(np.sign(t_a) == np.sign(t_b))
        correlations.append(_beat_correlation(minimal, corrected, pa, pb, fs))

    metrics.rpeak_shift_samples = float(max(shifts, default=0))
    metrics.qrs_width_change_ms = float(max(widths, default=0.0))
    metrics.qrs_amplitude_change_mv = float(max(amplitudes, default=0.0))
    metrics.st_level_shift_mv = float(max(st_shifts, default=0.0))
    metrics.twave_polarity_preserved = all(polarity) if polarity else True
    metrics.waveform_correlation_min = float(min(correlations, default=1.0))

    max_shift_samples = max(1, int(round(MORPHOLOGY.max_rpeak_shift_ms * fs / 1000.0)))
    checks = [
        (metrics.rpeak_shift_samples > max_shift_samples, f"R-peak shift {metrics.rpeak_shift_samples:.0f} samples"),
        (metrics.qrs_width_change_ms > MORPHOLOGY.max_qrs_width_change_ms, f"QRS width change {metrics.qrs_width_change_ms:.1f} ms"),
        (metrics.qrs_amplitude_change_mv > MORPHOLOGY.max_qrs_amplitude_change_mv, f"QRS amplitude change {metrics.qrs_amplitude_change_mv:.3f} mV"),
        (metrics.st_level_shift_mv > MORPHOLOGY.max_st_level_shift_mv, f"ST shift {metrics.st_level_shift_mv:.3f} mV"),
        (not metrics.twave_polarity_preserved, "T-wave polarity changed"),
        (metrics.waveform_correlation_min < 0.95, f"beat correlation {metrics.waveform_correlation_min:.3f}"),
        (abs(len(peaks_a) - len(peaks_b)) > 1, "beat count changed"),
    ]
    for failed, message in checks:
        if failed:
            metrics.violations.append(message)
    metrics.passed = not metrics.violations
    metrics.state = GateState.PASS.value if metrics.passed else GateState.FAIL.value
    return metrics


def _band_power(signal: np.ndarray, fs: int, low: float, high: float) -> float:
    values = []
    for lead in signal:
        freq, psd = welch(lead, fs=fs, nperseg=min(len(lead), max(128, fs * 2)))
        mask = (freq >= low) & (freq <= high)
        values.append(float(np.trapezoid(psd[mask], freq[mask])) if mask.any() else 0.0)
    return float(np.mean(values))


def _artifact_reduction(a: np.ndarray, b: np.ndarray, fs: int) -> float:
    baseline_before = _band_power(a, fs, 0.01, 0.5)
    baseline_after = _band_power(b, fs, 0.01, 0.5)
    reductions = [baseline_before - baseline_after]
    if fs > 120:
        for mains in (50.0, 60.0):
            if mains + 1 < fs / 2:
                reductions.append(_band_power(a, fs, mains - 1, mains + 1) - _band_power(b, fs, mains - 1, mains + 1))
    return float(sum(reductions))


def evaluate_gate(
    signal_minimal: np.ndarray,
    signal_corrected: np.ndarray,
    ecg_id: int,
    fs: int = 100,
    lead_mask: Optional[np.ndarray] = None,
) -> GateResult:
    start = time.perf_counter()
    gate = GateResult(ecg_id=int(ecg_id))
    if np.allclose(signal_minimal, signal_corrected, atol=1e-7):
        gate.reason = "No correction applied (View B == View A)"
        gate.processing_time_ms = (time.perf_counter() - start) * 1000
        return gate

    if lead_mask is None:
        lead_mask = np.ones(NUM_LEADS, dtype=bool)
    reference_a = _multilead_r_peaks(signal_minimal, np.asarray(lead_mask, bool), fs)
    reference_b = _multilead_r_peaks(signal_corrected, np.asarray(lead_mask, bool), fs)
    distortion, failures, indeterminate = 0.0, 0, 0
    for index, lead_name in enumerate(CANONICAL_LEAD_ORDER):
        if not lead_mask[index]:
            continue
        metrics = compare_lead_morphology(
            signal_minimal[index], signal_corrected[index], lead_name, fs,
            peaks_minimal=reference_a, peaks_corrected=reference_b,
        )
        gate.per_lead_metrics[lead_name] = metrics
        if metrics.state == GateState.INDETERMINATE.value:
            indeterminate += 1
        elif not metrics.passed:
            failures += 1
        distortion += (
            metrics.st_level_shift_mv
            + metrics.qrs_amplitude_change_mv
            + metrics.qrs_width_change_ms / 100.0
            + metrics.rpeak_shift_samples / max(fs, 1)
            + max(0.0, 0.95 - metrics.waveform_correlation_min)
        )

    gate.artifact_reduction = _artifact_reduction(signal_minimal, signal_corrected, fs)
    gate.morphology_distortion = float(distortion)
    evaluated = max(1, len(gate.per_lead_metrics))
    gate.utility_score = (
        gate.artifact_reduction
        - MORPHOLOGY.lambda_m * distortion
        - MORPHOLOGY.lambda_f * ((failures + indeterminate) / evaluated)
    )
    if indeterminate:
        gate.state = GateState.INDETERMINATE.value
        gate.gate_passed = False
        gate.reason = f"INDETERMINATE: {indeterminate} lead(s) could not be validated"
    elif failures or gate.artifact_reduction < -1e-9:
        gate.state = GateState.FAIL.value
        gate.gate_passed = False
        gate.reason = f"REJECTED: {failures} morphology violation(s); utility={gate.utility_score:.6g}"
    else:
        gate.state = GateState.PASS.value
        gate.gate_passed = True
        gate.reason = f"ACCEPTED: morphology preserved; utility={gate.utility_score:.6g}"
    gate.processing_time_ms = (time.perf_counter() - start) * 1000
    return gate
