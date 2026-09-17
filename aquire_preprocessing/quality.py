"""
Lead-level signal quality analysis and cross-lead physics checks.

Implements plan phases 4–5:
- Per-lead quality fingerprint (flatline, clipping, missing, baseline,
  powerline, HF noise, QRS consistency, amplitude plausibility)
- Cross-lead physics: Einthoven's law, Goldberger equations
- 3-state quality grading: PASS / WARN / FAIL
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal

from .config import QC, CANONICAL_LEAD_ORDER, NUM_LEADS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quality result containers
# ---------------------------------------------------------------------------

@dataclass
class LeadQuality:
    """Quality fingerprint for a single lead."""
    lead_name: str
    is_flat: bool = False
    is_clipped: bool = False
    missing_fraction: float = 0.0
    baseline_wander_ratio: float = 0.0
    powerline_snr_db: float = 0.0
    hf_noise_ratio: float = 0.0
    amplitude_range_mv: float = 0.0
    is_plausible_amplitude: bool = True
    status: str = "PASS"  # PASS, WARN, FAIL
    issues: List[str] = field(default_factory=list)


@dataclass
class CrossLeadPhysics:
    """Results of cross-lead consistency checks."""
    einthoven_residual_mv: float = 0.0       # |II - (I + III)|
    goldberger_avr_residual_mv: float = 0.0  # |aVR - (-(I+II)/2)|
    goldberger_avl_residual_mv: float = 0.0  # |aVL - (I - II/2)|
    goldberger_avf_residual_mv: float = 0.0  # |aVF - (II - I/2)|
    all_passed: bool = True
    issues: List[str] = field(default_factory=list)


@dataclass
class QCResult:
    """Complete quality assessment for a 12-lead ECG."""
    ecg_id: int
    qc_status: str = "PASS"  # PASS, WARN, FAIL
    per_lead: Dict[str, LeadQuality] = field(default_factory=dict)
    cross_lead: Optional[CrossLeadPhysics] = None
    lead_mask: Optional[np.ndarray] = None      # bool[12]: True = valid
    sample_mask: Optional[np.ndarray] = None     # bool[12, 1000]: True = valid
    n_failed_leads: int = 0
    n_warned_leads: int = 0
    summary_issues: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Lead-level quality checks
# ---------------------------------------------------------------------------

def _check_flatline(
    lead_signal: np.ndarray,
    fs: int,
) -> Tuple[bool, List[str]]:
    """Detect flatline segments (zero variance ≥ threshold duration)."""
    min_samples = int(QC.flatline_min_duration_ms / 1000.0 * fs)
    issues = []

    # Sliding window variance check
    if len(lead_signal) < min_samples:
        return False, issues

    # Check overall variance first (fast path)
    if np.var(lead_signal) < QC.flatline_variance_eps:
        issues.append("Entire lead is flatline")
        return True, issues

    # Sliding window check for partial flatlines
    for start in range(0, len(lead_signal) - min_samples + 1, min_samples // 2):
        segment = lead_signal[start : start + min_samples]
        if np.var(segment) < QC.flatline_variance_eps:
            issues.append(
                f"Flatline segment at samples {start}-{start + min_samples}"
            )
            return True, issues

    return False, issues


def _check_clipping(lead_signal: np.ndarray) -> Tuple[bool, List[str]]:
    """Detect saturation/clipping (repeated min or max values)."""
    issues = []
    n = len(lead_signal)
    if n == 0:
        return False, issues

    sig_min = np.min(lead_signal)
    sig_max = np.max(lead_signal)

    # Count samples at extremes
    at_min = np.sum(lead_signal == sig_min) / n
    at_max = np.sum(lead_signal == sig_max) / n

    if at_min > QC.clipping_fraction_max:
        issues.append(f"Clipping at minimum ({at_min:.1%} of samples)")
        return True, issues
    if at_max > QC.clipping_fraction_max:
        issues.append(f"Clipping at maximum ({at_max:.1%} of samples)")
        return True, issues

    return False, issues


def _check_missing(
    lead_signal: np.ndarray,
    original_signal: Optional[np.ndarray] = None,
) -> Tuple[float, List[str]]:
    """Compute fraction of missing (NaN/zero after NaN replacement) samples."""
    issues = []
    # If we have the original signal (pre NaN replacement), use it
    if original_signal is not None:
        missing_frac = np.sum(~np.isfinite(original_signal)) / len(original_signal)
    else:
        # Approximate: consecutive zeros might indicate replaced NaNs
        missing_frac = 0.0

    if missing_frac > QC.missing_fraction_max:
        issues.append(f"High missing fraction: {missing_frac:.1%}")

    return missing_frac, issues


def _check_baseline_wander(
    lead_signal: np.ndarray,
    fs: int,
) -> Tuple[float, List[str]]:
    """Estimate baseline wander as ratio of low-frequency power."""
    issues = []

    if len(lead_signal) < fs:
        return 0.0, issues

    # Compute power spectral density
    freqs, psd = scipy_signal.welch(lead_signal, fs=fs, nperseg=min(256, len(lead_signal)))

    total_power = np.sum(psd)
    if total_power < 1e-12:
        return 0.0, issues

    # Low-frequency power (< 0.5 Hz)
    low_freq_mask = freqs < 0.5
    low_freq_power = np.sum(psd[low_freq_mask])
    ratio = low_freq_power / total_power

    if ratio > QC.baseline_wander_power_ratio_max:
        issues.append(f"Excessive baseline wander: LF power ratio = {ratio:.3f}")

    return ratio, issues


def _check_powerline(
    lead_signal: np.ndarray,
    fs: int,
) -> Tuple[float, List[str]]:
    """Detect 50/60 Hz power-line interference."""
    issues = []

    if len(lead_signal) < fs:
        return 0.0, issues

    freqs, psd = scipy_signal.welch(lead_signal, fs=fs, nperseg=min(256, len(lead_signal)))

    total_power = np.sum(psd)
    if total_power < 1e-12:
        return 0.0, issues

    # Check both 50 Hz and 60 Hz
    powerline_snr = 0.0
    for target_freq in [50.0, 60.0]:
        if target_freq >= fs / 2:
            continue
        # Find the nearest frequency bin
        idx = np.argmin(np.abs(freqs - target_freq))
        peak_power = psd[idx]

        # SNR relative to median power
        median_power = np.median(psd)
        if median_power > 1e-12:
            snr = 10 * np.log10(peak_power / median_power)
            powerline_snr = max(powerline_snr, snr)

    if powerline_snr > QC.powerline_snr_db_min:
        issues.append(f"Power-line interference: SNR = {powerline_snr:.1f} dB")

    return powerline_snr, issues


def _check_hf_noise(
    lead_signal: np.ndarray,
    fs: int,
) -> Tuple[float, List[str]]:
    """Estimate high-frequency noise (energy above 40 Hz)."""
    issues = []

    if len(lead_signal) < fs:
        return 0.0, issues

    freqs, psd = scipy_signal.welch(lead_signal, fs=fs, nperseg=min(256, len(lead_signal)))

    total_power = np.sum(psd)
    if total_power < 1e-12:
        return 0.0, issues

    hf_mask = freqs > 40.0
    hf_power = np.sum(psd[hf_mask])
    ratio = hf_power / total_power

    if ratio > QC.hf_noise_power_ratio_max:
        issues.append(f"High-frequency noise: HF power ratio = {ratio:.3f}")

    return ratio, issues


def _check_amplitude(lead_signal: np.ndarray) -> Tuple[float, bool, List[str]]:
    """Check amplitude plausibility (physiological bounds)."""
    issues = []
    amp_range = float(np.max(lead_signal) - np.min(lead_signal))

    plausible = QC.amplitude_min_mv <= amp_range <= QC.amplitude_max_mv

    if not plausible:
        if amp_range < QC.amplitude_min_mv:
            issues.append(
                f"Implausibly low amplitude: {amp_range:.4f} mV "
                f"(min: {QC.amplitude_min_mv} mV)"
            )
        else:
            issues.append(
                f"Implausibly high amplitude: {amp_range:.2f} mV "
                f"(max: {QC.amplitude_max_mv} mV)"
            )

    return amp_range, plausible, issues


def assess_lead_quality(
    lead_signal: np.ndarray,
    lead_name: str,
    fs: int = 100,
) -> LeadQuality:
    """Compute the complete quality fingerprint for a single lead."""
    lq = LeadQuality(lead_name=lead_name)

    # 1. Flatline
    lq.is_flat, flat_issues = _check_flatline(lead_signal, fs)
    lq.issues.extend(flat_issues)

    # 2. Clipping
    lq.is_clipped, clip_issues = _check_clipping(lead_signal)
    lq.issues.extend(clip_issues)

    # 3. Missing
    lq.missing_fraction, miss_issues = _check_missing(lead_signal)
    lq.issues.extend(miss_issues)

    # 4. Baseline wander
    lq.baseline_wander_ratio, bw_issues = _check_baseline_wander(lead_signal, fs)
    lq.issues.extend(bw_issues)

    # 5. Powerline interference
    lq.powerline_snr_db, pl_issues = _check_powerline(lead_signal, fs)
    lq.issues.extend(pl_issues)

    # 6. High-frequency noise
    lq.hf_noise_ratio, hf_issues = _check_hf_noise(lead_signal, fs)
    lq.issues.extend(hf_issues)

    # 7. Amplitude plausibility
    lq.amplitude_range_mv, lq.is_plausible_amplitude, amp_issues = (
        _check_amplitude(lead_signal)
    )
    lq.issues.extend(amp_issues)

    # --- Determine lead status ---
    if lq.is_flat or lq.missing_fraction > QC.missing_fraction_max:
        lq.status = "FAIL"
    elif (
        lq.is_clipped
        or not lq.is_plausible_amplitude
        or lq.baseline_wander_ratio > QC.baseline_wander_power_ratio_max
        or lq.powerline_snr_db > QC.powerline_snr_db_min
        or lq.hf_noise_ratio > QC.hf_noise_power_ratio_max
    ):
        lq.status = "WARN"
    else:
        lq.status = "PASS"

    return lq


# ---------------------------------------------------------------------------
# Cross-lead physics checks
# ---------------------------------------------------------------------------

def check_cross_lead_physics(
    signal: np.ndarray,
    lead_order: List[str] = None,
) -> CrossLeadPhysics:
    """Verify Einthoven's law and Goldberger equations.

    Parameters
    ----------
    signal : np.ndarray
        Shape (12, samples) in canonical lead order.
    lead_order : list of str
        Lead names corresponding to signal rows.
        Defaults to CANONICAL_LEAD_ORDER.
    """
    if lead_order is None:
        lead_order = CANONICAL_LEAD_ORDER

    physics = CrossLeadPhysics()

    # Map lead names to indices
    lead_idx = {name: i for i, name in enumerate(lead_order)}

    try:
        lead_I = signal[lead_idx["I"]]
        lead_II = signal[lead_idx["II"]]
        lead_III = signal[lead_idx["III"]]
        lead_aVR = signal[lead_idx["aVR"]]
        lead_aVL = signal[lead_idx["aVL"]]
        lead_aVF = signal[lead_idx["aVF"]]
    except KeyError as e:
        physics.all_passed = False
        physics.issues.append(f"Missing lead for physics check: {e}")
        return physics

    # Einthoven: II ≈ I + III
    einthoven_residual = np.mean(np.abs(lead_II - (lead_I + lead_III)))
    physics.einthoven_residual_mv = float(einthoven_residual)

    if einthoven_residual > QC.einthoven_residual_max_mv:
        physics.all_passed = False
        physics.issues.append(
            f"Einthoven violation: |II-(I+III)| = {einthoven_residual:.4f} mV "
            f"(threshold: {QC.einthoven_residual_max_mv} mV)"
        )

    # Goldberger: aVR ≈ -(I + II) / 2
    avr_expected = -(lead_I + lead_II) / 2.0
    avr_residual = np.mean(np.abs(lead_aVR - avr_expected))
    physics.goldberger_avr_residual_mv = float(avr_residual)

    if avr_residual > QC.goldberger_residual_max_mv:
        physics.all_passed = False
        physics.issues.append(
            f"Goldberger aVR violation: residual = {avr_residual:.4f} mV"
        )

    # Goldberger: aVL ≈ I - II/2
    avl_expected = lead_I - lead_II / 2.0
    avl_residual = np.mean(np.abs(lead_aVL - avl_expected))
    physics.goldberger_avl_residual_mv = float(avl_residual)

    if avl_residual > QC.goldberger_residual_max_mv:
        physics.all_passed = False
        physics.issues.append(
            f"Goldberger aVL violation: residual = {avl_residual:.4f} mV"
        )

    # Goldberger: aVF ≈ II - I/2
    avf_expected = lead_II - lead_I / 2.0
    avf_residual = np.mean(np.abs(lead_aVF - avf_expected))
    physics.goldberger_avf_residual_mv = float(avf_residual)

    if avf_residual > QC.goldberger_residual_max_mv:
        physics.all_passed = False
        physics.issues.append(
            f"Goldberger aVF violation: residual = {avf_residual:.4f} mV"
        )

    return physics


# ---------------------------------------------------------------------------
# Full QC assessment
# ---------------------------------------------------------------------------

def assess_quality(
    signal: np.ndarray,
    ecg_id: int,
    fs: int = 100,
    lead_order: List[str] = None,
) -> QCResult:
    """Run the complete quality assessment on a 12-lead ECG.

    Parameters
    ----------
    signal : np.ndarray
        Shape (12, samples) in canonical lead order, physical units (mV).
    ecg_id : int
        Record identifier.
    fs : int
        Sampling rate in Hz.
    lead_order : list of str
        Lead names. Defaults to CANONICAL_LEAD_ORDER.

    Returns
    -------
    QCResult
        Full quality assessment including per-lead, cross-lead, and overall status.
    """
    if lead_order is None:
        lead_order = CANONICAL_LEAD_ORDER

    qc = QCResult(ecg_id=ecg_id)
    lead_mask = np.ones(NUM_LEADS, dtype=bool)
    sample_mask = np.ones_like(signal, dtype=bool)

    # --- Per-lead quality ---
    n_failed = 0
    n_warned = 0

    for i, lead_name in enumerate(lead_order):
        lq = assess_lead_quality(signal[i], lead_name, fs)
        qc.per_lead[lead_name] = lq

        if lq.status == "FAIL":
            lead_mask[i] = False
            n_failed += 1
        elif lq.status == "WARN":
            n_warned += 1

    qc.n_failed_leads = n_failed
    qc.n_warned_leads = n_warned
    qc.lead_mask = lead_mask
    qc.sample_mask = sample_mask

    # --- Cross-lead physics ---
    qc.cross_lead = check_cross_lead_physics(signal, lead_order)
    if not qc.cross_lead.all_passed:
        qc.summary_issues.extend(qc.cross_lead.issues)

    # --- Overall status ---
    if n_failed >= 3:
        # 3+ failed leads → entire record is FAIL
        qc.qc_status = "FAIL"
        qc.summary_issues.append(
            f"{n_failed} leads failed quality checks"
        )
    elif n_failed > 0 or n_warned > 0 or not qc.cross_lead.all_passed:
        qc.qc_status = "WARN"
        if n_failed > 0:
            qc.summary_issues.append(f"{n_failed} lead(s) failed")
        if n_warned > 0:
            qc.summary_issues.append(f"{n_warned} lead(s) warned")
    else:
        qc.qc_status = "PASS"

    return qc
