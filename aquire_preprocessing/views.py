"""
Signal view generator.

Implements plan phase 8:
- View A: Minimally processed (physical units, canonical leads, shape only)
- View B: Diagnostically corrected (router-indicated corrections applied)
- View C: Controlled stress-test (synthetic corruption for training augmentation)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import CANONICAL_LEAD_ORDER, NUM_LEADS
from .filters import apply_corrections
from .router import RoutingDecision, ProcessingAction

logger = logging.getLogger(__name__)


@dataclass
class SignalViews:
    """Container for the multiple signal views of a single ECG."""
    ecg_id: int
    signal_minimal: np.ndarray = None       # View A: float32[12, 1000]
    signal_corrected: np.ndarray = None     # View B: float32[12, 1000]
    lead_mask: np.ndarray = None            # bool[12]
    sample_mask: np.ndarray = None          # bool[12, 1000]
    corrections_applied: List[str] = field(default_factory=list)
    view_b_differs: bool = False            # True if B differs from A


def generate_view_a(
    signal: np.ndarray,
    ecg_id: int,
) -> np.ndarray:
    """Generate View A: minimally processed signal.

    View A applies only:
    - Physical-unit conversion (already done in structural validation)
    - Lead reordering (already done in structural validation)
    - Duration and shape validation (already done)

    This function just validates and returns a copy.

    Parameters
    ----------
    signal : np.ndarray
        Structurally validated signal, float32[12, samples].
    ecg_id : int
        Record identifier for logging.

    Returns
    -------
    np.ndarray
        View A signal (copy of input).
    """
    assert signal.shape[0] == NUM_LEADS, (
        f"Expected {NUM_LEADS} leads, got {signal.shape[0]}"
    )
    return signal.copy()


def generate_view_b(
    signal_minimal: np.ndarray,
    routing: RoutingDecision,
    ecg_id: int,
    fs: int = 100,
) -> tuple:
    """Generate View B: diagnostically corrected signal.

    Applies only the corrections indicated by the processing router.
    If no corrections are needed, View B equals View A.

    Parameters
    ----------
    signal_minimal : np.ndarray
        View A signal, float32[12, samples].
    routing : RoutingDecision
        Router output indicating per-lead actions.
    ecg_id : int
        Record identifier.
    fs : int
        Sampling rate.

    Returns
    -------
    signal_corrected : np.ndarray
        View B signal, float32[12, samples].
    sample_mask : np.ndarray
        Updated sample mask, bool[12, samples].
    lead_mask : np.ndarray
        Lead validity mask, bool[12].
    corrections_log : list of str
        Human-readable log of corrections applied.
    """
    n_leads, n_samples = signal_minimal.shape
    signal_corrected = signal_minimal.copy()
    sample_mask = np.ones((n_leads, n_samples), dtype=bool)
    lead_mask = np.ones(n_leads, dtype=bool)
    corrections_log = []

    if not routing.requires_correction:
        return signal_corrected, sample_mask, lead_mask, corrections_log

    for i, lead_name in enumerate(CANONICAL_LEAD_ORDER):
        actions = routing.per_lead_actions.get(lead_name, ["minimal"])

        # Handle lead mask
        if ProcessingAction.LEAD_MASK.value in actions:
            lead_mask[i] = False
            sample_mask[i, :] = False
            corrections_log.append(f"{lead_name}: masked (failed QC)")
            continue

        # Apply corrections
        if any(a != "minimal" for a in actions):
            corrected_lead, updated_mask = apply_corrections(
                signal_minimal[i],
                actions,
                fs=fs,
                sample_mask=sample_mask[i],
            )
            signal_corrected[i] = corrected_lead
            sample_mask[i] = updated_mask
            corrections_log.append(
                f"{lead_name}: {', '.join(a for a in actions if a != 'minimal')}"
            )

    return signal_corrected, sample_mask, lead_mask, corrections_log


def generate_views(
    signal: np.ndarray,
    routing: RoutingDecision,
    ecg_id: int,
    fs: int = 100,
) -> SignalViews:
    """Generate all signal views for a record.

    Parameters
    ----------
    signal : np.ndarray
        Structurally validated signal, float32[12, samples].
    routing : RoutingDecision
        Router output.
    ecg_id : int
        Record identifier.
    fs : int
        Sampling rate.

    Returns
    -------
    SignalViews
        Container with View A, View B, masks, and correction log.
    """
    views = SignalViews(ecg_id=ecg_id)

    # View A: minimal
    views.signal_minimal = generate_view_a(signal, ecg_id)

    # View B: corrected
    if routing.overall_action == ProcessingAction.FAIL.value:
        # FAIL records: View B = View A, all masks = False
        views.signal_corrected = views.signal_minimal.copy()
        views.lead_mask = np.zeros(NUM_LEADS, dtype=bool)
        views.sample_mask = np.zeros_like(signal, dtype=bool)
        views.corrections_applied = ["RECORD_FAILED"]
        views.view_b_differs = False
    else:
        corrected, s_mask, l_mask, log = generate_view_b(
            views.signal_minimal, routing, ecg_id, fs
        )
        views.signal_corrected = corrected
        views.sample_mask = s_mask
        views.lead_mask = l_mask
        views.corrections_applied = log
        views.view_b_differs = not np.allclose(
            views.signal_minimal, views.signal_corrected, atol=1e-6
        )

    return views


# ---------------------------------------------------------------------------
# View C: Stress-test corruption library (training only)
# ---------------------------------------------------------------------------

def generate_view_c_baseline_drift(
    signal: np.ndarray,
    fs: int = 100,
    amplitude_mv: float = 0.3,
    freq_hz: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """Inject synthetic baseline drift for robustness testing.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal, float32[12, samples].
    fs : int
        Sampling rate.
    amplitude_mv : float
        Drift amplitude in mV.
    freq_hz : float
        Drift frequency in Hz.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Corrupted signal with baseline drift added.
    """
    rng = np.random.RandomState(seed)
    n_leads, n_samples = signal.shape
    t = np.arange(n_samples) / fs

    corrupted = signal.copy()
    for i in range(n_leads):
        phase = rng.uniform(0, 2 * np.pi)
        drift = amplitude_mv * np.sin(2 * np.pi * freq_hz * t + phase)
        corrupted[i] += drift.astype(np.float32)

    return corrupted


def generate_view_c_gaussian_noise(
    signal: np.ndarray,
    snr_db: float = 20.0,
    seed: int = 42,
) -> np.ndarray:
    """Add Gaussian noise at a specified SNR level."""
    rng = np.random.RandomState(seed)
    corrupted = signal.copy()

    for i in range(signal.shape[0]):
        sig_power = np.mean(signal[i] ** 2)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = rng.normal(0, np.sqrt(noise_power), signal.shape[1])
        corrupted[i] += noise.astype(np.float32)

    return corrupted


def generate_view_c_lead_dropout(
    signal: np.ndarray,
    drop_leads: List[int] = None,
    seed: int = 42,
) -> tuple:
    """Simulate lead disconnection by zeroing selected leads.

    Returns the corrupted signal and a lead mask indicating which leads
    are still valid.
    """
    rng = np.random.RandomState(seed)
    corrupted = signal.copy()

    if drop_leads is None:
        # Randomly drop 1-2 leads
        n_drop = rng.randint(1, 3)
        drop_leads = rng.choice(signal.shape[0], n_drop, replace=False).tolist()

    lead_mask = np.ones(signal.shape[0], dtype=bool)
    for lead_idx in drop_leads:
        corrupted[lead_idx] = 0.0
        lead_mask[lead_idx] = False

    return corrupted, lead_mask
