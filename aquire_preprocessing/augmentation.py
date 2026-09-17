"""
Physiologically plausible training augmentation.

Implements plan phases 11–12:
- Permitted: amplitude scaling, time translation, baseline drift,
  muscular noise, electrode-motion, short masking, lead dropout,
  small sampling-rate variation
- Forbidden: lead permutation, time reversal, large nonlinear warping,
  ST/T-wave inversion
- Full provenance logging per augmented sample
- NEVER applied to Fold 9 or Fold 10
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import NUM_LEADS, DEV_FOLDS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentation provenance record
# ---------------------------------------------------------------------------

@dataclass
class AugmentationRecord:
    """Provenance record for a single augmented sample."""
    source_ecg_id: int
    corruption_type: str
    severity: float
    affected_leads: List[int] = field(default_factory=list)
    affected_interval: Optional[Tuple[int, int]] = None  # (start, end) samples
    random_seed: int = 0
    augmentation_version: str = "v0.1.0"


# ---------------------------------------------------------------------------
# Fold guard
# ---------------------------------------------------------------------------

def _check_fold_guard(fold: int, operation: str) -> None:
    """Raise an error if augmentation is attempted on non-training folds."""
    if fold not in DEV_FOLDS:
        raise ValueError(
            f"AUGMENTATION BLOCKED: {operation} attempted on fold {fold}. "
            f"Augmentation is only permitted on folds {DEV_FOLDS}."
        )


# ---------------------------------------------------------------------------
# Permitted augmentations
# ---------------------------------------------------------------------------

def augment_amplitude_scale(
    signal: np.ndarray,
    ecg_id: int,
    fold: int,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    seed: int = 42,
) -> Tuple[np.ndarray, AugmentationRecord]:
    """Apply small random amplitude scaling.

    Scales all leads by the same factor to preserve inter-lead ratios.
    """
    _check_fold_guard(fold, "amplitude_scale")
    rng = np.random.RandomState(seed)
    scale = rng.uniform(scale_range[0], scale_range[1])

    augmented = signal * scale
    record = AugmentationRecord(
        source_ecg_id=ecg_id,
        corruption_type="amplitude_scale",
        severity=abs(scale - 1.0),
        affected_leads=list(range(NUM_LEADS)),
        random_seed=seed,
    )
    return augmented.astype(np.float32), record


def augment_time_shift(
    signal: np.ndarray,
    ecg_id: int,
    fold: int,
    max_shift_samples: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, AugmentationRecord]:
    """Apply small time translation (circular shift)."""
    _check_fold_guard(fold, "time_shift")
    rng = np.random.RandomState(seed)
    shift = rng.randint(-max_shift_samples, max_shift_samples + 1)

    augmented = np.roll(signal, shift, axis=1)
    record = AugmentationRecord(
        source_ecg_id=ecg_id,
        corruption_type="time_shift",
        severity=abs(shift) / max_shift_samples,
        affected_leads=list(range(NUM_LEADS)),
        random_seed=seed,
    )
    return augmented.astype(np.float32), record


def augment_baseline_drift(
    signal: np.ndarray,
    ecg_id: int,
    fold: int,
    amplitude_range_mv: Tuple[float, float] = (0.05, 0.3),
    freq_range_hz: Tuple[float, float] = (0.05, 0.3),
    fs: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, AugmentationRecord]:
    """Add realistic baseline drift (low-frequency sinusoidal wander)."""
    _check_fold_guard(fold, "baseline_drift")
    rng = np.random.RandomState(seed)

    amplitude = rng.uniform(amplitude_range_mv[0], amplitude_range_mv[1])
    freq = rng.uniform(freq_range_hz[0], freq_range_hz[1])

    n_samples = signal.shape[1]
    t = np.arange(n_samples) / fs

    augmented = signal.copy()
    affected_leads = []

    for i in range(NUM_LEADS):
        if rng.random() > 0.3:  # 70% chance per lead
            phase = rng.uniform(0, 2 * np.pi)
            drift = amplitude * np.sin(2 * np.pi * freq * t + phase)
            augmented[i] += drift.astype(np.float32)
            affected_leads.append(i)

    record = AugmentationRecord(
        source_ecg_id=ecg_id,
        corruption_type="baseline_drift",
        severity=amplitude,
        affected_leads=affected_leads,
        random_seed=seed,
    )
    return augmented.astype(np.float32), record


def augment_muscle_noise(
    signal: np.ndarray,
    ecg_id: int,
    fold: int,
    snr_db_range: Tuple[float, float] = (15.0, 30.0),
    seed: int = 42,
) -> Tuple[np.ndarray, AugmentationRecord]:
    """Add realistic high-frequency muscular noise (EMG artifact)."""
    _check_fold_guard(fold, "muscle_noise")
    rng = np.random.RandomState(seed)

    snr_db = rng.uniform(snr_db_range[0], snr_db_range[1])
    augmented = signal.copy()
    affected_leads = []

    for i in range(NUM_LEADS):
        if rng.random() > 0.4:  # 60% chance per lead
            sig_power = np.mean(signal[i] ** 2)
            if sig_power < 1e-12:
                continue
            noise_power = sig_power / (10 ** (snr_db / 10))
            noise = rng.normal(0, np.sqrt(noise_power), signal.shape[1])
            augmented[i] += noise.astype(np.float32)
            affected_leads.append(i)

    record = AugmentationRecord(
        source_ecg_id=ecg_id,
        corruption_type="muscle_noise",
        severity=1.0 / snr_db,  # higher severity = lower SNR
        affected_leads=affected_leads,
        random_seed=seed,
    )
    return augmented.astype(np.float32), record


def augment_electrode_motion(
    signal: np.ndarray,
    ecg_id: int,
    fold: int,
    amplitude_mv: float = 0.5,
    duration_ms: float = 200.0,
    fs: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, AugmentationRecord]:
    """Simulate short electrode-motion artifact (transient spike)."""
    _check_fold_guard(fold, "electrode_motion")
    rng = np.random.RandomState(seed)

    n_samples = signal.shape[1]
    duration_samples = int(duration_ms / 1000.0 * fs)

    # Random start position
    start = rng.randint(0, max(1, n_samples - duration_samples))
    end = min(start + duration_samples, n_samples)

    # Random lead(s)
    n_affected = rng.randint(1, 4)
    affected_leads = rng.choice(NUM_LEADS, n_affected, replace=False).tolist()

    augmented = signal.copy()
    for lead_idx in affected_leads:
        artifact = rng.normal(0, amplitude_mv, end - start)
        augmented[lead_idx, start:end] += artifact.astype(np.float32)

    record = AugmentationRecord(
        source_ecg_id=ecg_id,
        corruption_type="electrode_motion",
        severity=amplitude_mv,
        affected_leads=affected_leads,
        affected_interval=(int(start), int(end)),
        random_seed=seed,
    )
    return augmented.astype(np.float32), record


def augment_short_masking(
    signal: np.ndarray,
    ecg_id: int,
    fold: int,
    max_mask_ms: float = 100.0,
    fs: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, AugmentationRecord]:
    """Apply short contiguous masking (set to zero + mask).

    Returns the augmented signal AND an updated sample mask.
    """
    _check_fold_guard(fold, "short_masking")
    rng = np.random.RandomState(seed)

    n_samples = signal.shape[1]
    mask_samples = int(max_mask_ms / 1000.0 * fs)

    start = rng.randint(0, max(1, n_samples - mask_samples))
    end = min(start + mask_samples, n_samples)

    # Random subset of leads
    n_affected = rng.randint(1, NUM_LEADS + 1)
    affected_leads = rng.choice(NUM_LEADS, n_affected, replace=False).tolist()

    augmented = signal.copy()
    sample_mask = np.ones_like(signal, dtype=bool)

    for lead_idx in affected_leads:
        augmented[lead_idx, start:end] = 0.0
        sample_mask[lead_idx, start:end] = False

    record = AugmentationRecord(
        source_ecg_id=ecg_id,
        corruption_type="short_masking",
        severity=mask_samples / n_samples,
        affected_leads=affected_leads,
        affected_interval=(int(start), int(end)),
        random_seed=seed,
    )
    return augmented.astype(np.float32), sample_mask, record


def augment_lead_dropout(
    signal: np.ndarray,
    ecg_id: int,
    fold: int,
    max_leads: int = 2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, AugmentationRecord]:
    """Simulate complete lead disconnection (zero the lead).

    Returns augmented signal and lead mask.
    """
    _check_fold_guard(fold, "lead_dropout")
    rng = np.random.RandomState(seed)

    n_drop = rng.randint(1, max_leads + 1)
    dropped = rng.choice(NUM_LEADS, n_drop, replace=False).tolist()

    augmented = signal.copy()
    lead_mask = np.ones(NUM_LEADS, dtype=bool)

    for lead_idx in dropped:
        augmented[lead_idx] = 0.0
        lead_mask[lead_idx] = False

    record = AugmentationRecord(
        source_ecg_id=ecg_id,
        corruption_type="lead_dropout",
        severity=n_drop / NUM_LEADS,
        affected_leads=dropped,
        random_seed=seed,
    )
    return augmented.astype(np.float32), lead_mask, record


# ---------------------------------------------------------------------------
# Composite augmentation pipeline
# ---------------------------------------------------------------------------

def apply_random_augmentation(
    signal: np.ndarray,
    ecg_id: int,
    fold: int,
    fs: int = 100,
    seed: int = 42,
    max_augmentations: int = 2,
) -> Tuple[np.ndarray, List[AugmentationRecord]]:
    """Apply a random combination of permitted augmentations.

    Parameters
    ----------
    signal : np.ndarray
        Clean signal, float32[12, samples].
    ecg_id : int
        Source record identifier.
    fold : int
        Stratification fold (must be in DEV_FOLDS).
    fs : int
        Sampling rate.
    seed : int
        Random seed.
    max_augmentations : int
        Maximum number of augmentations to apply.

    Returns
    -------
    augmented : np.ndarray
        Augmented signal.
    records : list of AugmentationRecord
        Full provenance trail.
    """
    _check_fold_guard(fold, "random_augmentation")
    rng = np.random.RandomState(seed)

    augmentation_fns = [
        augment_amplitude_scale,
        augment_time_shift,
        augment_baseline_drift,
        augment_muscle_noise,
        augment_electrode_motion,
    ]

    n_augs = rng.randint(1, max_augmentations + 1)
    selected = rng.choice(len(augmentation_fns), n_augs, replace=False)

    augmented = signal.copy()
    records = []

    for i, fn_idx in enumerate(selected):
        fn = augmentation_fns[fn_idx]
        sub_seed = seed + i * 1000 + fn_idx
        augmented, record = fn(
            augmented,
            ecg_id=ecg_id,
            fold=fold,
            seed=sub_seed,
        )
        records.append(record)

    return augmented, records
