"""
Train-only robust normalization.

Implements plan phase 10:
- Per-lead robust statistics (median, IQR) from Folds 1–8 only
- Serializable scaler parameters (JSON)
- Same transform applied to Folds 9, 10, and future inference
- Inverse-transform and physical-unit metadata retained
- NO per-record zero-mean/unit-variance (preserves absolute amplitudes)
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import (
    CANONICAL_LEAD_ORDER,
    DEV_FOLDS,
    NUM_LEADS,
    PIPELINE_VERSION,
)

logger = logging.getLogger(__name__)


@dataclass
class NormalizationParams:
    """Serializable normalization parameters fitted on training data."""

    # Per-lead statistics
    lead_medians: Dict[str, float] = field(default_factory=dict)
    lead_iqrs: Dict[str, float] = field(default_factory=dict)

    # Metadata
    n_records_fitted: int = 0
    folds_used: list = field(default_factory=list)
    pipeline_version: str = ""
    is_fitted: bool = False
    training_patient_checksum: str = ""

    def to_json(self, path: Path) -> None:
        """Serialize normalization parameters to JSON."""
        data = {
            "lead_medians": self.lead_medians,
            "lead_iqrs": self.lead_iqrs,
            "n_records_fitted": self.n_records_fitted,
            "folds_used": self.folds_used,
            "pipeline_version": self.pipeline_version,
            "is_fitted": self.is_fitted,
            "training_patient_checksum": self.training_patient_checksum,
            "lead_order": CANONICAL_LEAD_ORDER,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved normalization params to %s", path)

    @classmethod
    def from_json(cls, path: Path) -> "NormalizationParams":
        """Load normalization parameters from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        params = cls(
            lead_medians=data["lead_medians"],
            lead_iqrs=data["lead_iqrs"],
            n_records_fitted=data["n_records_fitted"],
            folds_used=data["folds_used"],
            pipeline_version=data["pipeline_version"],
            is_fitted=data["is_fitted"],
            training_patient_checksum=data.get("training_patient_checksum", ""),
        )
        logger.info(
            "Loaded normalization params from %s (fitted on %d records)",
            path, params.n_records_fitted,
        )
        return params


class LeadRobustScaler:
    """Per-lead robust normalization using median and IQR.

    Fitted strictly on training-fold data. The same transform is applied
    to all folds and inference inputs.

    Transform:
        x_normalized = (x - median) / IQR

    This preserves relative amplitude relationships across records
    while centering and scaling each lead independently.
    """

    def __init__(self):
        self.params = NormalizationParams()

    def fit(
        self,
        signals: np.ndarray,
        folds: Optional[np.ndarray] = None,
        allowed_folds: Optional[list] = None,
        sample_masks: Optional[np.ndarray] = None,
        lead_masks: Optional[np.ndarray] = None,
        patient_ids: Optional[np.ndarray] = None,
    ) -> "LeadRobustScaler":
        """Fit normalization statistics from training signals.

        Parameters
        ----------
        signals : np.ndarray
            Shape (n_records, 12, samples) — all training signals.
        folds : np.ndarray, optional
            Shape (n_records,) — fold assignment per record.
        allowed_folds : list of int, optional
            Only use records from these folds. Defaults to DEV_FOLDS.

        Returns
        -------
        self
        """
        if allowed_folds is None:
            allowed_folds = DEV_FOLDS

        # Filter to training folds if fold info provided
        if folds is not None:
            mask = np.isin(folds, allowed_folds)
            train_signals = signals[mask]
            train_sample_masks = sample_masks[mask] if sample_masks is not None else None
            train_lead_masks = lead_masks[mask] if lead_masks is not None else None
            train_patient_ids = patient_ids[mask] if patient_ids is not None else None
            logger.info(
                "Fitting normalizer on %d/%d records (folds %s)",
                mask.sum(), len(signals), allowed_folds,
            )
        else:
            train_signals = signals
            train_sample_masks = sample_masks
            train_lead_masks = lead_masks
            train_patient_ids = patient_ids
            logger.info(
                "Fitting normalizer on %d records (no fold filtering)",
                len(signals),
            )

        if len(train_signals) == 0:
            raise ValueError("No training signals to fit normalizer")

        # Compute per-lead statistics across all training records
        for i, lead_name in enumerate(CANONICAL_LEAD_ORDER):
            # Gather all samples for this lead across all training records
            lead_data = train_signals[:, i, :].reshape(-1)
            valid = np.isfinite(lead_data)
            if train_sample_masks is not None:
                valid &= np.asarray(train_sample_masks[:, i, :], dtype=bool).reshape(-1)
            if train_lead_masks is not None:
                valid &= np.repeat(np.asarray(train_lead_masks[:, i], dtype=bool), train_signals.shape[2])
            lead_data = lead_data[valid]
            if lead_data.size == 0:
                raise ValueError(f"No valid training samples for lead {lead_name}")

            median_val = float(np.median(lead_data))
            q75 = float(np.percentile(lead_data, 75))
            q25 = float(np.percentile(lead_data, 25))
            iqr_val = q75 - q25

            # Prevent division by zero
            if iqr_val < 1e-8:
                logger.warning(
                    "Lead %s has near-zero IQR (%.6f), using 1.0",
                    lead_name, iqr_val,
                )
                iqr_val = 1.0

            self.params.lead_medians[lead_name] = median_val
            self.params.lead_iqrs[lead_name] = iqr_val

        self.params.n_records_fitted = len(train_signals)
        self.params.folds_used = sorted(allowed_folds)
        self.params.pipeline_version = PIPELINE_VERSION
        self.params.is_fitted = True
        if train_patient_ids is not None:
            ids = np.sort(np.unique(np.asarray(train_patient_ids).astype(str)))
            self.params.training_patient_checksum = hashlib.sha256(
                "\n".join(ids).encode()
            ).hexdigest()

        logger.info("Normalization fitted. Per-lead stats:")
        for lead in CANONICAL_LEAD_ORDER:
            logger.info(
                "  %s: median=%.4f mV, IQR=%.4f mV",
                lead,
                self.params.lead_medians[lead],
                self.params.lead_iqrs[lead],
            )

        return self

    def transform(self, signal: np.ndarray) -> np.ndarray:
        """Apply normalization to a signal.

        Parameters
        ----------
        signal : np.ndarray
            Shape (12, samples) or (n_records, 12, samples).

        Returns
        -------
        np.ndarray
            Normalized signal, same shape as input.
        """
        if not self.params.is_fitted:
            raise RuntimeError(
                "Normalizer is not fitted. Call fit() on training data first."
            )

        normalized = signal.copy().astype(np.float32)

        if signal.ndim == 2:
            # Single record: (12, samples)
            for i, lead_name in enumerate(CANONICAL_LEAD_ORDER):
                median = self.params.lead_medians[lead_name]
                iqr = self.params.lead_iqrs[lead_name]
                normalized[i] = (normalized[i] - median) / iqr
        elif signal.ndim == 3:
            # Batch: (n_records, 12, samples)
            for i, lead_name in enumerate(CANONICAL_LEAD_ORDER):
                median = self.params.lead_medians[lead_name]
                iqr = self.params.lead_iqrs[lead_name]
                normalized[:, i, :] = (normalized[:, i, :] - median) / iqr
        else:
            raise ValueError(
                f"Expected 2D (12, samples) or 3D (n, 12, samples), "
                f"got shape {signal.shape}"
            )

        return normalized

    def inverse_transform(self, signal: np.ndarray) -> np.ndarray:
        """Reverse the normalization to recover physical units (mV).

        Parameters
        ----------
        signal : np.ndarray
            Normalized signal, shape (12, samples) or (n, 12, samples).

        Returns
        -------
        np.ndarray
            Signal in original physical units (mV).
        """
        if not self.params.is_fitted:
            raise RuntimeError("Normalizer is not fitted.")

        original = signal.copy().astype(np.float32)

        if signal.ndim == 2:
            for i, lead_name in enumerate(CANONICAL_LEAD_ORDER):
                median = self.params.lead_medians[lead_name]
                iqr = self.params.lead_iqrs[lead_name]
                original[i] = original[i] * iqr + median
        elif signal.ndim == 3:
            for i, lead_name in enumerate(CANONICAL_LEAD_ORDER):
                median = self.params.lead_medians[lead_name]
                iqr = self.params.lead_iqrs[lead_name]
                original[:, i, :] = original[:, i, :] * iqr + median
        else:
            raise ValueError(f"Unexpected shape {signal.shape}")

        return original

    def save(self, path: Path) -> None:
        """Save normalization parameters to JSON."""
        self.params.to_json(path)

    def load(self, path: Path) -> "LeadRobustScaler":
        """Load normalization parameters from JSON."""
        self.params = NormalizationParams.from_json(path)
        return self


def fit_fold_local_scalers_from_hdf5(
    hdf5_path: Path,
    output_dir: Path,
    samples_per_record: int = 100,
) -> Dict[int, LeadRobustScaler]:
    """Fit bounded-memory normalizers for each held-out development fold.

    Samples are taken at deterministic, evenly spaced positions from every
    eligible training record. Masks and failed leads are excluded. The saved
    patient checksum proves exactly which patient set fitted each scaler.
    """
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 normalization") from exc
    from .manifest import guard_fold_access

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scalers: Dict[int, LeadRobustScaler] = {}
    with h5py.File(Path(hdf5_path), "r") as h5:
        folds = h5["strat_fold"][:].astype(int)
        guard_fold_access(folds, purpose="feature_selection")
        if not set(np.unique(folds)).issubset(set(DEV_FOLDS)):
            raise ValueError("Fold-local development scalers accept folds 1–8 only")
        patients = h5["patient_id"][:]
        n_samples = int(h5["accepted_signal"].shape[2])
        positions = np.unique(np.linspace(0, n_samples - 1, min(samples_per_record, n_samples), dtype=int))
        for held_out in sorted(np.unique(folds)):
            train_rows = np.where(folds != held_out)[0]
            allowed = sorted(int(v) for v in np.unique(folds[train_rows]))
            scaler = LeadRobustScaler()
            for lead_index, lead_name in enumerate(CANONICAL_LEAD_ORDER):
                pieces = []
                for start in range(0, len(train_rows), 256):
                    rows = train_rows[start:start + 256]
                    # h5py requires increasing indices; train_rows is sorted.
                    values = h5["accepted_signal"][rows, lead_index, :][:, positions]
                    sample_valid = h5["sample_mask"][rows, lead_index, :][:, positions]
                    lead_valid = h5["lead_mask"][rows, lead_index][:, None]
                    valid = sample_valid & lead_valid & np.isfinite(values)
                    if valid.any():
                        pieces.append(values[valid].astype(np.float32))
                if not pieces:
                    raise ValueError(f"No valid samples for lead {lead_name}, holdout {held_out}")
                data = np.concatenate(pieces)
                median = float(np.median(data))
                iqr = float(np.percentile(data, 75) - np.percentile(data, 25))
                scaler.params.lead_medians[lead_name] = median
                scaler.params.lead_iqrs[lead_name] = iqr if iqr >= 1e-8 else 1.0
            patient_values = np.sort(np.unique(patients[train_rows].astype(str)))
            scaler.params.n_records_fitted = int(len(train_rows))
            scaler.params.folds_used = allowed
            scaler.params.pipeline_version = PIPELINE_VERSION
            scaler.params.training_patient_checksum = hashlib.sha256(
                "\n".join(patient_values).encode()
            ).hexdigest()
            scaler.params.is_fitted = True
            scaler.save(output_dir / f"normalizer_holdout_fold_{int(held_out)}.json")
            scalers[int(held_out)] = scaler
    return scalers
