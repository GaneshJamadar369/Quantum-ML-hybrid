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

    def to_json(self, path: Path) -> None:
        """Serialize normalization parameters to JSON."""
        data = {
            "lead_medians": self.lead_medians,
            "lead_iqrs": self.lead_iqrs,
            "n_records_fitted": self.n_records_fitted,
            "folds_used": self.folds_used,
            "pipeline_version": self.pipeline_version,
            "is_fitted": self.is_fitted,
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
            logger.info(
                "Fitting normalizer on %d/%d records (folds %s)",
                mask.sum(), len(signals), allowed_folds,
            )
        else:
            train_signals = signals
            logger.info(
                "Fitting normalizer on %d records (no fold filtering)",
                len(signals),
            )

        if len(train_signals) == 0:
            raise ValueError("No training signals to fit normalizer")

        # Compute per-lead statistics across all training records
        for i, lead_name in enumerate(CANONICAL_LEAD_ORDER):
            # Gather all samples for this lead across all training records
            lead_data = train_signals[:, i, :].flatten()

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
