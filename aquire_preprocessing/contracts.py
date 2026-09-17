"""Public, versioned data contracts for the AQUIRE-Med pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class Eligibility(str, Enum):
    PRIMARY = "PRIMARY"
    QUARANTINE = "QUARANTINE"
    REJECTED = "REJECTED"


class GateState(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    INDETERMINATE = "INDETERMINATE"


@dataclass
class ValidatedECG:
    """Canonical ECG plus the provenance needed to interpret every value."""

    ecg_id: int
    signal_mv: Optional[np.ndarray] = None
    sample_mask: Optional[np.ndarray] = None
    lead_mask: Optional[np.ndarray] = None
    sampling_rate: int = 0
    original_shape: Tuple[int, int] = (0, 0)
    lead_order: List[str] = field(default_factory=list)
    source_paths: List[str] = field(default_factory=list)
    source_checksum: str = ""
    transformations: List[str] = field(default_factory=list)
    is_valid: bool = False
    physical_units: str = "mV"
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def signal(self) -> Optional[np.ndarray]:
        """Compatibility alias for the v0.1 StructuralValidation API."""
        return self.signal_mv


@dataclass
class ProcessedECG:
    """Final preprocessing boundary consumed by feature/model code."""

    ecg_id: int
    patient_id: int
    minimal_signal: np.ndarray
    accepted_signal: np.ndarray
    sample_mask: np.ndarray
    lead_mask: np.ndarray
    qc_result: Any
    morphology_result: Any
    eligibility: Eligibility
    provenance: Dict[str, Any]
    mi_label: int
    strat_fold: int
    hard_negative: bool = False
    label_metadata: Dict[str, Any] = field(default_factory=dict)

