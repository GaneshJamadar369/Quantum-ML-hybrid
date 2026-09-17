"""
Central configuration for the AQUIRE-Med preprocessing pipeline.

All frozen constants, path detection, canonical lead order, QC thresholds,
morphology tolerances, and fold assignments live here.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# ---------------------------------------------------------------------------
# 1. Environment detection & dataset paths
# ---------------------------------------------------------------------------

def _detect_environment() -> str:
    """Detect whether we are running on Kaggle or locally."""
    if os.path.exists("/kaggle/input"):
        return "kaggle"
    return "local"


ENVIRONMENT = _detect_environment()

# PTB-XL v1.0.3 paths
_PTBXL_KAGGLE = os.environ.get(
    "AQUIRE_PTBXL_ROOT",
    "/kaggle/input/ptb-xl-1-0-3/ptb-xl-a-large-publicly-available-"
    "electrocardiography-dataset-1.0.3",
)
_PTBXL_LOCAL = "./data/ptb-xl-1.0.3"

# PTB-XL+ v1.0.1 paths
_PTBXLP_KAGGLE = os.environ.get(
    "AQUIRE_PTBXLP_ROOT",
    "/kaggle/input/ptb-xl-plus-1-0-1/"
    "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1",
)
_PTBXLP_LOCAL = "./data/ptb-xl-plus-1.0.1"

# Output paths
_OUTPUT_KAGGLE = os.environ.get("AQUIRE_OUTPUT_ROOT", "/kaggle/working/processed")
_OUTPUT_LOCAL = os.environ.get("AQUIRE_OUTPUT_ROOT", "./data/processed")


@dataclass(frozen=True)
class Paths:
    """Immutable path configuration resolved at import time."""

    ptbxl_root: Path
    ptbxlp_root: Path
    output_root: Path

    # Derived PTB-XL paths
    @property
    def ptbxl_database_csv(self) -> Path:
        return self.ptbxl_root / "ptbxl_database.csv"

    @property
    def scp_statements_csv(self) -> Path:
        return self.ptbxl_root / "scp_statements.csv"

    @property
    def records100_dir(self) -> Path:
        return self.ptbxl_root / "records100"

    @property
    def records500_dir(self) -> Path:
        return self.ptbxl_root / "records500"

    @property
    def sha256sums(self) -> Path:
        return self.ptbxl_root / "SHA256SUMS.txt"

    # Derived PTB-XL+ paths
    @property
    def features_12sl_csv(self) -> Path:
        return self.ptbxlp_root / "features" / "12sl_features.csv"

    @property
    def features_ecgdeli_csv(self) -> Path:
        return self.ptbxlp_root / "features" / "ecgdeli_features.csv"

    @property
    def features_unig_csv(self) -> Path:
        return self.ptbxlp_root / "features" / "unig_features.csv"

    @property
    def feature_description_csv(self) -> Path:
        return self.ptbxlp_root / "features" / "feature_description.csv"

    # Output paths
    @property
    def manifests_dir(self) -> Path:
        return self.output_root / "manifests"

    @property
    def quality_reports_dir(self) -> Path:
        return self.output_root / "quality_reports"

    @property
    def view_minimal_dir(self) -> Path:
        return self.output_root / "view_100hz_minimal"

    @property
    def view_corrected_dir(self) -> Path:
        return self.output_root / "view_100hz_corrected"


def get_paths() -> Paths:
    """Return the path config for the detected environment."""
    if ENVIRONMENT == "kaggle":
        return Paths(
            ptbxl_root=Path(_PTBXL_KAGGLE),
            ptbxlp_root=Path(_PTBXLP_KAGGLE),
            output_root=Path(_OUTPUT_KAGGLE),
        )
    return Paths(
        ptbxl_root=Path(_PTBXL_LOCAL),
        ptbxlp_root=Path(_PTBXLP_LOCAL),
        output_root=Path(_OUTPUT_LOCAL),
    )


PATHS = get_paths()

# ---------------------------------------------------------------------------
# 2. Canonical lead order and signal shape
# ---------------------------------------------------------------------------

CANONICAL_LEAD_ORDER: List[str] = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
]

# Acceptable WFDB lead name variants → canonical name
LEAD_NAME_ALIASES: Dict[str, str] = {
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3",
    "v4": "V4", "v5": "V5", "v6": "V6",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "I": "I", "II": "II", "III": "III",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3",
    "V4": "V4", "V5": "V5", "V6": "V6",
}

NUM_LEADS: int = 12

# Target tensor shapes
SAMPLING_RATE_100HZ: int = 100
SAMPLING_RATE_500HZ: int = 500
DURATION_SECONDS: float = 10.0
SAMPLES_100HZ: int = int(SAMPLING_RATE_100HZ * DURATION_SECONDS)   # 1000
SAMPLES_500HZ: int = int(SAMPLING_RATE_500HZ * DURATION_SECONDS)   # 5000

TARGET_SHAPE_100HZ = (NUM_LEADS, SAMPLES_100HZ)   # (12, 1000)
TARGET_SHAPE_500HZ = (NUM_LEADS, SAMPLES_500HZ)   # (12, 5000)

# ---------------------------------------------------------------------------
# 3. Pipeline version for reproducibility
# ---------------------------------------------------------------------------

PIPELINE_VERSION: str = "aquire-preproc-v0.2.0"
DATASET_VERSION_PTBXL: str = "1.0.3"
DATASET_VERSION_PTBXLP: str = "1.0.1"

# ---------------------------------------------------------------------------
# 4. Quality-control thresholds (frozen before any modeling)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QCThresholds:
    """Frozen signal-quality thresholds.

    These must be set from domain knowledge and training-fold analysis
    before touching Fold 9 or Fold 10.
    """

    # Lead-level checks
    flatline_min_duration_ms: float = 200.0       # ≥200 ms of zero variance → flat
    flatline_variance_eps: float = 1e-6            # variance below this = flat
    clipping_fraction_max: float = 0.01            # >1% samples at ADC rail = clipped
    missing_fraction_max: float = 0.05             # >5% NaN samples = FAIL
    amplitude_min_mv: float = 0.05                 # below = implausible (disconnected)
    amplitude_max_mv: float = 6.0                  # above = implausible (gain error)

    # Baseline wander
    baseline_wander_power_ratio_max: float = 0.4   # low-freq (<0.5 Hz) / total power

    # Powerline interference
    powerline_snr_db_min: float = 10.0             # 50/60 Hz peak SNR threshold

    # High-frequency noise
    hf_noise_power_ratio_max: float = 0.3          # energy >40 Hz / total power

    # Cross-lead physics (Einthoven / Goldberger)
    einthoven_residual_max_mv: float = 0.15        # mean |II - (I + III)|
    goldberger_residual_max_mv: float = 0.15       # mean residuals for aVR, aVL, aVF

    # Short gap interpolation
    max_interpolatable_gap_ms: float = 50.0        # gaps > this → mask, not interpolate

    # QRS consistency
    qrs_rr_cv_max: float = 0.25                    # coefficient of variation of RR intervals

    # Structural repair is deliberately narrow. Larger mismatches fail.
    max_length_adjustment_ms: float = 20.0


QC = QCThresholds()

# ---------------------------------------------------------------------------
# 5. Morphology-preservation gate tolerances
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MorphologyTolerances:
    """Frozen tolerances for the preprocessing utility gate U_P.

    If correction violates any of these, it is rejected and
    the minimal (View A) signal is retained.
    """

    max_rpeak_shift_ms: float = 10.0
    max_qrs_width_change_ms: float = 10.0          # milliseconds
    max_qrs_amplitude_change_mv: float = 0.10      # millivolts
    max_st_level_shift_mv: float = 0.05            # millivolts (critical for MI)
    twave_polarity_must_match: bool = True          # polarity inversion = rejection

    # Utility gate weights
    lambda_m: float = 2.0    # morphology distortion penalty
    lambda_f: float = 1.0    # failure rate penalty
    lambda_t: float = 0.1    # processing time penalty


MORPHOLOGY = MorphologyTolerances()

# ---------------------------------------------------------------------------
# 6. Fold assignments
# ---------------------------------------------------------------------------

DEV_FOLDS: List[int] = [1, 2, 3, 4, 5, 6, 7, 8]
CALIBRATION_FOLD: int = 9
LOCKED_TEST_FOLD: int = 10

# ---------------------------------------------------------------------------
# 7. MI superclass SCP code mapping
# ---------------------------------------------------------------------------

# Used only as a release-integrity assertion. Runtime labels are derived from
# scp_statements.csv where diagnostic_class == "MI".
EXPECTED_MI_SCP_CODES: List[str] = [
    "AMI", "ALMI", "ASMI", "ILMI", "IMI", "INJAL", "INJAS",
    "INJIL", "INJIN", "INJLA", "IPLMI", "IPMI", "LMI", "PMI",
]

# Compatibility alias. Label construction must still derive and validate this
# set dynamically from the mounted release.
MI_SCP_CODES = EXPECTED_MI_SCP_CODES

HARD_NEGATIVE_DIAGNOSTIC_CLASSES = ["STTC", "CD", "HYP"]
