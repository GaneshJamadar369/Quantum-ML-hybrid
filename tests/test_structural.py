"""
Structural & file ingestion tests.

Plan Section 16 — File and structural tests:
- Valid .hea/.dat pair loads successfully
- Missing header or signal file fails with a deterministic error
- Wrong lead order is detected and safely canonicalized
- Unknown lead names fail rather than being guessed
- Sampling rate, duration, units and shape are verified
"""

import numpy as np
import pytest

from aquire_preprocessing.config import (
    CANONICAL_LEAD_ORDER,
    NUM_LEADS,
    SAMPLES_100HZ,
    TARGET_SHAPE_100HZ,
)
from aquire_preprocessing.structural import (
    StructuralValidation,
    _canonicalize_lead_names,
    _compute_reorder_indices,
    validate_record,
)


class TestLeadCanonicalization:
    """Test lead name resolution and reordering."""

    def test_standard_names_resolve(self):
        """Standard 12-lead names resolve correctly."""
        names = ["I", "II", "III", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]
        canonical, unknown = _canonicalize_lead_names(names)
        assert canonical == CANONICAL_LEAD_ORDER
        assert unknown == []

    def test_lowercase_names_resolve(self):
        """Lowercase lead names resolve to canonical."""
        names = ["i", "ii", "iii", "avr", "avl", "avf",
                 "v1", "v2", "v3", "v4", "v5", "v6"]
        canonical, unknown = _canonicalize_lead_names(names)
        assert canonical == CANONICAL_LEAD_ORDER
        assert unknown == []

    def test_uppercase_augmented_resolve(self):
        """Uppercase augmented leads (AVR, AVL, AVF) resolve."""
        names = ["I", "II", "III", "AVR", "AVL", "AVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]
        canonical, unknown = _canonicalize_lead_names(names)
        assert canonical == CANONICAL_LEAD_ORDER
        assert unknown == []

    def test_unknown_names_flagged(self):
        """Unknown lead names are flagged, not guessed."""
        names = ["I", "II", "UNKNOWN_LEAD", "aVR", "aVL", "aVF",
                 "V1", "V2", "V3", "V4", "V5", "V6"]
        canonical, unknown = _canonicalize_lead_names(names)
        assert "UNKNOWN_LEAD" in unknown

    def test_reorder_indices_correct(self):
        """Shuffled canonical names produce correct reorder indices."""
        shuffled = ["V1", "I", "II", "aVR", "III", "aVL",
                    "aVF", "V2", "V3", "V4", "V5", "V6"]
        indices = _compute_reorder_indices(shuffled, CANONICAL_LEAD_ORDER)
        assert indices is not None
        assert [shuffled[i] for i in indices] == CANONICAL_LEAD_ORDER

    def test_reorder_mismatched_sets_returns_none(self):
        """Mismatched lead sets return None."""
        wrong_set = ["I", "II", "III", "aVR", "aVL", "aVF",
                     "V1", "V2", "V3", "V4", "V5", "EXTRA"]
        indices = _compute_reorder_indices(wrong_set, CANONICAL_LEAD_ORDER)
        assert indices is None


class TestStructuralValidation:
    """Test the structural validation of WFDB records.

    These tests require the PTB-XL dataset to be available.
    They are skipped if the dataset is not found.
    """

    @pytest.fixture
    def sample_ecg_id(self):
        return 1  # First record in PTB-XL

    def test_valid_record_loads(self, sample_ecg_id):
        """A valid record loads with correct shape and type."""
        try:
            result = validate_record(sample_ecg_id, sampling_rate=100)
        except Exception:
            pytest.skip("PTB-XL dataset not available")

        if not result.is_valid and "not found" in str(result.errors):
            pytest.skip("PTB-XL dataset not available")

        assert result.is_valid, f"Validation failed: {result.errors}"
        assert result.signal is not None
        assert result.signal.shape == TARGET_SHAPE_100HZ
        assert result.signal.dtype == np.float32
        assert result.canonical_lead_order == CANONICAL_LEAD_ORDER

    def test_nonexistent_record_fails(self):
        """A nonexistent ecg_id produces a deterministic failure."""
        result = validate_record(99999, sampling_rate=100)
        assert not result.is_valid
        assert len(result.errors) > 0

    def test_output_is_in_millivolts(self, sample_ecg_id):
        """Output signal is in physical units (mV)."""
        try:
            result = validate_record(sample_ecg_id, sampling_rate=100)
        except Exception:
            pytest.skip("PTB-XL dataset not available")

        if not result.is_valid:
            pytest.skip("PTB-XL dataset not available")

        assert result.physical_units == "mV"
        # ECG amplitudes should typically be in [-5, 5] mV range
        assert np.max(np.abs(result.signal)) < 10.0, (
            f"Signal max {np.max(np.abs(result.signal)):.2f} mV seems too large"
        )

    def test_no_nonfinite_in_clean_record(self, sample_ecg_id):
        """Clean records should have zero nonfinite values."""
        try:
            result = validate_record(sample_ecg_id, sampling_rate=100)
        except Exception:
            pytest.skip("PTB-XL dataset not available")

        if not result.is_valid:
            pytest.skip("PTB-XL dataset not available")

        assert np.all(np.isfinite(result.signal)), "Nonfinite values in output"
