"""
Anti-leakage & split integrity tests.

Plan Section 16 — Leakage tests:
- Normalization statistics use training patients only
- QC and routing thresholds are fixed without fold 9 or fold 10
- Augmentation never runs on validation or test records
- Diagnostic metadata is excluded from predictive tensors
- No patient crosses dataset roles
"""

import numpy as np
import pytest

from aquire_preprocessing.config import (
    DEV_FOLDS,
    CALIBRATION_FOLD,
    LOCKED_TEST_FOLD,
    NUM_LEADS,
    SAMPLES_100HZ,
)
from aquire_preprocessing.normalization import LeadRobustScaler
from aquire_preprocessing.augmentation import (
    augment_amplitude_scale,
    augment_baseline_drift,
    augment_muscle_noise,
    apply_random_augmentation,
)
from aquire_preprocessing.manifest import FEATURE_DENYLIST_PATTERNS


# ---------------------------------------------------------------------------
# Normalization leakage tests
# ---------------------------------------------------------------------------

class TestNormalizationLeakage:
    """Verify normalizer is fitted strictly on training data."""

    def test_unfitted_transform_raises(self):
        """Transforming before fitting should raise RuntimeError."""
        scaler = LeadRobustScaler()
        signal = np.random.randn(NUM_LEADS, SAMPLES_100HZ).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(signal)

    def test_unfitted_inverse_raises(self):
        """Inverse transform before fitting should raise RuntimeError."""
        scaler = LeadRobustScaler()
        signal = np.random.randn(NUM_LEADS, SAMPLES_100HZ).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.inverse_transform(signal)

    def test_fit_uses_only_specified_folds(self):
        """Normalizer should only use signals from allowed folds."""
        rng = np.random.RandomState(42)
        n_records = 100
        signals = rng.randn(n_records, NUM_LEADS, SAMPLES_100HZ).astype(np.float32)
        folds = np.array([((i % 10) + 1) for i in range(n_records)])

        scaler = LeadRobustScaler()
        scaler.fit(signals, folds=folds, allowed_folds=DEV_FOLDS)

        assert scaler.params.is_fitted
        assert scaler.params.folds_used == sorted(DEV_FOLDS)
        # Verify fold 9 and 10 were NOT used
        assert CALIBRATION_FOLD not in scaler.params.folds_used
        assert LOCKED_TEST_FOLD not in scaler.params.folds_used

    def test_inverse_recovers_original(self):
        """Inverse transform should recover the original signal."""
        rng = np.random.RandomState(42)
        signals = rng.randn(50, NUM_LEADS, SAMPLES_100HZ).astype(np.float32)
        folds = np.array([((i % 8) + 1) for i in range(50)])

        scaler = LeadRobustScaler()
        scaler.fit(signals, folds=folds)

        original = signals[0].copy()
        normalized = scaler.transform(original)
        recovered = scaler.inverse_transform(normalized)

        np.testing.assert_allclose(recovered, original, atol=1e-4)


# ---------------------------------------------------------------------------
# Augmentation leakage tests
# ---------------------------------------------------------------------------

class TestAugmentationLeakage:
    """Verify augmentation is blocked on non-training folds."""

    def _make_signal(self):
        return np.random.randn(NUM_LEADS, SAMPLES_100HZ).astype(np.float32)

    def test_augmentation_blocked_on_fold_9(self):
        """Augmentation on fold 9 (validation) should raise ValueError."""
        signal = self._make_signal()
        with pytest.raises(ValueError, match="AUGMENTATION BLOCKED"):
            augment_amplitude_scale(signal, ecg_id=1, fold=9, seed=42)

    def test_augmentation_blocked_on_fold_10(self):
        """Augmentation on fold 10 (locked test) should raise ValueError."""
        signal = self._make_signal()
        with pytest.raises(ValueError, match="AUGMENTATION BLOCKED"):
            augment_baseline_drift(signal, ecg_id=1, fold=10, seed=42)

    def test_augmentation_allowed_on_dev_folds(self):
        """Augmentation on dev folds (1-8) should succeed."""
        signal = self._make_signal()
        for fold in DEV_FOLDS:
            augmented, record = augment_amplitude_scale(
                signal, ecg_id=1, fold=fold, seed=42
            )
            assert augmented.shape == signal.shape
            assert record.source_ecg_id == 1

    def test_random_augmentation_blocked_on_test(self):
        """Composite random augmentation blocked on test fold."""
        signal = self._make_signal()
        with pytest.raises(ValueError, match="AUGMENTATION BLOCKED"):
            apply_random_augmentation(
                signal, ecg_id=1, fold=LOCKED_TEST_FOLD, seed=42
            )

    def test_augmentation_provenance_tracked(self):
        """Every augmentation must produce a provenance record."""
        signal = self._make_signal()
        augmented, record = augment_muscle_noise(
            signal, ecg_id=42, fold=1, seed=123
        )
        assert record.source_ecg_id == 42
        assert record.corruption_type == "muscle_noise"
        assert record.random_seed == 123
        assert record.augmentation_version is not None


# ---------------------------------------------------------------------------
# Feature denylist tests
# ---------------------------------------------------------------------------

class TestFeatureDenylist:
    """Verify diagnostic metadata is excluded from model inputs."""

    def test_denylist_catches_diagnostic_columns(self):
        """Columns containing diagnostic info should be denied."""
        from aquire_preprocessing.manifest import _is_denied_column

        assert _is_denied_column("diagnostic_class")
        assert _is_denied_column("scp_code_primary")
        assert _is_denied_column("report_text")
        assert _is_denied_column("infarction_stage")
        assert _is_denied_column("statement_summary")

    def test_denylist_allows_measurement_columns(self):
        """Measurement columns should NOT be denied."""
        from aquire_preprocessing.manifest import _is_denied_column

        assert not _is_denied_column("heart_rate")
        assert not _is_denied_column("qrs_duration")
        assert not _is_denied_column("p_wave_axis")
        assert not _is_denied_column("rr_interval")
        assert not _is_denied_column("st_level_v2")


# ---------------------------------------------------------------------------
# Patient split integrity tests
# ---------------------------------------------------------------------------

class TestPatientSplitIntegrity:
    """Verify fold definitions are correct and non-overlapping."""

    def test_dev_folds_are_1_through_8(self):
        assert DEV_FOLDS == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_calibration_fold_is_9(self):
        assert CALIBRATION_FOLD == 9

    def test_locked_test_fold_is_10(self):
        assert LOCKED_TEST_FOLD == 10

    def test_no_fold_overlap(self):
        all_folds = set(DEV_FOLDS) | {CALIBRATION_FOLD} | {LOCKED_TEST_FOLD}
        assert len(all_folds) == 10
        assert all_folds == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
