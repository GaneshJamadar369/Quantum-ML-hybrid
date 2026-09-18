import sys
from types import SimpleNamespace

import numpy as np
import pytest

from aquire_preprocessing.features import extract_deployable_features


def _synthetic_ecg() -> np.ndarray:
    signal = np.zeros((12, 1000), dtype=np.float32)
    for peak in range(100, 1000, 100):
        signal[:, peak - 5:peak + 9] = -0.2
        signal[:, peak] = 1.0
        signal[:, peak + 14] = 0.15  # ST60: QRS offset (peak+8) + 6 samples.
        signal[:, peak + 25] = 0.3
    return signal


def _fake_neurokit() -> SimpleNamespace:
    peaks = np.arange(100, 1000, 100)
    waves = {
        "ECG_P_Onsets": peaks - 20,
        "ECG_R_Onsets": peaks - 5,
        "ECG_R_Offsets": peaks + 8,
        "ECG_T_Offsets": peaks + 35,
    }
    return SimpleNamespace(
        ecg_clean=lambda signal, sampling_rate, method: signal,
        ecg_peaks=lambda signal, sampling_rate: (None, {"ECG_R_Peaks": peaks}),
        ecg_delineate=lambda signal, rpeaks, sampling_rate, method, show: (None, waves),
    )


def test_validated_extractor_uses_common_delineated_beats(monkeypatch):
    monkeypatch.setitem(sys.modules, "neurokit2", _fake_neurokit())
    bundle = extract_deployable_features(
        _synthetic_ecg(), 100, 7, require_delineation=True
    )
    assert bundle.extractor == "aquire-local-v0.4.0"
    assert bundle.values["rr_median_ms"] == pytest.approx(1000.0)
    assert bundle.values["pr_interval_ms"] == pytest.approx(150.0)
    assert bundle.values["qrs_duration_ms"] == pytest.approx(130.0)
    assert bundle.values["qt_interval_ms"] == pytest.approx(400.0)
    assert bundle.values["ii__r_amp_mv"] == pytest.approx(1.0)
    assert bundle.values["ii__s_amp_mv"] == pytest.approx(-0.2)
    assert bundle.values["ii__st60_mv"] == pytest.approx(0.15)
    assert bundle.values["ii__t_polarity"] == 1.0
    assert not bundle.failures


def test_r_amplitude_is_signed_at_common_fiducial(monkeypatch):
    monkeypatch.setitem(sys.modules, "neurokit2", _fake_neurokit())
    signal = _synthetic_ecg()
    signal[6] = 0.0  # V1
    for peak in range(100, 1000, 100):
        signal[6, peak - 5:peak + 9] = -1.0
        signal[6, peak] = -0.8
        signal[6, peak + 2] = 0.2
    bundle = extract_deployable_features(signal, 100, 10, require_delineation=True)
    assert bundle.values["v1__r_amp_mv"] == pytest.approx(-0.8)


def test_fallback_never_fabricates_intervals(monkeypatch):
    monkeypatch.setitem(sys.modules, "neurokit2", None)
    bundle = extract_deployable_features(
        _synthetic_ecg(), 100, 8, require_delineation=False
    )
    assert "neurokit2_not_installed" in bundle.failures
    assert "unvalidated_fallback_r_peaks" in bundle.failures
    for feature in [
        "pr_interval_ms", "qrs_duration_ms", "qt_interval_ms",
        "qtc_bazett_ms", "qtc_fridericia_ms", "qtc_framingham_ms",
    ]:
        assert np.isnan(bundle.values[feature])


def test_strict_extraction_requires_neurokit(monkeypatch):
    monkeypatch.setitem(sys.modules, "neurokit2", None)
    with pytest.raises(RuntimeError, match="NeuroKit2 is required"):
        extract_deployable_features(
            _synthetic_ecg(), 100, 9, require_delineation=True
        )
