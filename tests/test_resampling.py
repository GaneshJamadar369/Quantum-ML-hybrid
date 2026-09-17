import numpy as np
import pytest

from aquire_preprocessing.resampling import resample_500_to_100, resample_corrected_pair


def _ecg_500():
    t = np.arange(5000) / 500.0
    base = 0.2 * np.sin(2 * np.pi * 5 * t)
    return np.stack([base * (1 + i / 20) for i in range(12)]).astype(np.float32)


def test_500_to_100_shape_mask_and_alias_suppression():
    signal = _ecg_500()
    t = np.arange(5000) / 500.0
    signal += (0.1 * np.sin(2 * np.pi * 120 * t))[None, :]
    mask = np.ones_like(signal, dtype=bool)
    mask[1, 10] = False
    output, output_mask, leads = resample_500_to_100(signal, mask)
    assert output.shape == (12, 1000)
    assert output_mask.shape == output.shape
    assert not output_mask[1, 2]
    assert leads.all()
    # The 120 Hz component would alias to 20 Hz without anti-alias filtering.
    clean, _, _ = resample_500_to_100(_ecg_500())
    assert np.std(output - clean) < 0.01


def test_resampling_rejects_wrong_shape_and_nonfinite():
    with pytest.raises(ValueError):
        resample_500_to_100(np.zeros((12, 1000)))
    signal = _ecg_500(); signal[0, 0] = np.nan
    with pytest.raises(ValueError):
        resample_500_to_100(signal)


def test_corrected_pair_falls_back_when_gate_indeterminate():
    signal = np.zeros((12, 5000), dtype=np.float32)
    # A flat record has no valid QRS delineation; a differing correction must
    # therefore be rejected instead of trusted.
    result = resample_corrected_pair(signal, signal + 0.01, 1)
    assert not result.accepted
    assert np.allclose(result.signal_100hz, resample_500_to_100(signal)[0])
