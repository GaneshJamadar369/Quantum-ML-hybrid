import numpy as np

from run_train_reference_score_alignment import (
    _cross_fitted_calibration,
    _train_reference_scores,
)


def test_train_reference_scores_use_reference_distribution_only():
    reference = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    validation = np.array([-3.0, 0.0, 3.0])
    z_score, cdf_score, audit = _train_reference_scores(reference, validation)
    assert np.allclose(z_score, validation / np.std(reference))
    assert np.all(np.diff(cdf_score) > 0)
    assert np.all((cdf_score > 0) & (cdf_score < 1))
    assert audit["reference_mean"] == 0.0


def test_cross_fitted_calibration_is_complete_and_bounded():
    score = np.linspace(-2.0, 2.0, 80)
    labels = (score + np.sin(np.arange(80)) * 0.2 > 0).astype(int)
    folds = np.tile(np.arange(1, 9), 10)
    probability, audits = _cross_fitted_calibration(score, labels, folds)
    assert np.isfinite(probability).all()
    assert np.all((probability > 0) & (probability < 1))
    assert len(audits) == 8
