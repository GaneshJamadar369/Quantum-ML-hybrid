import numpy as np

from run_residual_quantum_fusion_screen import (
    _residual_angles,
    _safe_logit,
    _source_matrix,
)


def test_safe_logit_is_finite_at_probability_boundaries():
    result = _safe_logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(result).all()
    assert result[0] < 0 < result[-1]


def test_residual_source_contracts():
    h = np.ones((5, 128), dtype=np.float32)
    clinical = np.ones((5, 106), dtype=np.float32)
    assert _source_matrix("h128_residual", h, clinical).shape == (5, 128)
    assert _source_matrix("clinical_residual", h, clinical).shape == (5, 106)
    assert _source_matrix("combined_residual", h, clinical).shape == (5, 234)


def test_residual_angles_are_fold_local_and_bounded():
    rng = np.random.default_rng(7)
    train = rng.normal(size=(100, 12))
    validation = rng.normal(size=(20, 12))
    residual = rng.normal(size=100)
    fit = np.arange(80)
    q_fit, q_val, audit = _residual_angles(
        train, validation, residual, fit, seed=11, n_components=4
    )
    assert q_fit.shape == (80, 4)
    assert q_val.shape == (20, 4)
    assert np.isfinite(q_fit).all() and np.isfinite(q_val).all()
    assert np.max(np.abs(q_fit)) <= np.pi + 1e-6
    assert audit["target_dim"] == 4
