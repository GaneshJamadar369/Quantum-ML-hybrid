import json
from pathlib import Path

import numpy as np

from run_independent_dual_route_screen import (
    _cross_fitted_fusion,
    _fit_q4_representation,
    _safe_logit,
    load_route_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_route_b_is_a_complete_disjoint_partition():
    config = load_route_config(ROOT / "configs" / "independent_route_v1.json")
    manifest = json.loads(
        (ROOT / "configs" / "approved_feature_manifest_v0_4.json").read_text()
    )
    classical = config["route_b"]["classical_features"]
    quantum = config["route_b"]["quantum_features"]
    assert len(classical) == 60
    assert len(quantum) == 46
    assert set(classical).isdisjoint(quantum)
    assert set(classical) | set(quantum) == set(manifest["approved_features"])
    assert config["sealed_folds"] == [9, 10]


def test_direct_label_representation_is_train_only_and_bounded():
    rng = np.random.default_rng(12)
    features = rng.normal(size=(120, 10))
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    fit = np.arange(90)
    validation = np.arange(90, 120)
    train_q, validation_q, audit = _fit_q4_representation(
        features, labels, fit, validation, seed=12
    )
    assert train_q.shape == (90, 4)
    assert validation_q.shape == (30, 4)
    assert np.isfinite(train_q).all() and np.isfinite(validation_q).all()
    assert np.max(np.abs(train_q)) <= np.pi / 2 + 1e-7
    assert audit["fit_records"] == 90
    assert "residual" not in audit["method"].lower()


def test_cross_fitted_fusion_is_finite_and_nonnegative():
    rng = np.random.default_rng(23)
    folds = np.repeat(np.arange(1, 9), 40)
    labels = rng.integers(0, 2, size=len(folds))
    classical = (2 * labels - 1) * 0.8 + rng.normal(size=len(folds))
    quantum = (2 * labels - 1) * 0.4 + rng.normal(size=len(folds))
    fused, audits = _cross_fitted_fusion(classical, quantum, labels, folds)
    assert fused.shape == labels.shape
    assert np.isfinite(fused).all()
    assert len(audits) == 8
    assert all(item["beta_c"] >= 0 and item["beta_q"] >= 0 for item in audits)
    assert all(item["meta_fold"] in range(1, 9) for item in audits)


def test_safe_logit_remains_finite_at_probability_boundaries():
    values = _safe_logit(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(values).all()
