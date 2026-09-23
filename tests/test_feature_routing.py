import numpy as np
import pandas as pd
import pytest

from aquire_preprocessing.feature_routing import (
    binary_log_loss_per_record,
    feature_family,
    route_feature_hypotheses,
)


def test_feature_family_maps_ecg_groups():
    assert feature_family("heart_rate_bpm") == "rhythm"
    assert feature_family("ii__st60_mv") == "lead_st"
    assert feature_family("v3__rs_ratio") == "lead_qrs_morphology"
    assert feature_family("clinical__inferior__st_mean_mv") == "territorial_st_pattern"


def test_binary_log_loss_is_recordwise_and_finite():
    value = binary_log_loss_per_record(np.array([0, 1]), np.array([0.0, 1.0]))
    assert value.shape == (2,)
    assert np.isfinite(value).all()
    assert np.max(value) < 1e-4


def test_routing_rejects_locked_folds():
    n = 80
    frame = pd.DataFrame({"heart_rate_bpm": np.linspace(50, 100, n)})
    labels = np.tile([0, 1], n // 2)
    with pytest.raises(ValueError, match="folds 1-8"):
        route_feature_hypotheses(
            frame, labels, np.full(n, 9), np.full(n, 0.5), np.full(n, 0.5),
            approved_features=["heart_rate_bpm"],
        )


def test_excluded_measurement_cannot_be_routed():
    n = 160
    frame = pd.DataFrame({"heart_rate_bpm": np.linspace(50, 100, n)})
    labels = np.tile([0, 1], n // 2)
    folds = np.tile(np.arange(1, 9), n // 8)
    result = route_feature_hypotheses(
        frame, labels, folds, np.full(n, 0.5), np.full(n, 0.5),
        approved_features=[],
        excluded_features={"heart_rate_bpm": "failed measurement gate"},
    )
    assert result.iloc[0].route == "EXCLUDE_MEASUREMENT_INVALID"
    assert "failed measurement" in result.iloc[0].route_reason
