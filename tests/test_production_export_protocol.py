from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from train_export_hybrid_prototype import (
    _load_split,
    _select_fusion_c,
    _threshold_at_specificity,
)


def test_export_loader_rejects_fold_10(tmp_path):
    h5py = pytest.importorskip("h5py")
    hdf5 = tmp_path / "locked.h5"
    with h5py.File(hdf5, "w") as handle:
        handle.create_dataset("ecg_id", data=[7])
        handle.create_dataset("accepted_signal", data=np.zeros((1, 12, 1000), np.float32))
        handle.create_dataset("sample_mask", data=np.ones((1, 12, 1000), bool))
        handle.create_dataset("lead_mask", data=np.ones((1, 12), bool))
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {"ecg_id": [7], "patient_id": [17], "strat_fold": [10],
         "mi_label": [0], "hard_negative": [False], "eligibility": ["PRIMARY"]}
    ).to_csv(metadata, index=False)
    with pytest.raises(PermissionError, match="Unexpected folds"):
        _load_split(hdf5, metadata, {9})


def test_final_fusion_fit_keeps_both_weights_nonnegative():
    rng = np.random.default_rng(42)
    labels = np.tile([0, 1], 80)
    folds = np.tile(np.arange(1, 9), 20)
    classical = np.clip(0.15 + 0.65 * labels + rng.normal(0, 0.12, len(labels)), 0.01, 0.99)
    quantum = np.clip(0.20 + 0.55 * labels + rng.normal(0, 0.15, len(labels)), 0.01, 0.99)
    selected_c, losses, raw = _select_fusion_c(classical, quantum, labels, folds)
    from run_independent_dual_route_screen import _fit_nonnegative_logistic

    mean, scale = raw.mean(0), raw.std(0).clip(1e-6)
    parameters = _fit_nonnegative_logistic((raw - mean) / scale, labels, selected_c)
    assert set(losses) == {0.1, 1.0, 10.0}
    assert np.all(parameters[1:] >= 0)
    assert np.all(parameters[1:] > 0), "Synthetic complementary routes should both remain active"


def test_operating_threshold_meets_prespecified_specificity():
    labels = np.array([0] * 10 + [1] * 10)
    probability = np.array([0.02, .05, .08, .10, .12, .18, .22, .28, .35, .65,
                            .20, .30, .40, .50, .60, .70, .75, .80, .90, .98])
    threshold, sensitivity, specificity = _threshold_at_specificity(labels, probability, 0.90)
    assert 0 < threshold < 1
    assert specificity >= 0.90
    assert sensitivity > 0
