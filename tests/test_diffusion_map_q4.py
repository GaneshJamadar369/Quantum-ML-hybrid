import numpy as np
import pandas as pd

from aquire_preprocessing.diffusion_map import (
    LandmarkDiffusionMap,
    select_patient_unique_landmarks,
)
from run_diffusion_map_q4_screen import _load_pls_reference


def _data(seed=4):
    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, 180)
    values = np.column_stack(
        [np.sin((index + 1) * phase) + rng.normal(0, 0.02, len(phase)) for index in range(12)]
    )
    patients = np.repeat(np.arange(90), 2)
    return values, patients


def test_landmarks_are_patient_unique_and_deterministic():
    _, patients = _data()
    first = select_patient_unique_landmarks(patients, 48, 17)
    second = select_patient_unique_landmarks(patients, 48, 17)
    assert np.array_equal(first, second)
    assert len(np.unique(patients[first])) == len(first) == 48


def test_diffusion_map_is_finite_bounded_and_inductive():
    values, patients = _data()
    train, validation = values[:140], values[140:]
    mapper = LandmarkDiffusionMap(
        n_components=4,
        n_landmarks=64,
        graph_neighbors=14,
        bandwidth_neighbors=8,
        random_state=9,
    ).fit(train, patients[:140])
    train_q = mapper.transform(train)
    val_q = mapper.transform(validation)
    val_q_with_outlier = mapper.transform(np.vstack([validation, np.full((1, 12), 1e6)]))[:-1]
    assert train_q.shape == (140, 4)
    assert val_q.shape == (40, 4)
    assert np.isfinite(train_q).all() and np.isfinite(val_q).all()
    assert np.max(np.abs(train_q)) <= np.pi / 2 + 1e-6
    assert np.max(np.abs(val_q)) <= np.pi / 2 + 1e-6
    assert np.allclose(val_q, val_q_with_outlier)
    assert mapper.fit_audit_["n_training_records"] == 140
    assert mapper.fit_audit_["n_landmarks"] == 64


def test_diffusion_map_rejects_nonfinite_input():
    values, patients = _data()
    values[0, 0] = np.nan
    mapper = LandmarkDiffusionMap(n_landmarks=32)
    try:
        mapper.fit(values, patients)
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite input was accepted")


def test_pls_reference_alignment_is_strict(tmp_path):
    joined = pd.DataFrame(
        {
            "patient_id": [10, 20],
            "strat_fold": [1, 2],
            "mi_label": [0, 1],
        },
        index=pd.Index([100, 200], name="ecg_id"),
    )
    reference = pd.DataFrame(
        {
            "ecg_id": [100, 200],
            "patient_id": [10, 20],
            "strat_fold": [1, 2],
            "y_true": [0, 1],
            "vqc_train_cdf": [0.2, 0.8],
            "q4_mlp_ensemble": [0.3, 0.7],
            "fusion_vqc_train_cdf": [0.25, 0.75],
            "fusion_q4_mlp": [0.35, 0.65],
        }
    )
    path = tmp_path / "reference.csv"
    reference.to_csv(path, index=False)
    loaded = _load_pls_reference(path, joined)
    assert loaded.index.tolist() == [100, 200]

