import numpy as np
import pandas as pd
import pytest

from run_neurips_q4_screen import _load_reference, _model


def _joined():
    return pd.DataFrame(
        {
            "patient_id": [10, 20],
            "strat_fold": [1, 2],
            "mi_label": [0, 1],
        },
        index=pd.Index([101, 102], name="ecg_id"),
    )


def _reference():
    values = {
        "ecg_id": [101, 102],
        "patient_id": [10, 20],
        "strat_fold": [1, 2],
        "y_true": [0, 1],
        "vqc_train_cdf": [0.2, 0.8],
        "q4_logistic": [0.3, 0.7],
        "q4_mlp_ensemble": [0.25, 0.75],
        "clinical_teacher": [0.1, 0.9],
        "fusion_vqc_train_cdf": [0.15, 0.85],
        "fusion_q4_mlp": [0.12, 0.88],
    }
    values.update({f"q4_{index}": [0.0, 1.0] for index in range(4)})
    return pd.DataFrame(values)


def test_retained_reference_is_aligned_by_ecg_id(tmp_path):
    path = tmp_path / "reference.csv"
    _reference().iloc[::-1].to_csv(path, index=False)
    observed = _load_reference(path, _joined())
    assert observed.index.tolist() == [101, 102]
    assert np.array_equal(observed.y_true.to_numpy(), [0, 1])


def test_retained_reference_rejects_label_mismatch(tmp_path):
    frame = _reference()
    frame.loc[0, "y_true"] = 1
    path = tmp_path / "reference.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="y_true"):
        _load_reference(path, _joined())


def test_screen_model_factory_has_distinct_arms():
    frequency = _model("frequency", steps=3, damping=0.5, device="cpu")
    equilibrium = _model("equilibrium", steps=3, damping=0.5, device="cpu")
    assert type(frequency) is not type(equilibrium)
    with pytest.raises(ValueError):
        _model("unknown", steps=3, damping=0.5, device="cpu")
