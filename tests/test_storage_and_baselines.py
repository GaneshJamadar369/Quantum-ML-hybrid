from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aquire_preprocessing.contracts import Eligibility, ProcessedECG
from aquire_preprocessing.normalization import fit_fold_local_scalers_from_hdf5
from aquire_preprocessing.storage import HDF5RecordWriter


h5py = pytest.importorskip("h5py")


def _record(ecg_id: int, fold: int, value: float = 0.1) -> ProcessedECG:
    signal = np.full((12, 1000), value, dtype=np.float32)
    return ProcessedECG(
        ecg_id=ecg_id, patient_id=1000 + ecg_id,
        minimal_signal=signal, accepted_signal=signal,
        sample_mask=np.ones_like(signal, dtype=bool), lead_mask=np.ones(12, dtype=bool),
        qc_result=SimpleNamespace(qc_status="PASS"), morphology_result=None,
        eligibility=Eligibility.PRIMARY,
        provenance={"source_checksum": "a" * 64, "pipeline_version": "test"},
        mi_label=ecg_id % 2, strat_fold=fold,
    )


def test_hdf5_atomic_resume_and_completed_file_protection(tmp_path):
    output = tmp_path / "records.h5"
    writer = HDF5RecordWriter(output, 1000)
    writer.append(_record(1, 1))
    writer.close(commit=False)
    assert not output.exists()
    resumed = HDF5RecordWriter(output, 1000)
    assert resumed.count == 1
    resumed.append(_record(2, 2))
    resumed.close(commit=True)
    with h5py.File(output, "r") as handle:
        assert handle["ecg_id"][:].tolist() == [1, 2]
        assert int(handle.attrs["committed_count"]) == 2
        assert handle["accepted_checksum"].shape == (2,)
    with pytest.raises(FileExistsError):
        HDF5RecordWriter(output, 1000)


def test_fold_local_hdf5_normalizers_exclude_masks_and_holdout(tmp_path):
    output = tmp_path / "records.h5"
    writer = HDF5RecordWriter(output, 1000)
    for fold in range(1, 9):
        record = _record(fold, fold, value=float(fold))
        if fold == 1:
            record.accepted_signal[0, 0] = 1e6
            record.sample_mask[0, 0] = False
        writer.append(record)
    writer.close(commit=True)
    scalers = fit_fold_local_scalers_from_hdf5(output, tmp_path / "normalizers", samples_per_record=50)
    assert set(scalers) == set(range(1, 9))
    assert scalers[1].params.folds_used == list(range(2, 9))
    assert scalers[1].params.lead_medians["I"] < 10
    assert scalers[1].params.training_patient_checksum


def test_oof_artifact_contains_required_fields(tmp_path, monkeypatch):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler
    from aquire_preprocessing import baselines

    monkeypatch.setattr(
        baselines, "_models",
        lambda seed=42: {"logistic": make_pipeline(SimpleImputer(), RobustScaler(), LogisticRegression())},
    )
    folds = np.repeat(np.arange(1, 9), 4)
    labels = np.tile([0, 0, 1, 1], 8)
    rng = np.random.default_rng(7)
    features = pd.DataFrame({
        "signal_feature": labels + rng.normal(0, 0.1, len(labels)),
        "redundant_feature": labels + rng.normal(0, 0.1, len(labels)),
    }, index=np.arange(100, 100 + len(labels)))
    metrics, predictions = baselines.run_oof_baselines(
        features, labels, folds, np.arange(len(labels)), tmp_path,
        qc_groups=np.where(labels, "WARN", "PASS"), hard_negative=~labels.astype(bool),
    )
    assert {"auprc", "brier", "calibration_error"}.issubset(metrics.columns)
    assert {"ecg_id", "logit", "probability", "qc_group", "hard_negative"}.issubset(predictions.columns)
    assert predictions.probability.notna().all()
    assert (tmp_path / "fold_local_feature_selection.json").exists()
