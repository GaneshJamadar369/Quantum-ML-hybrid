import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from aquire_preprocessing.config import CANONICAL_LEAD_ORDER, EXPECTED_MI_SCP_CODES
from aquire_preprocessing.manifest import build_mi_label, derive_mi_codes, guard_fold_access
from aquire_preprocessing.morphology_gate import compare_lead_morphology
from aquire_preprocessing.qc_calibration import calibrate_qc_threshold, annotation_present_for_lead
from aquire_preprocessing.quality import assess_lead_quality, check_cross_lead_physics
from aquire_preprocessing.structural import validate_record


def _scp_table():
    return pd.DataFrame(
        {
            "diagnostic_class": ["MI"] * len(EXPECTED_MI_SCP_CODES) + ["STTC", "CD", "HYP"],
        },
        index=EXPECTED_MI_SCP_CODES + ["NDT", "LBBB", "LVH"],
    )


def test_mi_mapping_is_dynamic_and_complete():
    scp = _scp_table()
    assert set(derive_mi_codes(scp)) == set(EXPECTED_MI_SCP_CODES)
    altered = scp.drop(index="ALMI")
    with pytest.raises(ValueError, match="does not match"):
        derive_mi_codes(altered)


def test_labels_preserve_overlap_and_hard_negative_groups():
    frame = pd.DataFrame(
        {
            "scp_codes": [{"AMI": 80, "LBBB": 60}, {"NDT": 70}, {"NORM": 100}],
            "validated_by_human": [True, False, False],
            "second_opinion": [True, False, False],
        },
        index=[1, 2, 3],
    )
    labelled = build_mi_label(frame, _scp_table())
    assert labelled.loc[1, "mi_label"] == 1
    assert labelled.loc[1, "annotation_likelihood_max"] == pytest.approx(0.8)
    assert "CD" in labelled.loc[1, "overlapping_superclasses"]
    assert labelled.loc[2, "hard_negative"]
    assert np.isnan(labelled.loc[3, "annotation_likelihood_max"])
    assert labelled.loc[3, "label_quality_group"] == "negative_unquantified"


def test_missing_validation_metadata_is_not_true():
    frame = pd.DataFrame({
        "scp_codes": [{"AMI": 80}],
        "validated_by_human": [np.nan], "second_opinion": [np.nan],
    }, index=[1])
    labelled = build_mi_label(frame, _scp_table())
    assert not labelled.loc[1, "validated_by_human_flag"]
    assert not labelled.loc[1, "second_opinion_flag"]
    assert labelled.loc[1, "label_quality_group"] == "supported"


def test_fold_10_and_calibration_fold_are_locked():
    with pytest.raises(PermissionError):
        guard_fold_access([1, 10], purpose="tuning")
    with pytest.raises(PermissionError):
        guard_fold_access([1, 9], purpose="qc_calibration")
    guard_fold_access([10], purpose="final_locked_evaluation")


def test_nonfinite_and_padding_masks_survive_ingestion(monkeypatch):
    raw = np.ones((999, 12), dtype=float) * 0.1
    raw[5, 2] = np.nan
    record = SimpleNamespace(
        fs=100, n_sig=12, sig_name=CANONICAL_LEAD_ORDER,
        p_signal=raw,
    )
    monkeypatch.setattr("aquire_preprocessing.structural.wfdb.rdrecord", lambda _: record)
    result = validate_record(42, sampling_rate=100, record_path="/missing/synthetic")
    assert result.is_valid
    assert result.signal_mv.shape == (12, 1000)
    assert not result.sample_mask[2, 5]
    assert not result.sample_mask[:, -1].any()
    assert np.isfinite(result.signal_mv).all()
    assert "nonfinite_replaced:1" in result.transformations
    assert "padded_tail_samples:1" in result.transformations


def test_sampling_rate_mismatch_rejected(monkeypatch):
    record = SimpleNamespace(
        fs=500, n_sig=12, sig_name=CANONICAL_LEAD_ORDER,
        p_signal=np.zeros((5000, 12)),
    )
    monkeypatch.setattr("aquire_preprocessing.structural.wfdb.rdrecord", lambda _: record)
    result = validate_record(42, sampling_rate=100, record_path="/missing/synthetic")
    assert not result.is_valid
    assert "Sampling rate mismatch" in result.errors[0]


def test_powerline_detector_is_unavailable_at_100_hz():
    time = np.arange(1000) / 100
    quality = assess_lead_quality(np.sin(2 * np.pi * 5 * time), "II", fs=100)
    assert quality.powerline_supported is False
    assert quality.powerline_snr_db == 0.0


def test_cross_lead_indeterminate_is_finite():
    signal = np.zeros((12, 1000), dtype=float)
    mask = np.zeros_like(signal, dtype=bool)
    result = check_cross_lead_physics(signal, sample_mask=mask)
    assert result.status == "INDETERMINATE"
    numeric = [
        result.einthoven_residual_mv, result.goldberger_avr_residual_mv,
        result.goldberger_avl_residual_mv, result.goldberger_avf_residual_mv,
    ]
    assert np.isfinite(numeric).all()


def test_negative_multibeat_qrs_is_supported():
    signal = np.zeros(1000, dtype=float)
    for center in [100, 220, 340, 460, 580, 700, 820, 940]:
        signal[center - 2:center + 3] = [-0.3, -0.8, -1.2, -0.8, -0.3]
    metrics = compare_lead_morphology(signal, signal.copy(), "aVR", fs=100)
    assert metrics.matched_beats >= 7
    assert metrics.passed


def test_qc_calibration_uses_only_development_folds():
    frame = pd.DataFrame({
        "strat_fold": [1, 2, 3, 4, 5, 6, 7, 8],
        "score": [0.1, 0.2, 0.8, 0.9, 0.15, 0.75, 0.3, 0.7],
        "baseline_drift": [0, 0, 1, 1, 0, 1, 0, 1],
    })
    selected, curve = calibrate_qc_threshold(frame, "score", ["baseline_drift"], 0.75)
    assert selected["downstream_model_metric_used"] is False
    assert selected["sensitivity"] >= 0.75
    assert len(curve)
    with pytest.raises(PermissionError):
        calibrate_qc_threshold(pd.concat([frame, frame.iloc[[0]].assign(strat_fold=9)]), "score", ["baseline_drift"])


def test_technical_annotation_is_lead_specific():
    assert annotation_present_for_lead("['I', 'V2']", "V2")
    assert not annotation_present_for_lead("['I', 'V2']", "V3")
    assert annotation_present_for_lead("true", "V6")


def test_kaggle_driver_contains_no_preprocessing_implementation():
    path = Path(__file__).parents[1] / "kaggle_notebook_pipeline.py"
    tree = ast.parse(path.read_text())
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert function_names <= {"_bootstrap", "main"}
