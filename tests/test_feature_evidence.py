import json

import numpy as np
import pandas as pd
import pytest

from aquire_preprocessing.feature_evidence import (
    analyse_extracted_features,
    benjamini_hochberg,
    derive_clinical_composites,
    raw_waveform_statistical_audit,
    run_feature_evidence_gate,
)


def _deployable_features(n=32):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({"ecg_id": np.arange(1, n + 1)})
    for lead in ["i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"]:
        frame[f"{lead}__st60_mv"] = rng.normal(0, 0.08, n)
        frame[f"{lead}__t_polarity"] = rng.choice([-1.0, 0.0, 1.0], n)
        frame[f"{lead}__r_amp_mv"] = rng.normal(0.8, 0.2, n)
        frame[f"{lead}__s_amp_mv"] = rng.normal(-0.4, 0.1, n)
        frame[f"{lead}__rs_ratio"] = np.abs(frame[f"{lead}__r_amp_mv"] / frame[f"{lead}__s_amp_mv"])
        frame[f"{lead}__range_mv"] = rng.normal(1.5, 0.2, n)
        frame[f"{lead}__valid_fraction"] = 1.0
    frame["heart_rate_bpm"] = rng.normal(72, 8, n)
    frame["rr_median_ms"] = 60000 / frame.heart_rate_bpm
    frame["extractor_failures"] = "[]"
    return frame


def test_benjamini_hochberg_monotonic_and_nan_safe():
    adjusted = benjamini_hochberg([0.01, 0.04, np.nan, 0.03])
    assert np.isnan(adjusted[2])
    assert np.all((adjusted[np.isfinite(adjusted)] >= 0) & (adjusted[np.isfinite(adjusted)] <= 1))
    assert adjusted[0] <= adjusted[1]


def test_clinical_composites_are_interpretable_and_do_not_overwrite_inputs():
    features = _deployable_features(12).set_index("ecg_id")
    original = features.copy()
    derived, registry = derive_clinical_composites(features)
    assert "clinical__inferior__st_mean_mv" in derived
    assert "clinical__r_progression_slope_v1_v6" in derived
    assert "clinical__frontal_axis_proxy_deg" in derived
    assert registry.clinical_rationale.notna().all()
    pd.testing.assert_frame_equal(features, original)


def test_feature_statistics_are_development_only_and_never_auto_delete(tmp_path):
    features = _deployable_features(32).set_index("ecg_id")
    labels = np.tile([0, 1], 16)
    folds = np.tile(np.arange(1, 9), 4)
    evidence, _, decisions = analyse_extracted_features(features, labels, folds, tmp_path)
    assert len(evidence) > 0
    assert not decisions.automatic_removal.any()
    assert (tmp_path / "feature_evidence_protocol.json").exists()
    with pytest.raises((ValueError, PermissionError)):
        analyse_extracted_features(features, labels, np.where(folds == 8, 10, folds), tmp_path / "forbidden")


def test_complete_gate_aligns_by_ecg_id_and_builds_decision_register(tmp_path):
    n = 32
    features = _deployable_features(n)
    metadata = pd.DataFrame({
        "ecg_id": np.arange(1, n + 1),
        "patient_id": np.arange(100, 100 + n),
        "mi_label": np.tile([0, 1], n // 2),
        "strat_fold": np.tile(np.arange(1, 9), n // 8),
        "hard_negative": False,
        "eligibility": "PRIMARY",
        "qc_status": "PASS",
    })
    manifest = pd.DataFrame({
        "ecg_id": np.arange(1, n + 1),
        "patient_id": np.arange(100, 100 + n),
        "mi_label": np.tile([0, 1], n // 2),
        "strat_fold": np.tile(np.arange(1, 9), n // 8),
        "hard_negative": False,
        "sex": np.tile([0, 1], n // 2),
        "age": np.linspace(35, 85, n),
        "label_quality_group": "high",
    })
    reference = tmp_path / "ecgdeli_features.csv"
    pd.DataFrame({
        "ecg_id": np.arange(1, n + 1),
        "RR_Mean_Global": features.rr_median_ms,
        "unused_large_reference_field": np.arange(n),
    }).to_csv(reference, index=False)
    summary = run_feature_evidence_gate(
        features, metadata, manifest, tmp_path,
        reference_features=reference,
        reference_pairs={"rr_median_ms": "ref_ecgdeli__RR_Mean_Global"},
    )
    assert summary["folds_used"] == list(range(1, 9))
    assert summary["fold_9_accessed"] is False
    assert summary["fold_10_accessed"] is False
    assert summary["ready_for_automatic_feature_deletion"] is False
    assert summary["reference_pairs_evaluated"] == 1
    assert (tmp_path / "candidate_feature_registry.csv").exists()
    assert (tmp_path / "post_extraction" / "feature_decision_register.csv").exists()


def test_raw_waveform_audit_generates_pre_extraction_evidence(tmp_path):
    h5py = pytest.importorskip("h5py")
    h5_path = tmp_path / "primary.h5"
    rng = np.random.default_rng(11)
    n = 16
    with h5py.File(h5_path, "w") as h5:
        signal = rng.normal(0, 0.2, (n, 12, 1000)).astype(np.float32)
        h5["accepted_signal"] = signal
        h5["minimal_signal"] = signal.copy()
        h5["sample_mask"] = np.ones_like(signal, dtype=bool)
        h5["lead_mask"] = np.ones((n, 12), dtype=bool)
        h5["ecg_id"] = np.arange(1, n + 1)
        h5["patient_id"] = np.arange(101, 101 + n)
        h5["mi_label"] = np.tile([0, 1], n // 2)
        h5["strat_fold"] = np.tile(np.arange(1, 9), 2)
    records, leads = raw_waveform_statistical_audit(h5_path, tmp_path / "raw")
    assert len(records) == n
    assert set(leads.lead) == {"i", "ii", "iii", "avr", "avl", "avf", "v1", "v2", "v3", "v4", "v5", "v6"}
    assert (tmp_path / "raw" / "raw_waveform_class_evidence.csv").exists()
    assert (tmp_path / "raw" / "raw_lead_class_evidence.csv").exists()
