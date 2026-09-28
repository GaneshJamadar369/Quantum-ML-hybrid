from __future__ import annotations

import json
from pathlib import Path

import pytest

from aquire_preprocessing.prototype_bundle import (
    BundleError,
    artifact_records,
    verify_bundle,
)


BASE_ROLES = (
    "transformer",
    "waveform_normalizer",
    "h128_imputer",
    "h128_scaler",
    "pls_q4",
    "angle_quantiles",
    "vqc_score_alignment",
    "vqc_model",
    "morphology_feature_manifest",
    "morphology_conditioner",
    "morphology_hgb",
    "fusion",
)


def _bundle(tmp_path: Path, *, calibrated: bool = False) -> Path:
    entries = []
    for role in BASE_ROLES:
        relative = f"{role}.bin"
        (tmp_path / relative).write_bytes(role.encode())
        entries.append((role, relative))
    if calibrated:
        for role in ("platt_calibrator", "decision_threshold"):
            relative = f"{role}.bin"
            (tmp_path / relative).write_bytes(role.encode())
            entries.append((role, relative))
    manifest = {
        "schema_version": 1,
        "model_version": "test-v1",
        "task": "mi_pattern_vs_non_mi_pattern",
        "prediction_routing": "fixed_parallel_all_eligible_inputs",
        "input": {"shape": [12, 1000], "sampling_rate_hz": 100},
        "routes": {
            "quantum": {"active": True},
            "classical": {"active": True},
            "fusion": {"active": True},
        },
        "calibration_state": "fold9_frozen" if calibrated else "development_uncalibrated",
        "artifacts": artifact_records(tmp_path, entries),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_bundle_requires_explicit_uncalibrated_override(tmp_path):
    root = _bundle(tmp_path)
    with pytest.raises(BundleError, match="Fold-9 calibration"):
        verify_bundle(root)
    result = verify_bundle(root, allow_uncalibrated=True)
    assert not result.calibrated
    assert result.paths_for("vqc_model")


def test_bundle_requires_both_active_routes(tmp_path):
    root = _bundle(tmp_path, calibrated=True)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["routes"]["quantum"]["active"] = False
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(BundleError, match="Quantum route must be active"):
        verify_bundle(root)


def test_bundle_detects_tampering(tmp_path):
    root = _bundle(tmp_path, calibrated=True)
    (root / "fusion.bin").write_bytes(b"changed")
    with pytest.raises(BundleError, match="Artifact size mismatch|Artifact hash mismatch"):
        verify_bundle(root)
