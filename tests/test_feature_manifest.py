import json

import pytest

from aquire_preprocessing.feature_manifest import (
    freeze_feature_manifest,
    load_feature_manifest,
)


def test_frozen_feature_manifest_is_checksum_verified(tmp_path):
    path = tmp_path / "approved.json"
    frozen = freeze_feature_manifest(
        ["rr_median_ms", "i__r_amp_mv"], path,
        extractor_version="aquire-local-v0.4.0",
        evidence_commit="abc123",
        evidence_artifact="kernel/version",
        exclusions={"pr_interval_ms": "failed measurement gate"},
    )
    loaded = load_feature_manifest(path, ["rr_median_ms", "i__r_amp_mv", "other"])
    assert loaded["manifest_sha256"] == frozen["manifest_sha256"]
    payload = json.loads(path.read_text())
    payload["approved_features"].append("other")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_feature_manifest(path, ["rr_median_ms", "i__r_amp_mv", "other"])


def test_feature_manifest_rejects_identifiers_and_labels(tmp_path):
    with pytest.raises(ValueError, match="Forbidden predictor"):
        freeze_feature_manifest(
            ["rr_median_ms", "patient_id", "mi_label"], tmp_path / "bad.json",
            "v", "commit", "artifact", {},
        )
