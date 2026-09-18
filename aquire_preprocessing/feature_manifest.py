"""Signed predictor-manifest boundary between feature evidence and modeling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


FORBIDDEN_EXACT = {
    "ecg_id", "patient_id", "mi_label", "strat_fold", "hard_negative",
    "array_index", "extractor_version", "extractor_failures", "eligibility",
}
FORBIDDEN_TOKENS = ("label", "target", "diagn", "report", "scp", "infarction")


def manifest_checksum(payload: dict) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def freeze_feature_manifest(
    approved_features: Iterable[str],
    output: Path,
    extractor_version: str,
    evidence_commit: str,
    evidence_artifact: str,
    exclusions: dict[str, str],
) -> dict:
    features = list(dict.fromkeys(str(value) for value in approved_features))
    if not features:
        raise ValueError("Cannot freeze an empty feature manifest")
    _validate_names(features)
    payload = {
        "schema_version": 1,
        "status": "FROZEN_APPROVED",
        "extractor_version": extractor_version,
        "evidence_commit": evidence_commit,
        "evidence_artifact": evidence_artifact,
        "approved_features": features,
        "excluded_features": dict(sorted(exclusions.items())),
    }
    payload["manifest_sha256"] = manifest_checksum(payload)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(output)
    return payload


def _validate_names(features: list[str]) -> None:
    forbidden = [
        feature for feature in features
        if feature in FORBIDDEN_EXACT
        or any(token in feature.lower() for token in FORBIDDEN_TOKENS)
    ]
    if forbidden:
        raise ValueError(f"Forbidden predictor names in approved manifest: {forbidden}")


def load_feature_manifest(path: Path, available_features: Iterable[str]) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("status") != "FROZEN_APPROVED":
        raise ValueError("Feature manifest is not frozen and approved")
    if payload.get("manifest_sha256") != manifest_checksum(payload):
        raise ValueError("Feature manifest checksum mismatch")
    features = list(payload.get("approved_features", []))
    if len(features) != len(set(features)):
        raise ValueError("Feature manifest contains duplicate predictors")
    _validate_names(features)
    missing = sorted(set(features).difference(set(available_features)))
    if missing:
        raise ValueError(f"Approved predictors are absent from the feature table: {missing}")
    return payload

