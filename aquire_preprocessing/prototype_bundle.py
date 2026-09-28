"""Integrity contract for the fixed parallel hybrid inference bundle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
FIXED_ROUTING = "fixed_parallel_all_eligible_inputs"

SINGLETON_REQUIRED_ROLES = frozenset(
    {
        "transformer",
        "waveform_normalizer",
        "h128_imputer",
        "h128_scaler",
        "pls_q4",
        "angle_quantiles",
        "vqc_score_alignment",
        "morphology_feature_manifest",
        "morphology_conditioner",
        "morphology_hgb",
        "fusion",
    }
)
CALIBRATED_REQUIRED_ROLES = frozenset({"platt_calibrator", "decision_threshold"})


class BundleError(ValueError):
    """Raised when a bundle cannot safely be used for inference."""


@dataclass(frozen=True)
class ArtifactRecord:
    role: str
    path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class VerifiedBundle:
    root: Path
    manifest: dict[str, Any]
    artifacts: tuple[ArtifactRecord, ...]

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    @property
    def calibrated(self) -> bool:
        return self.manifest["calibration_state"] == "fold9_frozen"

    def paths_for(self, role: str) -> tuple[Path, ...]:
        return tuple(self.root / item.path for item in self.artifacts if item.role == role)


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BundleError(f"Artifact path must remain inside the bundle: {value!r}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise BundleError(f"Artifact path escapes the bundle: {value!r}") from error
    return resolved


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def verify_bundle(root: Path, *, allow_uncalibrated: bool = False) -> VerifiedBundle:
    """Validate schema, fixed dual-route semantics and every artifact digest."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), f"Missing bundle manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"Cannot read bundle manifest: {error}") from error

    _require(manifest.get("schema_version") == SCHEMA_VERSION, "Unsupported bundle schema")
    _require(bool(manifest.get("model_version")), "model_version is required")
    _require(
        manifest.get("task") == "mi_pattern_vs_non_mi_pattern",
        "Bundle task must be MI-pattern versus non-MI-pattern",
    )
    _require(manifest.get("prediction_routing") == FIXED_ROUTING, "Dynamic routing is prohibited")

    input_spec = manifest.get("input", {})
    _require(input_spec.get("shape") == [12, 1000], "Bundle input must be [12,1000]")
    _require(input_spec.get("sampling_rate_hz") == 100, "Bundle sampling rate must be 100 Hz")

    routes = manifest.get("routes", {})
    _require(routes.get("quantum", {}).get("active") is True, "Quantum route must be active")
    _require(routes.get("classical", {}).get("active") is True, "Classical route must be active")
    _require(routes.get("fusion", {}).get("active") is True, "Fusion must be active")

    calibration_state = manifest.get("calibration_state")
    _require(
        calibration_state in {"fold9_frozen", "development_uncalibrated"},
        "Invalid calibration_state",
    )
    if calibration_state != "fold9_frozen" and not allow_uncalibrated:
        raise BundleError("Fold-9 calibration is absent; set explicit prototype mode to continue")

    raw_artifacts = manifest.get("artifacts")
    _require(isinstance(raw_artifacts, list) and raw_artifacts, "Artifact registry is empty")
    records: list[ArtifactRecord] = []
    seen_paths: set[str] = set()
    for raw in raw_artifacts:
        try:
            record = ArtifactRecord(
                role=str(raw["role"]),
                path=str(raw["path"]),
                sha256=str(raw["sha256"]),
                bytes=int(raw["bytes"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise BundleError(f"Malformed artifact record: {raw!r}") from error
        _require(record.path not in seen_paths, f"Duplicate artifact path: {record.path}")
        seen_paths.add(record.path)
        path = _safe_relative_path(root, record.path)
        _require(path.is_file(), f"Missing artifact: {record.path}")
        _require(path.stat().st_size == record.bytes, f"Artifact size mismatch: {record.path}")
        _require(sha256_file(path) == record.sha256, f"Artifact hash mismatch: {record.path}")
        records.append(record)

    roles = {record.role for record in records}
    missing = SINGLETON_REQUIRED_ROLES - roles
    _require(not missing, f"Missing required artifact roles: {sorted(missing)}")
    _require(sum(record.role == "vqc_model" for record in records) >= 1, "At least one VQC is required")
    if calibration_state == "fold9_frozen":
        missing_calibration = CALIBRATED_REQUIRED_ROLES - roles
        _require(not missing_calibration, f"Missing calibration roles: {sorted(missing_calibration)}")

    for role in SINGLETON_REQUIRED_ROLES | CALIBRATED_REQUIRED_ROLES:
        count = sum(record.role == role for record in records)
        if role in roles:
            _require(count == 1, f"Artifact role {role!r} must occur exactly once")
    return VerifiedBundle(root=root.resolve(), manifest=manifest, artifacts=tuple(records))


def artifact_records(root: Path, entries: Iterable[tuple[str, str]]) -> list[dict[str, Any]]:
    """Create deterministic manifest entries for already exported files."""

    root = Path(root)
    output: list[dict[str, Any]] = []
    for role, relative in sorted(entries, key=lambda item: (item[0], item[1])):
        path = _safe_relative_path(root, relative)
        if not path.is_file():
            raise BundleError(f"Cannot register missing artifact: {relative}")
        output.append(
            {
                "role": role,
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return output
