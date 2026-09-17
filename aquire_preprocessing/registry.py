"""Dataset identity, checksum and immutable-file registry helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

EXPECTED_PTBXL_VERSION = "1.0.3"
EXPECTED_PTBXL_RECORDS = 21_799
EXPECTED_PTBXLP_VERSION = "1.0.1"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_ptbxl_identity(root: Path, strict: bool = True) -> Dict[str, object]:
    """Verify the mounted PTB-XL release using content, not a label constant."""
    database = root / "ptbxl_database.csv"
    statements = root / "scp_statements.csv"
    checksum_file = root / "SHA256SUMS.txt"
    required = [database, statements, root / "records100", checksum_file]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing PTB-XL inputs: {missing}")

    header = pd.read_csv(database, usecols=["ecg_id", "patient_id", "strat_fold"])
    report = {
        "declared_version": EXPECTED_PTBXL_VERSION,
        "record_count": int(len(header)),
        "unique_ecg_ids": int(header["ecg_id"].nunique()),
        "unique_patients": int(header["patient_id"].nunique()),
        "folds": sorted(int(v) for v in header["strat_fold"].dropna().unique()),
        "database_sha256": sha256_file(database),
        "statements_sha256": sha256_file(statements),
        "checksum_manifest": str(checksum_file.resolve()),
    }
    errors = []
    if report["record_count"] != EXPECTED_PTBXL_RECORDS:
        errors.append(
            f"expected {EXPECTED_PTBXL_RECORDS} records for PTB-XL v{EXPECTED_PTBXL_VERSION}, "
            f"found {report['record_count']}"
        )
    if report["record_count"] != report["unique_ecg_ids"]:
        errors.append("ecg_id is not unique")
    if report["folds"] != list(range(1, 11)):
        errors.append(f"expected folds 1..10, found {report['folds']}")
    expected_checksums = {}
    for line in checksum_file.read_text(errors="replace").splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2:
            expected_checksums[fields[1].lstrip("*./")] = fields[0]
    for relative, actual in [
        ("ptbxl_database.csv", report["database_sha256"]),
        ("scp_statements.csv", report["statements_sha256"]),
    ]:
        expected = expected_checksums.get(relative)
        if expected is None:
            errors.append(f"checksum manifest lacks {relative}")
        elif expected.lower() != str(actual).lower():
            errors.append(f"checksum mismatch for {relative}")
    report["checksum_manifest_verified"] = not any("checksum" in error for error in errors)
    report["verified"] = not errors
    report["errors"] = errors
    if strict and errors:
        raise ValueError("PTB-XL identity verification failed: " + "; ".join(errors))
    return report


def verify_waveform_references(root: Path, manifest: pd.DataFrame) -> Dict[str, object]:
    """Verify every manifest waveform points to a paired WFDB header/data file."""
    missing = []
    checked = 0
    for field in ("filename_lr", "filename_hr"):
        if field not in manifest:
            raise ValueError(f"Manifest lacks {field}")
        for ecg_id, relative in manifest[field].items():
            base = Path(root) / str(relative)
            absent = [
                str(base.with_suffix(ext)) for ext in (".hea", ".dat")
                if not base.with_suffix(ext).exists()
            ]
            checked += 1
            if absent:
                missing.append({"ecg_id": int(ecg_id), "field": field, "files": absent})
    return {
        "records": int(len(manifest)), "references_checked": checked,
        "missing_pair_count": len(missing), "missing_examples": missing[:100],
        "verified": not missing,
    }


def build_file_registry(paths: Iterable[Path], output: Path) -> Dict[str, object]:
    files = []
    for path in sorted(Path(p) for p in paths):
        if path.is_file():
            files.append({
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    payload = {"files": files, "registry_sha256": ""}
    canonical = json.dumps(files, sort_keys=True).encode()
    payload["registry_sha256"] = hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload
