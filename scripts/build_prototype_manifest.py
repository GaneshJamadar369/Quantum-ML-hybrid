#!/usr/bin/env python3
"""Create a signed-by-content manifest after every model artifact is exported."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from aquire_preprocessing.prototype_bundle import artifact_records, verify_bundle


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _entry(value: str) -> tuple[str, str]:
    role, separator, relative = value.partition("=")
    if not separator or not role or not relative:
        raise argparse.ArgumentTypeError("Artifacts use ROLE=RELATIVE_PATH")
    return role, relative


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument(
        "--calibration-state",
        choices=("fold9_frozen", "development_uncalibrated"),
        required=True,
    )
    parser.add_argument("--artifact", action="append", type=_entry, default=[], required=True)
    parser.add_argument("--training-patient-sha256", required=True)
    parser.add_argument("--feature-manifest-sha256", required=True)
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "model_version": args.model_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "task": "mi_pattern_vs_non_mi_pattern",
        "prediction_routing": "fixed_parallel_all_eligible_inputs",
        "input": {"shape": [12, 1000], "sampling_rate_hz": 100, "physical_units": "mV"},
        "routes": {
            "quantum": {"active": True, "model": "transformer_pls_q4_vqc"},
            "classical": {"active": True, "model": "morphology_106_hgb"},
            "fusion": {"active": True, "model": "nonnegative_logistic"},
        },
        "calibration_state": args.calibration_state,
        "training_patient_sha256": args.training_patient_sha256,
        "feature_manifest_sha256": args.feature_manifest_sha256,
        "artifacts": artifact_records(bundle, args.artifact),
    }
    temporary = bundle / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(bundle / "manifest.json")
    verified = verify_bundle(
        bundle,
        allow_uncalibrated=args.calibration_state == "development_uncalibrated",
    )
    print(
        json.dumps(
            {
                "valid": True,
                "model_version": verified.model_version,
                "calibrated": verified.calibrated,
                "artifacts": len(verified.artifacts),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
