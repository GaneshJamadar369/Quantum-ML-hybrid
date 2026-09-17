"""Generate the data-dependent evidence for Gates G0–G3 and visual G4 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquire_preprocessing.config import PATHS
from aquire_preprocessing.feature_validation import coverage_report
from aquire_preprocessing.manifest import build_manifest, join_ptbxl_plus_features
from aquire_preprocessing.registry import build_file_registry, verify_ptbxl_identity
from aquire_preprocessing.reports import write_cohort_report, write_uncertain_mi_manifest
from aquire_preprocessing.visual_audit import create_visual_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/gates"))
    parser.add_argument("--visual-audit", action="store_true")
    parser.add_argument("--skip-ptbxl-plus", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    identity = verify_ptbxl_identity(PATHS.ptbxl_root, strict=True)
    (args.output / "dataset_identity.json").write_text(json.dumps(identity, indent=2))
    build_file_registry([
        PATHS.ptbxl_database_csv, PATHS.scp_statements_csv, PATHS.sha256sums,
    ], args.output / "file_registry.json")
    manifest = build_manifest(join_features=False, save=True, verify_release=True)
    write_cohort_report(manifest, args.output / "cohort_report.json")
    write_uncertain_mi_manifest(manifest, args.output / "uncertain_mi_manifest.csv")

    if not args.skip_ptbxl_plus:
        joined, audit = join_ptbxl_plus_features(manifest)
        (args.output / "ptbxl_plus_audit.json").write_text(json.dumps(audit, indent=2))
        reference_columns = [c for c in joined if c.startswith("ref_")]
        coverage_report(joined, reference_columns).to_csv(
            args.output / "ptbxl_plus_coverage.csv", index=False
        )
    if args.visual_audit:
        create_visual_audit(manifest, args.output / "visual_audit.png")


if __name__ == "__main__":
    main()
