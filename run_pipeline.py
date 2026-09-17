"""Single source-of-truth CLI for local and Kaggle preprocessing."""

from __future__ import annotations

import argparse
import json
import logging

from aquire_preprocessing.config import CALIBRATION_FOLD, DEV_FOLDS, ENVIRONMENT, LOCKED_TEST_FOLD, PATHS
from aquire_preprocessing.manifest import build_manifest
from aquire_preprocessing.pipeline import process_batch
from aquire_preprocessing.registry import verify_ptbxl_identity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="AQUIRE-Med research preprocessing")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--sampling-rate", type=int, choices=[100, 500], default=100)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--role", choices=["development", "calibration", "final_locked_evaluation"],
        default="development",
    )
    parser.add_argument("--allow-unverified-release", action="store_true", help="Development-only; outputs remain marked unverified")
    args = parser.parse_args()

    print(json.dumps({
        "environment": ENVIRONMENT,
        "ptbxl_root": str(PATHS.ptbxl_root),
        "ptbxl_plus_root": str(PATHS.ptbxlp_root),
        "output_root": str(PATHS.output_root),
        "sampling_rate": args.sampling_rate,
    }, indent=2))
    identity = verify_ptbxl_identity(PATHS.ptbxl_root, strict=not args.allow_unverified_release)
    print("Dataset identity:", json.dumps(identity, indent=2))
    manifest = build_manifest(
        join_features=False,
        save=not args.no_save,
        verify_release=not args.allow_unverified_release,
    )
    if args.manifest_only:
        return
    if args.role == "development":
        selected = manifest[manifest.strat_fold.isin(DEV_FOLDS)]
        purpose = "preprocessing_tuning"
    elif args.role == "calibration":
        selected = manifest[manifest.strat_fold.eq(CALIBRATION_FOLD)]
        purpose = "calibration_evaluation"
    else:
        selected = manifest[manifest.strat_fold.eq(LOCKED_TEST_FOLD)]
        purpose = "final_locked_evaluation"
    _, summary = process_batch(
        selected,
        sampling_rate=args.sampling_rate,
        max_records=args.sample_size,
        save_output=not args.no_save,
        collect_records=False,
        purpose=purpose,
    )
    print("Pipeline summary:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
