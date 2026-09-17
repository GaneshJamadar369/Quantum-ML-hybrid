"""
AQUIRE-Med Preprocessing Pipeline — Main Entry Point

Run this script to execute the full preprocessing pipeline.
Works both on Kaggle (with datasets mounted at /kaggle/input/)
and locally (with datasets in ./data/).

Usage:
    python run_pipeline.py                    # Full run
    python run_pipeline.py --sample-size 100  # Quick test with 100 records
    python run_pipeline.py --manifest-only    # Build manifest without processing signals
"""

import argparse
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aquire_pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="AQUIRE-Med Signal Preprocessing Pipeline"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Process only N records (for testing). Default: all records.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only build the patient manifest (no signal processing).",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=100,
        choices=[100, 500],
        help="Sampling rate to use (100 or 500 Hz). Default: 100.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save outputs to disk.",
    )
    args = parser.parse_args()

    # Import here to allow CLI help without loading heavy modules
    from aquire_preprocessing.config import ENVIRONMENT, PATHS
    from aquire_preprocessing.manifest import build_manifest
    from aquire_preprocessing.pipeline import process_batch

    print(f"\n{'='*60}")
    print(f"AQUIRE-Med Preprocessing Pipeline")
    print(f"{'='*60}")
    print(f"Environment:    {ENVIRONMENT}")
    print(f"PTB-XL root:    {PATHS.ptbxl_root}")
    print(f"PTB-XL+ root:   {PATHS.ptbxlp_root}")
    print(f"Output root:    {PATHS.output_root}")
    print(f"Sampling rate:  {args.sampling_rate} Hz")
    print(f"Sample size:    {args.sample_size or 'ALL'}")
    print(f"{'='*60}\n")

    # Step 1: Build manifest
    logger.info("Step 1: Building patient manifest...")
    manifest = build_manifest(
        join_features=True,
        save=not args.no_save,
    )

    if args.manifest_only:
        logger.info("Manifest-only mode. Done.")
        return

    # Step 2: Process signals
    logger.info("Step 2: Processing ECG signals...")
    contracts, summary = process_batch(
        manifest=manifest,
        sampling_rate=args.sampling_rate,
        max_records=args.sample_size,
        save_output=not args.no_save,
    )

    # Step 3: Summary
    if contracts:
        import numpy as np
        mi_labels = [c["mi_label"] for c in contracts]
        folds = [c["strat_fold"] for c in contracts]
        print(f"\nMI label distribution:")
        print(f"  MI=1: {sum(mi_labels)} ({sum(mi_labels)/len(mi_labels)*100:.1f}%)")
        print(f"  MI=0: {len(mi_labels)-sum(mi_labels)} ({(len(mi_labels)-sum(mi_labels))/len(mi_labels)*100:.1f}%)")
        print(f"\nFold distribution:")
        for fold in sorted(set(folds)):
            n = folds.count(fold)
            print(f"  Fold {fold}: {n} records")

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
