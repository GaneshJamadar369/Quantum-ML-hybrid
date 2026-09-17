"""Run the clinical/statistical feature-evidence gate on processed PTB-XL data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aquire_preprocessing.feature_evidence import run_feature_evidence_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="AQUIRE-Med feature evidence gate")
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--primary-hdf5", type=Path)
    parser.add_argument("--reference-features", type=Path)
    parser.add_argument(
        "--reference-pairs",
        type=Path,
        default=Path("configs/deployable_reference_pairs.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/g5_feature_evidence"))
    parser.add_argument("--max-raw-records", type=int)
    args = parser.parse_args()

    features = pd.read_csv(args.features)
    metadata = pd.read_csv(args.metadata)
    manifest = pd.read_csv(args.manifest)
    reference_pairs = None
    if args.reference_features:
        reference_pairs = json.loads(args.reference_pairs.read_text())
    summary = run_feature_evidence_gate(
        features=features,
        metadata=metadata,
        manifest=manifest,
        output_dir=args.output,
        primary_hdf5=args.primary_hdf5,
        reference_features=args.reference_features,
        reference_pairs=reference_pairs,
        max_raw_records=args.max_raw_records,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
