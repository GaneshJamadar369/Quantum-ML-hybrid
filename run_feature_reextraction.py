"""Re-extract validated local ECG measurements from processed waveforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aquire_preprocessing.feature_reextraction import reextract_features_from_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--allow-unvalidated-fallback", action="store_true")
    parser.add_argument(
        "--exclude-ecg-ids",
        type=Path,
        help="CSV containing ecg_id values used by an earlier calibration run",
    )
    args = parser.parse_args()
    excluded = None
    if args.exclude_ecg_ids:
        excluded = set(pd.read_csv(args.exclude_ecg_ids, usecols=["ecg_id"]).ecg_id.astype(int))
    result = reextract_features_from_hdf5(
        args.hdf5,
        args.output,
        sample_size=args.sample_size,
        seed=args.seed,
        shard_size=args.shard_size,
        require_delineation=not args.allow_unvalidated_fallback,
        exclude_ecg_ids=excluded,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
