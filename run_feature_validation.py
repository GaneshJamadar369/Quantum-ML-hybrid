"""Compare locally reproducible ECG measurements with PTB-XL+ references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aquire_preprocessing.feature_validation import compare_local_to_reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-features", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, default=Path("configs/deployable_reference_pairs.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/g3_features"))
    args = parser.parse_args()
    local = pd.read_csv(args.local_features).set_index("ecg_id")
    reference = pd.read_csv(args.reference_features).set_index("ecg_id")
    reference = reference.add_prefix("ref_ecgdeli__")
    joined = local.join(reference, how="left", validate="one_to_one")
    compare_local_to_reference(joined, json.loads(args.pairs.read_text()), args.output)


if __name__ == "__main__":
    main()
