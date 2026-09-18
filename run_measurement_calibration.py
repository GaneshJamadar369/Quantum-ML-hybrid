"""Evaluate repaired local measurements before a full feature re-extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aquire_preprocessing.measurement_calibration import run_measurement_calibration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-features", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--feature-pairs", type=Path, default=Path("configs/deployable_reference_pairs.json"))
    parser.add_argument("--thresholds", type=Path, default=Path("configs/measurement_acceptance.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conclusion = run_measurement_calibration(
        args.local_features,
        args.reference_features,
        json.loads(args.feature_pairs.read_text()),
        json.loads(args.thresholds.read_text()),
        args.output,
    )
    print(json.dumps(conclusion, indent=2))


if __name__ == "__main__":
    main()
