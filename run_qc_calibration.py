"""Create development-only QC operating-point reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from aquire_preprocessing.qc_calibration import write_qc_calibration_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True, help="JSON list of detector specifications")
    parser.add_argument("--output", type=Path, default=Path("artifacts/g4_qc"))
    args = parser.parse_args()
    frame = pd.read_csv(args.metrics)
    specs = json.loads(args.spec.read_text())
    write_qc_calibration_report(frame, specs, args.output)


if __name__ == "__main__":
    main()
