"""Fit one masked waveform normalizer per held-out development fold."""

from __future__ import annotations

import argparse
from pathlib import Path

from aquire_preprocessing.normalization import fit_fold_local_scalers_from_hdf5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/g5/normalizers"))
    parser.add_argument("--samples-per-record", type=int, default=100)
    args = parser.parse_args()
    fit_fold_local_scalers_from_hdf5(
        args.hdf5, args.output, samples_per_record=args.samples_per_record
    )


if __name__ == "__main__":
    main()
