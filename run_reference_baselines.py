"""Run fold-safe PTB-XL+ reference/oracle baselines after Gates G0–G5 pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aquire_preprocessing.baselines import run_oof_baselines
from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.manifest import build_manifest, join_ptbxl_plus_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["ecgdeli", "12sl", "unig"], required=True)
    parser.add_argument("--processing-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/g6_reference"))
    args = parser.parse_args()
    manifest = build_manifest(join_features=False, save=False, verify_release=True)
    joined, audit = join_ptbxl_plus_features(
        manifest,
        load_12sl=args.source == "12sl",
        load_ecgdeli=args.source == "ecgdeli",
        load_unig=args.source == "unig",
    )
    metadata = pd.read_csv(args.processing_metadata).set_index("ecg_id")
    joined = joined.join(metadata[["eligibility", "qc_status"]], how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")]
    prefix = f"ref_{args.source}__"
    features = joined[[column for column in joined if column.startswith(prefix)]]
    role = "open_reference" if args.source == "ecgdeli" else "commercial_oracle"
    output = args.output / f"{args.source}_{role}"
    run_oof_baselines(
        features, joined.mi_label, joined.strat_fold, joined.patient_id, output,
        record_ids=joined.index, qc_groups=joined.qc_status,
        hard_negative=joined.hard_negative,
    )


if __name__ == "__main__":
    main()
