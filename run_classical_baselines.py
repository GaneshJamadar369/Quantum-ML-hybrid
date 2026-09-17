"""Run fold-safe tabular classical baselines from processed artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from aquire_preprocessing.baselines import run_oof_baselines
from aquire_preprocessing.config import DEV_FOLDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/g6"))
    parser.add_argument(
        "--mi-label-policy", choices=["all", "supported_or_high", "high_only"],
        default="all", help="Sensitivity analysis excludes uncertain positive annotations",
    )
    args = parser.parse_args()
    features = pd.read_csv(args.features).set_index("ecg_id")
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    joined = metadata.join(features, how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")]
    if args.mi_label_policy != "all":
        allowed = {"high", "supported"} if args.mi_label_policy == "supported_or_high" else {"high"}
        joined = joined[~joined.mi_label.eq(1) | joined.label_quality_group.isin(allowed)]
    excluded = {"patient_id", "mi_label", "strat_fold", "hard_negative", "array_index"}
    x = joined[[c for c in features.columns if c in joined.columns and c not in excluded]]
    run_oof_baselines(
        x, joined.mi_label, joined.strat_fold, joined.patient_id,
        args.output / args.mi_label_policy,
        record_ids=joined.index,
        qc_groups=joined.qc_status if "qc_status" in joined else None,
        hard_negative=joined.hard_negative if "hard_negative" in joined else None,
    )


if __name__ == "__main__":
    main()
