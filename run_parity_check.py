"""Verify that the package and Kaggle driver share one deterministic path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import kaggle_notebook_pipeline
import run_pipeline
from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.manifest import build_manifest
from aquire_preprocessing.pipeline import process_record


def _digest(record) -> str:
    digest = hashlib.sha256()
    for value in [record.minimal_signal, record.accepted_signal, record.sample_mask, record.lead_mask]:
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(record.eligibility.value.encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("artifacts/g5/package_kaggle_parity.json"))
    args = parser.parse_args()
    if kaggle_notebook_pipeline.main is not run_pipeline.main:
        raise AssertionError("Kaggle entrypoint does not import the local package driver")
    manifest = build_manifest(join_features=False, save=False, verify_release=True)
    fixture = manifest[manifest.strat_fold.isin(DEV_FOLDS)].iloc[:args.records]
    matched = 0
    failures = []
    for ecg_id, row in fixture.iterrows():
        first = process_record(int(ecg_id), row, 100)
        second = process_record(int(ecg_id), row, 100)
        if first is None or second is None:
            failures.append({"ecg_id": int(ecg_id), "reason": "structural_failure"})
        elif _digest(first) != _digest(second):
            failures.append({"ecg_id": int(ecg_id), "reason": "digest_mismatch"})
        else:
            matched += 1
    report = {
        "fixture_records": int(len(fixture)), "matched": matched,
        "failures": failures, "same_entrypoint_callable": True,
        "passed": not failures and matched == len(fixture),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Package/Kaggle parity failed")


if __name__ == "__main__":
    main()
