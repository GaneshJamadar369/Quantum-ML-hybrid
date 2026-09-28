"""Kaggle GPU job that freezes the two-route prototype without opening Fold 10."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
SOURCE_COMMIT = "REPLACE_AFTER_COMMIT"


def command(argv, *, cwd=None, env=None):
    print(">>>", " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=cwd, env=env, check=True)


def unique(marker: str, *, contains: str | None = None) -> Path:
    matches = sorted(Path("/kaggle/input").rglob(marker))
    if contains:
        matches = [path for path in matches if contains in str(path)]
    print(f"locate {marker} ({contains=}): {matches}", flush=True)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {marker}, found {len(matches)}")
    return matches[0]


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The final Transformer/VQC export requires a Kaggle GPU")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    repo = Path("/tmp/Quantum-ML-hybrid")
    shutil.rmtree(repo, ignore_errors=True)
    command(["git", "init", repo])
    command(["git", "remote", "add", "origin", REPO], cwd=repo)
    command(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=repo)
    command(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"Source mismatch: {actual} != {SOURCE_COMMIT}")
    command([sys.executable, "-m", "pip", "install", "-q", "wfdb", "neurokit2", "joblib"])
    command([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", repo])

    ptbxl_database = unique("ptbxl_database.csv", contains="1.0.3")
    dev_h5 = unique("primary_development_100hz.h5")
    dev_metadata = unique("processing_metadata_development.csv")
    dev_features = unique("deployable_features_with_clinical_composites.csv")
    oof_candidates = sorted(Path("/kaggle/input").rglob("oof_predictions_and_observables.csv"))
    oof_candidates = [path for path in oof_candidates if "q4-score-alignment" in str(path) and "seed_42" in str(path)]
    if len(oof_candidates) != 1:
        raise FileNotFoundError(f"Expected canonical seed-42 score-alignment OOF file: {oof_candidates}")
    oof_scores = oof_candidates[0]

    calibration_root = Path("/kaggle/working/calibration")
    env = os.environ.copy()
    env["AQUIRE_PTBXL_ROOT"] = str(ptbxl_database.parent)
    env["AQUIRE_OUTPUT_ROOT"] = str(calibration_root)
    env["PYTHONUNBUFFERED"] = "1"
    # This command selects strat_fold==9 internally. Fold 10 remains untouched.
    command([sys.executable, "run_pipeline.py", "--role", "calibration"], cwd=repo, env=env)

    cal_h5 = calibration_root / "primary_calibration_100hz.h5"
    cal_metadata = calibration_root / "processing_metadata_calibration.csv"
    cal_features = calibration_root / "deployable_features_calibration.csv"
    for required in (cal_h5, cal_metadata, cal_features):
        if not required.is_file():
            raise FileNotFoundError(required)

    bundle = Path("/kaggle/working/aquire-hybrid-q4-v1")
    command([
        sys.executable, "train_export_hybrid_prototype.py",
        "--development-hdf5", dev_h5,
        "--development-metadata", dev_metadata,
        "--development-features", dev_features,
        "--calibration-hdf5", cal_h5,
        "--calibration-metadata", cal_metadata,
        "--calibration-features", cal_features,
        "--oof-scores", oof_scores,
        "--feature-manifest", "configs/approved_feature_manifest_v0_4.json",
        "--bundle", bundle,
        "--model-version", "aquire-hybrid-q4-v1",
        "--transformer-epochs", "13",
        "--vqc-epochs", "60",
        "--vqc-seeds", "42", "31415", "27182",
        "--device", "cuda",
    ], cwd=repo, env=env)
    command([sys.executable, "scripts/verify_prototype_bundle.py", bundle], cwd=repo, env=env)
    print("Frozen hybrid bundle exported. Fold 10 was not accessed.", flush=True)


if __name__ == "__main__":
    main()
