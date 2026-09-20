"""Kaggle Cloud Runner for Phase 6Q Quantum ML & Phase 7 Locked Holdout Evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

REPOSITORY = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("\n>>> " + " ".join(command), flush=True)
    started = time.time()
    process = subprocess.Popen(
        command, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    print(f"[exit={process.returncode}, seconds={time.time() - started:.1f}]", flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required mounted input is absent: {path}")
    print(f"input: {path}", flush=True)
    return path


def main() -> None:
    print("=== AQUIRE-Med Phase 6Q (Quantum) & Phase 7 (Holdout) Runner ===", flush=True)
    root = Path("/kaggle/input")

    # Locate mounted inputs
    preprocessing = (
        root / "notebooks/swayamjeetbhagat4/aquire-med-preprocessing-pipeline/aquire-artifacts"
    )
    feature_repair = (
        root / "notebooks/swayamjeetbhagat4/aquire-med-full-feature-repair/full-feature-repair"
    )

    primary_h5 = require(preprocessing / "primary_development_100hz.h5")
    metadata_csv = require(preprocessing / "processing_metadata_development.csv")
    features_csv = require(
        feature_repair / "feature-evidence-v0-4/deployable_features_with_clinical_composites.csv"
    )

    out_dir_6q = Path("/kaggle/working/g6-quantum-baselines")
    out_dir_7 = Path("/kaggle/working/g7-locked-holdout")
    out_dir_6q.mkdir(parents=True, exist_ok=True)
    out_dir_7.mkdir(parents=True, exist_ok=True)

    # Clone latest repo
    repo = Path("/tmp/Quantum-ML-hybrid")
    if repo.exists():
        run(["rm", "-rf", str(repo)])
    run(["git", "clone", REPOSITORY, str(repo)])
    run(["git", "fetch", "--all"], cwd=repo)
    run(["git", "checkout", "main"], cwd=repo)
    run(["git", "reset", "--hard", "origin/main"], cwd=repo)
    run(["git", "log", "-1", "--oneline"], cwd=repo)

    # Install dependencies
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q", "pennylane", "pennylane-lightning", "xgboost", "shap"])

    manifest = require(repo / "configs/approved_feature_manifest_v0_4.json")

    # Run Phase 6Q Quantum Baselines (8-Fold OOF)
    print("\n--- Running Phase 6Q Quantum Baselines (Folds 1-8 OOF) ---", flush=True)
    run([
        sys.executable, "run_quantum_baselines.py",
        "--h5-path", str(primary_h5),
        "--metadata-path", str(metadata_csv),
        "--features-csv", str(features_csv),
        "--manifest-path", str(manifest),
        "--output-dir", str(out_dir_6q),
        "--epochs", "15",
        "--batch-size", "64",
        "--lr", "1e-3",
    ], cwd=repo)

    # Run Phase 7 Locked Holdout Single-Pass Evaluation (Folds 9 & 10)
    print("\n--- Running Phase 7 Locked Holdout Single-Pass Evaluation (Folds 9 & 10) ---", flush=True)
    run([
        sys.executable, "run_sealed_holdout_evaluation.py",
        "--h5-path", str(primary_h5),
        "--metadata-path", str(metadata_csv),
        "--features-csv", str(features_csv),
        "--manifest-path", str(manifest),
        "--output-dir", str(out_dir_7),
    ], cwd=repo)

    # Manifest summary
    files_6q = sorted(
        ({"path": str(p.relative_to(out_dir_6q)), "bytes": p.stat().st_size}
         for p in out_dir_6q.rglob("*") if p.is_file()),
        key=lambda item: item["path"],
    )
    (out_dir_6q / "artifact_manifest.json").write_text(json.dumps(files_6q, indent=2))

    files_7 = sorted(
        ({"path": str(p.relative_to(out_dir_7)), "bytes": p.stat().st_size}
         for p in out_dir_7.rglob("*") if p.is_file()),
        key=lambda item: item["path"],
    )
    (out_dir_7 / "artifact_manifest.json").write_text(json.dumps(files_7, indent=2))

    print("\n=== Phase 6Q & Phase 7 Completed Successfully ===", flush=True)


if __name__ == "__main__":
    main()
