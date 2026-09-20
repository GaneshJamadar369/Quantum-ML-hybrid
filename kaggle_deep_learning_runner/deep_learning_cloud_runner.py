"""Kaggle Cloud Runner for Phase 6B (1D Deep Learning) and Phase 6C (Multimodal Hybrid Fusion)."""

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

    output = Path("/kaggle/working/g6-deep-learning")
    output.mkdir(parents=True, exist_ok=True)

    # Clone latest repo
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPOSITORY, str(repo)])
    run(["git", "fetch", "--all"], cwd=repo)
    run(["git", "checkout", "main"], cwd=repo)
    run(["git", "reset", "--hard", "origin/main"], cwd=repo)
    run(["git", "log", "-1", "--oneline"], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q", "neurokit2==0.2.10", "imbalanced-learn", "matplotlib"])

    manifest = require(repo / "configs/approved_feature_manifest_v0_4.json")

    # Run Deep Learning & Multimodal Hybrid 8-Fold Training
    run([
        sys.executable, "run_deep_learning_oof.py",
        "--hdf5", str(primary_h5),
        "--metadata", str(metadata_csv),
        "--features", str(features_csv),
        "--approved-feature-manifest", str(manifest),
        "--output", str(output),
        "--epochs", "20",
        "--batch-size", "64",
    ], cwd=repo)

    # Artifact manifest
    files = sorted(
        ({"path": str(path.relative_to(output)), "bytes": path.stat().st_size}
         for path in output.rglob("*") if path.is_file()),
        key=lambda item: item["path"],
    )
    (output / "artifact_manifest.json").write_text(json.dumps(files, indent=2))

    print("\nPhase 6B/6C Deep Learning Artifacts:", flush=True)
    for item in files:
        print(f"  {item['path']} ({item['bytes'] / 1024:.1f} KiB)", flush=True)


if __name__ == "__main__":
    main()
