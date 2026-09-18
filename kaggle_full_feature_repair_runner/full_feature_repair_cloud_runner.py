"""Run approved v0.4 extraction and repeat the full G5F evidence gate."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


CODE_COMMIT = "54902add83cdce9fcc36e27e2b8f9e9934ab2880"
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
    preprocessing = (
        root / "notebooks/swayamjeetbhagat4/aquire-med-preprocessing-pipeline/aquire-artifacts"
    )
    primary = require(preprocessing / "primary_development_100hz.h5")
    metadata = require(preprocessing / "processing_metadata_development.csv")
    manifest = require(preprocessing / "manifests/patient_manifest.csv")
    reference = require(
        root / "datasets/antonymgitau/ptb-xl-a-comprehensive-ecg-feature-dataset/"
        "ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/"
        "features/ecgdeli_features.csv"
    )

    output = Path("/kaggle/working/full-feature-repair")
    output.mkdir(parents=True, exist_ok=True)
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPOSITORY, str(repo)])
    run(["git", "checkout", CODE_COMMIT], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q", "neurokit2==0.2.10"])

    features = output / "deployable_features_v0_4_development.csv"
    run([
        sys.executable, "run_feature_reextraction.py",
        "--hdf5", str(primary),
        "--output", str(features),
        "--shard-size", "256",
    ], cwd=repo)
    evidence = output / "feature-evidence-v0-4"
    run([
        sys.executable, "run_feature_evidence.py",
        "--features", str(features),
        "--metadata", str(metadata),
        "--manifest", str(manifest),
        "--primary-hdf5", str(primary),
        "--reference-features", str(reference),
        "--reference-pairs", "configs/deployable_reference_pairs_v0_4.json",
        "--feature-family-decisions", "configs/feature_family_decisions.json",
        "--output", str(evidence),
    ], cwd=repo)

    files = sorted(
        ({"path": str(path.relative_to(output)), "bytes": path.stat().st_size}
         for path in output.rglob("*") if path.is_file()),
        key=lambda item: item["path"],
    )
    (output / "artifact_manifest.json").write_text(json.dumps(files, indent=2))
    print("\nFull feature repair artifacts:", flush=True)
    for item in files:
        print(f"  {item['path']} ({item['bytes'] / 1024:.1f} KiB)", flush=True)


if __name__ == "__main__":
    main()
