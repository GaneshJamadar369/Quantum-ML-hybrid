"""Kaggle GPU runner for fold-coherent waveform representation export."""

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
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
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
    preprocessing_root = (
        root / "notebooks/swayamjeetbhagat4/aquire-med-preprocessing-pipeline"
    )
    preprocessing = preprocessing_root / "aquire-artifacts"
    primary_h5 = require(preprocessing / "primary_development_100hz.h5")
    metadata = require(preprocessing / "processing_metadata_development.csv")
    normalizers = require(preprocessing_root / "artifacts/g5/normalizers")
    for held_out in range(1, 9):
        require(normalizers / f"normalizer_holdout_fold_{held_out}.json")

    output = Path("/kaggle/working/waveform-representation-v1")
    output.mkdir(parents=True, exist_ok=True)
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPOSITORY, str(repo)])
    run(["git", "reset", "--hard", "origin/main"], cwd=repo)
    run(["git", "log", "-1", "--oneline"], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo)])
    run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_waveform_representation.py",
            "tests/test_deep_learning.py",
        ],
        cwd=repo,
    )
    run(
        [
            sys.executable,
            "run_waveform_representation_export.py",
            "--hdf5",
            str(primary_h5),
            "--metadata",
            str(metadata),
            "--normalizers",
            str(normalizers),
            "--output",
            str(output),
            "--epochs",
            "20",
            "--batch-size",
            "128",
            "--seed",
            "20260922",
        ],
        cwd=repo,
    )
    files = sorted(
        (
            {"path": str(path.relative_to(output)), "bytes": path.stat().st_size}
            for path in output.rglob("*")
            if path.is_file()
        ),
        key=lambda item: item["path"],
    )
    (output / "artifact_manifest.json").write_text(json.dumps(files, indent=2))
    print("=== Waveform representation export complete; folds 9 and 10 sealed ===")


if __name__ == "__main__":
    main()
