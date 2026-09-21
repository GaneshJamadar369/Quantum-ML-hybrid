"""Kaggle runner for the fold-coherent waveform VQC screen."""

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


def main() -> None:
    representation_dir = Path(
        "/kaggle/input/notebooks/swayamjeetbhagat4/"
        "aquire-med-waveform-representation-v1/waveform-representation-v1"
    )
    if not representation_dir.exists():
        raise FileNotFoundError(representation_dir)
    metadata = Path(
        "/kaggle/input/notebooks/swayamjeetbhagat4/"
        "aquire-med-preprocessing-pipeline/aquire-artifacts/processing_metadata_development.csv"
    )
    if not metadata.exists():
        raise FileNotFoundError(metadata)
    output = Path("/kaggle/working/waveform-quantum-q4-screen")
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
            "pip",
            "install",
            "-q",
            "pennylane==0.45.1",
            "pennylane-lightning==0.45.0",
        ]
    )
    run([sys.executable, "-m", "pytest", "-q", "tests/test_quantum_models.py"], cwd=repo)
    run(
        [
            sys.executable,
            "run_waveform_quantum_core_screen.py",
            "--representations",
            str(representation_dir),
            "--metadata",
            str(metadata),
            "--output",
            str(output),
            "--qubits",
            "4",
            "--per-class",
            "500",
            "--epochs",
            "20",
            "--batch-size",
            "64",
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
    print("=== Fold-coherent waveform VQC screen complete; folds 9 and 10 sealed ===")


if __name__ == "__main__":
    main()
