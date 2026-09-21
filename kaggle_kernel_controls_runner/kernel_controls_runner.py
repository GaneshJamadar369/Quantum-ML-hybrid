"""Kaggle Phase 6Q-B: stronger matched classical-kernel controls."""

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
    print("=== AQUIRE-Med Phase 6Q-B: matched kernel controls ===", flush=True)
    root = Path("/kaggle/input")
    preprocessing = root / "notebooks/swayamjeetbhagat4/aquire-med-preprocessing-pipeline/aquire-artifacts"
    feature_repair = root / "notebooks/swayamjeetbhagat4/aquire-med-full-feature-repair/full-feature-repair"
    metadata_csv = require(preprocessing / "processing_metadata_development.csv")
    features_csv = require(
        feature_repair / "feature-evidence-v0-4/deployable_features_with_clinical_composites.csv"
    )
    output = Path("/kaggle/working/g6b-kernel-controls")
    output.mkdir(parents=True, exist_ok=True)

    repo = Path("/tmp/Quantum-ML-hybrid")
    if repo.exists():
        run(["rm", "-rf", str(repo)])
    run(["git", "clone", REPOSITORY, str(repo)])
    run(["git", "reset", "--hard", "origin/main"], cwd=repo)
    run(["git", "log", "-1", "--oneline"], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "pennylane==0.45.1",
        "pennylane-lightning==0.45.0",
    ])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_quantum_models.py"], cwd=repo)

    manifest = require(repo / "configs/approved_feature_manifest_v0_4.json")
    run([
        sys.executable,
        "run_quantum_baselines.py",
        "--metadata-path",
        str(metadata_csv),
        "--features-csv",
        str(features_csv),
        "--manifest-path",
        str(manifest),
        "--output-dir",
        str(output),
        "--models",
        "qsvm",
        "rbf_svc_angle_z8",
        "polynomial_svc_angle_z8",
        "laplacian_svc_angle_z8",
        "product_cosine_svc_angle_z8",
        "--qsvm-per-class",
        "500",
        "--bootstrap-iterations",
        "2000",
    ], cwd=repo)

    files = sorted(
        ({"path": str(path.relative_to(output)), "bytes": path.stat().st_size}
         for path in output.rglob("*") if path.is_file()),
        key=lambda item: item["path"],
    )
    (output / "artifact_manifest.json").write_text(json.dumps(files, indent=2))
    print("=== Phase 6Q-B complete; folds 9 and 10 remain untouched ===", flush=True)


if __name__ == "__main__":
    main()
