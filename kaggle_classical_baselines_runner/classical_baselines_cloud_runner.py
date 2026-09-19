"""Kaggle runner: Phase 6A — OOF classical baselines with G5.5 conditioning.

Execution order on Kaggle:
  1. Clone repo at commit a92bcf8 (G5.5 — pre-modelling pipeline complete)
  2. pip install the package + dependencies
  3. run_classical_baselines.py  (OOF LR/RF/XGB/MLP/SVC + Platt calibration)
  4. run_shap_audit.py           (tree-model SHAP on fold-8 held-out records)
  5. run_conformal_analysis.py   (RAPS conformal sets from OOF predictions)

Required Kaggle inputs (attach as datasets):
  - aquire-med-preprocessing-pipeline / aquire-artifacts
      primary_development_100hz.h5
      processing_metadata_development.csv
      manifests/patient_manifest.csv
  - Quantum-ML-hybrid output from previous run (features CSV + manifest)
      deployable_features_v0_4_development.csv
      feature-evidence-v0-4/approved_feature_manifest_v0_4.json
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


CODE_COMMIT = "8ac91d6010bfca4bac742efa881adf08e870b62c"
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

    # ------------------------------------------------------------------ #
    # Locate mounted inputs
    # ------------------------------------------------------------------ #
    preprocessing = (
        root / "notebooks/swayamjeetbhagat4/aquire-med-preprocessing-pipeline/aquire-artifacts"
    )
    # Features from the G5F full-feature-repair run
    feature_repair = (
        root / "notebooks/swayamjeetbhagat4/aquire-med-full-feature-repair/full-feature-repair"
    )

    primary = require(preprocessing / "primary_development_100hz.h5")
    metadata = require(preprocessing / "processing_metadata_development.csv")
    features_csv = require(
        feature_repair / "feature-evidence-v0-4/deployable_features_with_clinical_composites.csv"
    )

    output = Path("/kaggle/working/g6-classical-baselines")
    output.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Clone and install the package at the latest commit
    # ------------------------------------------------------------------ #
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPOSITORY, str(repo)])
    run(["git", "fetch", "--all"], cwd=repo)
    run(["git", "checkout", "main"], cwd=repo)
    run(["git", "reset", "--hard", "origin/main"], cwd=repo)
    run(["git", "log", "-1", "--oneline"], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q",
         "neurokit2==0.2.10",
         "imbalanced-learn",
         "shap",
         "matplotlib",
    ])

    manifest = require(repo / "configs/approved_feature_manifest_v0_4.json")

    # ------------------------------------------------------------------ #
    # Step 1 — OOF classical baselines (all three MI label policies)
    # ------------------------------------------------------------------ #
    for policy in ["all", "supported_or_high", "high_only"]:
        policy_out = output / policy
        run([
            sys.executable, "run_classical_baselines.py",
            "--features", str(features_csv),
            "--metadata", str(metadata),
            "--approved-feature-manifest", str(manifest),
            "--output", str(policy_out),
            "--mi-label-policy", policy,
        ], cwd=repo)

    # Primary policy is "all" — use its predictions for downstream steps
    primary_policy_dir = output / "all"

    # ------------------------------------------------------------------ #
    # Step 2 — SHAP attribution audit (tree models, fold-8 held-out)
    # ------------------------------------------------------------------ #
    shap_out = output / "shap"
    run([
        sys.executable, "run_shap_audit.py",
        "--features", str(features_csv),
        "--metadata", str(metadata),
        "--approved-feature-manifest", str(manifest),
        "--output", str(shap_out),
    ], cwd=repo)

    # ------------------------------------------------------------------ #
    # Step 3 — RAPS conformal prediction sets (fold-8 calibration, 90 %)
    # ------------------------------------------------------------------ #
    conformal_out = output / "conformal"
    run([
        sys.executable, "run_conformal_analysis.py",
        "--predictions", str(primary_policy_dir / "classical_oof_predictions.csv"),
        "--output", str(conformal_out),
    ], cwd=repo)

    # ------------------------------------------------------------------ #
    # Artifact manifest
    # ------------------------------------------------------------------ #
    files = sorted(
        ({"path": str(path.relative_to(output)), "bytes": path.stat().st_size}
         for path in output.rglob("*") if path.is_file()),
        key=lambda item: item["path"],
    )
    (output / "artifact_manifest.json").write_text(json.dumps(files, indent=2))

    print("\nPhase 6A artifacts:", flush=True)
    for item in files:
        print(f"  {item['path']} ({item['bytes'] / 1024:.1f} KiB)", flush=True)


if __name__ == "__main__":
    main()
