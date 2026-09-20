"""Cloud runner for AQUIRE-Med Phase 6Q Quantum ML & Phase 7 Locked Holdout Evaluation."""

import os
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: str) -> None:
    print(f"\n>>> {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, text=True)
    if res.returncode != 0:
        print(f"Command failed with exit code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)


def main() -> None:
    print("=== AQUIRE-Med Phase 6Q (Quantum) & Phase 7 (Holdout) Runner ===", flush=True)

    # 1. Install PennyLane and PyTorch dependencies
    print("\n--- Installing dependencies ---", flush=True)
    run_cmd("pip install pennylane pennylane-lightning xgboost shap")

    # 2. Clone latest repo
    repo_url = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
    work_dir = Path("/tmp/Quantum-ML-hybrid")
    if work_dir.exists():
        run_cmd(f"rm -rf {work_dir}")

    run_cmd(f"git clone {repo_url} {work_dir}")
    os.chdir(work_dir)

    # 3. Locate inputs
    input_root = Path("/kaggle/input")
    h5_candidates = list(input_root.glob("**/primary_development_100hz.h5"))
    meta_candidates = list(input_root.glob("**/processing_metadata_development.csv"))
    feat_candidates = list(input_root.glob("**/deployable_features_with_clinical_composites.csv"))

    if not h5_candidates:
        raise FileNotFoundError("primary_development_100hz.h5 not found in /kaggle/input")
    if not meta_candidates:
        raise FileNotFoundError("processing_metadata_development.csv not found in /kaggle/input")
    if not feat_candidates:
        raise FileNotFoundError("deployable_features_with_clinical_composites.csv not found in /kaggle/input")

    h5_path = h5_candidates[0]
    meta_path = meta_candidates[0]
    feat_path = feat_candidates[0]
    manifest_path = Path("aquire_preprocessing/feature_manifest.json")

    out_dir_6q = Path("/kaggle/working/g6-quantum-baselines")
    out_dir_7 = Path("/kaggle/working/g7-locked-holdout")

    # 4. Execute Phase 6Q Quantum Baselines
    print("\n--- Running Phase 6Q Quantum Baselines (Folds 1-8 OOF) ---", flush=True)
    run_cmd(
        f"python3 run_quantum_baselines.py "
        f"--h5-path {h5_path} "
        f"--metadata-path {meta_path} "
        f"--features-csv {feat_path} "
        f"--manifest-path {manifest_path} "
        f"--output-dir {out_dir_6q} "
        f"--epochs 15 "
        f"--batch-size 64 "
        f"--lr 1e-3"
    )

    # 5. Execute Phase 7 Locked Holdout Evaluation
    print("\n--- Running Phase 7 Locked Holdout Single-Pass Evaluation (Folds 9 & 10) ---", flush=True)
    run_cmd(
        f"python3 run_sealed_holdout_evaluation.py "
        f"--h5-path {h5_path} "
        f"--metadata-path {meta_path} "
        f"--features-csv {feat_path} "
        f"--manifest-path {manifest_path} "
        f"--output-dir {out_dir_7}"
    )

    print("\n=== All Quantum & Holdout Phases Completed Successfully ===", flush=True)


if __name__ == "__main__":
    main()
