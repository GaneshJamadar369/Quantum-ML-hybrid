"""
AQUIRE-Med Kaggle Cloud Autonomous Runner
==========================================
Executes the full G0-G5 gate verification, development fold preprocessing,
QC calibration, and waveform normalization directly in Kaggle Cloud.
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path

def banner(msg: str):
    print("\n" + "=" * 75)
    print(f"  {msg}")
    print("=" * 75, flush=True)

def run_cmd(cmd, cwd=None, env=None):
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n>>> Running: {cmd_str}", flush=True)
    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    elapsed = time.time() - start
    print(f"\n[Exit code: {proc.returncode} | Duration: {elapsed:.1f}s]", flush=True)
    if proc.returncode != 0:
        print(f"ERROR: Command failed with code {proc.returncode}", flush=True)
        sys.exit(proc.returncode)

def main():
    banner("AQUIRE-Med Cloud Preprocessing & Verification Pipeline")
    print(f"Python: {sys.version}")
    print(f"CWD:    {Path.cwd()}", flush=True)

    # 1. Discover Datasets
    banner("STEP 1: Discovering Datasets in /kaggle/input")
    input_dir = Path("/kaggle/input")
    if not input_dir.exists():
        raise FileNotFoundError("/kaggle/input does not exist!")

    print(f"Listing /kaggle/input:")
    for item in input_dir.glob("*"):
        print(f"  - {item}")

    # Search for ptbxl_database.csv
    ptbxl_matches = list(input_dir.glob("**/ptbxl_database.csv"))
    if not ptbxl_matches:
        raise FileNotFoundError("Could not find ptbxl_database.csv anywhere in /kaggle/input!")
    
    # Prefer 1.0.3 if available
    ptbxl_root = None
    for m in ptbxl_matches:
        if "1.0.3" in str(m) or "1-0-3" in str(m):
            ptbxl_root = m.parent
            break
    if ptbxl_root is None:
        ptbxl_root = ptbxl_matches[0].parent
    print(f"\n✓ Found PTB-XL Root:  {ptbxl_root}")

    # Search for 12sl_features.csv (PTB-XL+)
    ptbxlp_matches = list(input_dir.glob("**/12sl_features.csv"))
    if not ptbxlp_matches:
        raise FileNotFoundError("Could not find 12sl_features.csv anywhere in /kaggle/input!")
    
    feat_parent = ptbxlp_matches[0].parent
    ptbxlp_root = feat_parent.parent if feat_parent.name == "features" else feat_parent
    print(f"✓ Found PTB-XL+ Root: {ptbxlp_root}")

    output_root = Path("/kaggle/working/aquire-artifacts")
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"✓ Output Directory:   {output_root}", flush=True)

    # 2. Clone Latest Code from GitHub
    banner("STEP 2: Syncing Code from GitHub")
    repo_dir = Path("/kaggle/working/Quantum-ML-hybrid")
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    
    run_cmd(["git", "clone", "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git", str(repo_dir)])

    # 3. Install Dependencies
    banner("STEP 3: Installing Requirements & Package")
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "wfdb", "h5py", "scipy", "scikit-learn", "xgboost", "matplotlib"])
    run_cmd([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo_dir)])

    # Prepare environment variables
    env = os.environ.copy()
    env["AQUIRE_PTBXL_ROOT"] = str(ptbxl_root)
    env["AQUIRE_PTBXLP_ROOT"] = str(ptbxlp_root)
    env["AQUIRE_OUTPUT_ROOT"] = str(output_root)
    env["PYTHONUNBUFFERED"] = "1"

    # 4. Gate Verification & Visual Audit
    banner("STEP 4: Running G0-G3 Gates & Visual Audit")
    run_cmd([sys.executable, "run_gates.py", "--visual-audit"], cwd=str(repo_dir), env=env)

    # 5. Execute Pipeline on Development Folds (Folds 1-8)
    banner("STEP 5: Running Preprocessing Pipeline (Folds 1-8)")
    run_cmd([sys.executable, "run_pipeline.py", "--role", "development"], cwd=str(repo_dir), env=env)

    # 6. Run QC Calibration
    banner("STEP 6: Running QC Calibration")
    qc_metrics_csv = output_root / "qc_metrics_development.csv"
    if qc_metrics_csv.exists():
        run_cmd([
            sys.executable, "run_qc_calibration.py",
            "--metrics", str(qc_metrics_csv),
            "--spec", "configs/qc_detectors.json"
        ], cwd=str(repo_dir), env=env)
    else:
        print(f"Note: {qc_metrics_csv} not found, checking primary output directory...")

    # 7. Run Normalization
    banner("STEP 7: Fitting Waveform Normalization")
    primary_h5 = output_root / "primary_development_100hz.h5"
    if primary_h5.exists():
        run_cmd([
            sys.executable, "run_normalization.py",
            "--hdf5", str(primary_h5)
        ], cwd=str(repo_dir), env=env)
    else:
        print(f"Warning: {primary_h5} not found, skipping normalization.")

    # 8. Run Parity Check
    banner("STEP 8: Running Parity Check")
    run_cmd([sys.executable, "run_parity_check.py"], cwd=str(repo_dir), env=env)

    # 9. Copy all artifacts to working output for download
    banner("STEP 9: Preparing Artifacts for Download")
    artifacts_dir = repo_dir / "artifacts"
    if artifacts_dir.exists():
        dest = Path("/kaggle/working/artifacts")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(artifacts_dir, dest)
        print(f"✓ Copied {artifacts_dir} to {dest}")

    print("\nListing /kaggle/working:")
    for item in Path("/kaggle/working").glob("**/*"):
        if item.is_file():
            print(f"  - {item.relative_to('/kaggle/working')} ({item.stat().st_size / (1024*1024):.2f} MB)")

    banner("PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
