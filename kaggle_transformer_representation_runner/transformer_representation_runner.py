"""Kaggle GPU runner for the compact patch-Transformer representation."""

from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
REVISION = "REPLACE_WITH_COMMIT"


def run(command, cwd=None):
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    source = Path("/kaggle/input/notebooks/swayamjeetbhagat4/aquire-med-preprocessing-pipeline")
    hdf5 = require(source / "aquire-artifacts/primary_development_100hz.h5")
    metadata = require(source / "aquire-artifacts/processing_metadata_development.csv")
    normalizers = require(source / "artifacts/g5/normalizers")
    for fold in range(1, 9):
        require(normalizers / f"normalizer_holdout_fold_{fold}.json")
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPO, str(repo)])
    run(["git", "checkout", REVISION], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo)])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_transformer_representation.py", "tests/test_waveform_representation.py"], cwd=repo)
    run([sys.executable, "run_transformer_representation_export.py",
         "--hdf5", str(hdf5), "--metadata", str(metadata),
         "--normalizers", str(normalizers),
         "--output", "/kaggle/working/transformer-representation-v1",
         "--max-epochs", "20", "--batch-size", "128",
         "--lr", "0.0003", "--seed", "20260922"], cwd=repo)
    print("Transformer representation complete; folds 9 and 10 sealed", flush=True)


if __name__ == "__main__":
    main()
