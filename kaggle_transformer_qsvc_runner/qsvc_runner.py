"""Private Kaggle IQP-QSVC screen on frozen Transformer ECG embeddings."""

from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
REVISION = "REVISION_TO_PIN"


def run(command, cwd=None):
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    root = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    representations = root / "aquire-med-compact-ecg-transformer-representation/transformer-representation-v1"
    metadata = root / "aquire-med-preprocessing-pipeline/aquire-artifacts/processing_metadata_development.csv"
    for path in (representations, metadata):
        if not path.exists():
            raise FileNotFoundError(path)
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPO, str(repo)])
    run(["git", "checkout", REVISION], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q", "pennylane==0.45.1", "pennylane-lightning==0.45.0"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_quantum_models.py"], cwd=repo)
    run([sys.executable, "run_qsvc_hqnn_screen.py", "--task", "qsvc", "--representations", str(representations),
         "--metadata", str(metadata), "--output", "/kaggle/working/transformer-qsvc-screen",
         "--per-class", "500", "--seed", "20260922"], cwd=repo)
    print("Transformer QSVC screen complete; folds 9 and 10 sealed", flush=True)


if __name__ == "__main__":
    main()
