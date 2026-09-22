"""Kaggle CPU runner for matched q4 VQC/classical heads on Transformer ECG vectors."""

from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
REVISION = "9678e1f"


def run(command, cwd=None):
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    source = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    representations = source / "aquire-med-transformer-representation-v1/transformer-representation-v1"
    metadata = source / "aquire-med-preprocessing-pipeline/aquire-artifacts/processing_metadata_development.csv"
    for path in (representations, metadata):
        if not path.exists():
            raise FileNotFoundError(path)
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPO, str(repo)])
    run(["git", "checkout", REVISION], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q", "pennylane==0.45.1", "pennylane-lightning==0.45.0"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_quantum_models.py", "tests/test_waveform_representation.py"], cwd=repo)
    run([sys.executable, "run_waveform_quantum_core_screen.py",
         "--representations", str(representations), "--metadata", str(metadata),
         "--output", "/kaggle/working/transformer-quantum-q4-screen",
         "--qubits", "4", "--per-class", "500", "--epochs", "20",
         "--batch-size", "64", "--seed", "20260922"], cwd=repo)
    print("Transformer q4 screen complete; folds 9 and 10 sealed", flush=True)


if __name__ == "__main__":
    main()
