"""Private Kaggle GPU runner for q4 replicated across a 16-qubit VQC."""

from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
REVISION = "5bf5d69ae355ea0499eb6c81187761a2257bd116"


def run(command, cwd=None):
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def locate_input(preferred: Path, marker: str) -> Path:
    if preferred.exists():
        return preferred
    matches = sorted(Path("/kaggle/input").rglob(marker))
    print(f"Preferred input missing: {preferred}; marker matches={matches}", flush=True)
    if len(matches) != 1:
        raise FileNotFoundError(preferred)
    return matches[0].parent if preferred.suffix == "" else matches[0]


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("The 16-qubit exact-statevector screen requires a Kaggle GPU")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    source = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    representations = locate_input(
        source / "aquire-med-compact-ecg-transformer-representation/transformer-representation-v1",
        "outer_fold_1_representations.npz",
    )
    metadata = locate_input(
        source / "aquire-med-preprocessing-pipeline/aquire-artifacts/processing_metadata_development.csv",
        "processing_metadata_development.csv",
    )
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPO, str(repo)])
    run(["git", "checkout", REVISION], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q", "pennylane==0.45.1", "pennylane-lightning==0.45.0"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_supervised_qubit_scaling.py"], cwd=repo)
    run([
        sys.executable, "run_q4_on_16_qubit_capacity_screen.py",
        "--representations", str(representations),
        "--metadata", str(metadata),
        "--output", "/kaggle/working/q4-on-16q-capacity-screen",
        "--per-class", "500", "--epochs", "20", "--batch-size", "16",
        "--seed", "20260922", "--device", "cuda",
    ], cwd=repo)
    print("q4-on-16q OOF screen complete; folds 9 and 10 sealed", flush=True)


if __name__ == "__main__":
    main()
