"""Private Kaggle comparison of q4 and h128 classical Transformer heads."""

from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
REVISION = "5202e07"


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
    run([sys.executable, "run_transformer_classical_comparison.py",
         "--representations", str(representations), "--metadata", str(metadata),
         "--output", "/kaggle/working/transformer-classical-comparison",
         "--per-class", "500", "--seed", "20260922"], cwd=repo)
    print("Transformer classical comparison complete; folds 9 and 10 sealed", flush=True)


if __name__ == "__main__":
    main()
