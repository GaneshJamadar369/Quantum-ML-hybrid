"""Private Kaggle HQNN screening job with frozen source revision."""

from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
REVISION = "10f280d"


def run(command, cwd=None):
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    root = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    representations = root / "aquire-med-waveform-representation-v1/waveform-representation-v1"
    metadata = root / "aquire-med-preprocessing-pipeline/aquire-artifacts/processing_metadata_development.csv"
    features = root / "aquire-med-full-feature-repair/full-feature-repair/feature-evidence-v0-4/deployable_features_with_clinical_composites.csv"
    for path in (representations, metadata, features):
        if not path.exists():
            raise FileNotFoundError(path)
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPO, str(repo)])
    run(["git", "checkout", REVISION], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q", "pennylane==0.45.1", "pennylane-lightning==0.45.0"])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_quantum_models.py"], cwd=repo)
    run([sys.executable, "run_qsvc_hqnn_screen.py", "--task", "hqnn", "--representations", str(representations),
         "--metadata", str(metadata), "--features", str(features),
         "--manifest", str(repo / "configs/approved_feature_manifest_v0_4.json"),
         "--output", "/kaggle/working/hqnn-screen", "--per-class", "500",
         "--epochs", "20", "--seed", "20260922"], cwd=repo)
    print("HQNN screen complete; folds 9 and 10 sealed", flush=True)


if __name__ == "__main__":
    main()
