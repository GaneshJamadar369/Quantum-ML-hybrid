"""Private Kaggle GPU runner for residual q4 feature-routing research."""

from pathlib import Path
import subprocess
import sys


REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
REVISION = "edebd85155599d7a587d1b58c5f9d6dce5976b52"


def run(command, cwd=None):
    print(">>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def locate_input(preferred: Path, marker: str, *, directory: bool = False) -> Path:
    if preferred.exists():
        return preferred
    matches = sorted(Path("/kaggle/input").rglob(marker))
    print(f"Preferred input missing: {preferred}; marker matches={matches}", flush=True)
    if len(matches) != 1:
        raise FileNotFoundError(preferred)
    return matches[0].parent if directory else matches[0]


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Residual q4 study requires a Kaggle GPU")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    source = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    representations = locate_input(
        source / "aquire-med-compact-ecg-transformer-representation/transformer-representation-v1",
        "outer_fold_1_representations.npz",
        directory=True,
    )
    metadata = locate_input(
        source / "aquire-med-preprocessing-pipeline/aquire-artifacts/processing_metadata_development.csv",
        "processing_metadata_development.csv",
    )
    features = locate_input(
        source / "aquire-med-full-feature-repair/full-feature-repair/feature-evidence-v0-4/deployable_features_with_clinical_composites.csv",
        "deployable_features_with_clinical_composites.csv",
    )

    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPO, str(repo)])
    run(["git", "checkout", REVISION], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(repo)])
    run([
        sys.executable, "-m", "pytest", "-q",
        "tests/test_feature_routing.py",
        "tests/test_residual_quantum_fusion.py",
        "tests/test_quantum_models.py",
    ], cwd=repo)
    run([
        sys.executable, "run_residual_quantum_fusion_screen.py",
        "--representations", str(representations),
        "--metadata", str(metadata),
        "--features", str(features),
        "--manifest", "configs/approved_feature_manifest_v0_4.json",
        "--output", "/kaggle/working/residual-q4-feature-routing",
        "--per-class", "500",
        "--epochs", "20",
        "--batch-size", "128",
        "--seed", "20260923",
        "--device", "cuda",
    ], cwd=repo)
    print("Residual q4 routing screen complete; folds 9 and 10 sealed", flush=True)


if __name__ == "__main__":
    main()
