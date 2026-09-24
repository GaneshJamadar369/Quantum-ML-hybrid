"""Private Kaggle GPU runner for independent dual-route fusion research."""

from pathlib import Path
import subprocess
import sys


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
        raise RuntimeError("Independent dual-route study requires a Kaggle GPU")
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

    # Clone repo from GitHub
    repo = Path("/tmp/Quantum-ML-hybrid")
    if not repo.exists():
        run(["git", "clone", "--depth", "1",
             "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git",
             str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])

    # Install quantum dependencies
    run([sys.executable, "-m", "pip", "install", "-q",
         "pennylane", "pennylane-lightning",
         "scikit-learn>=1.4", "scipy>=1.12", "h5py"])
    run([sys.executable, "-m", "pip", "install", "-q",
         "neurokit2==0.2.10", "imbalanced-learn", "matplotlib"])

    # Verify route config
    route_config = repo / "configs" / "independent_route_v1.json"
    print(f"input: {route_config}", flush=True)
    if not route_config.exists():
        raise FileNotFoundError(route_config)

    # Run the experiment
    run([
        sys.executable, "run_independent_dual_route_screen.py",
        "--representations", str(representations),
        "--metadata", str(metadata),
        "--features", str(features),
        "--manifest", "configs/approved_feature_manifest_v0_4.json",
        "--route-config", str(route_config),
        "--output", "/kaggle/working/independent-dual-route-v1",
        "--per-class", "500",
        "--epochs", "20",
        "--batch-size", "128",
        "--seed", "42",
        "--device", "cuda",
    ], cwd=repo)
    print("Independent dual-route fusion screen complete; folds 9 and 10 sealed", flush=True)


if __name__ == "__main__":
    main()
