"""Pinned Kaggle GPU runner for nested q4 seeds 16180/14142."""

from pathlib import Path
import subprocess
import sys


SOURCE_COMMIT = "a4e32dfca2e3165adc430d92c69739b9e109cae7"
SEEDS = (16180, 14142)


def command(values, cwd=None):
    print(">>>", " ".join(str(value) for value in values), flush=True)
    subprocess.run([str(value) for value in values], cwd=cwd, check=True)


def locate(preferred: Path, marker: str, *, directory=False) -> Path:
    if preferred.exists():
        return preferred
    matches = sorted(Path("/kaggle/input").rglob(marker))
    print(f"locate {marker}: {matches}", flush=True)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {marker}, found {matches}")
    return matches[0].parent if directory else matches[0]


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Nested q4 optimization requires a Kaggle GPU")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    source = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    representations = locate(
        source / "aquire-med-compact-ecg-transformer-representation/transformer-representation-v1",
        "outer_fold_1_representations.npz", directory=True,
    )
    metadata = locate(
        source / "aquire-med-preprocessing-pipeline/aquire-artifacts/processing_metadata_development.csv",
        "processing_metadata_development.csv",
    )
    features = locate(
        source / "aquire-med-full-feature-repair/full-feature-repair/feature-evidence-v0-4/deployable_features_with_clinical_composites.csv",
        "deployable_features_with_clinical_composites.csv",
    )
    repo = Path("/tmp/Quantum-ML-hybrid")
    command(["git", "init", str(repo)])
    command(["git", "remote", "add", "origin", "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"], cwd=repo)
    command(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=repo)
    command(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"Source mismatch: {actual} != {SOURCE_COMMIT}")
    command([sys.executable, "-m", "pytest", "-q", "tests/test_nested_q4_optimization.py"], cwd=repo)
    base = [
        sys.executable, "run_nested_q4_optimization.py",
        "--representations", representations,
        "--metadata", metadata,
        "--features", features,
        "--manifest", "configs/approved_feature_manifest_v0_4.json",
        "--output", "/kaggle/working/nested-q4-seeds-b",
        "--seeds", *SEEDS,
        "--search-per-class", "750", "--final-per-class", "2000",
        "--max-epochs", "30", "--batch-size", "128",
        "--bootstrap-iterations", "1000", "--device", "cuda",
    ]
    command(base + ["--preflight-only"], cwd=repo)
    command(base, cwd=repo)


if __name__ == "__main__":
    main()
