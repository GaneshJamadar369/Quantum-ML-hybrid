"""Pinned Kaggle runner for the film advanced q4 fusion screen."""
from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
REVISION = "dfc416219affb9e6dbff3b4aa3547772510ec601"
FUSION = "film"


def run(command, cwd=None):
    print(">>>", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def unique(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} under {root}, found {matches}")
    return matches[0]


def main():
    notebooks = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    preprocessing = notebooks / "aquire-med-preprocessing-pipeline"
    feature_source = notebooks / "aquire-med-full-feature-repair"
    transformer_source = notebooks / "aquire-med-compact-ecg-transformer-representation"
    hdf5 = preprocessing / "aquire-artifacts/primary_development_100hz.h5"
    metadata = preprocessing / "aquire-artifacts/processing_metadata_development.csv"
    normalizers = preprocessing / "artifacts/g5/normalizers"
    features = feature_source / "full-feature-repair/feature-evidence-v0-4/deployable_features_with_clinical_composites.csv"
    transformer_dir = transformer_source / "transformer-representation-v1"
    for required in (hdf5, metadata, features, transformer_dir, normalizers):
        if not required.exists():
            raise FileNotFoundError(required)
    for fold in range(1, 9):
        for required in (
            transformer_dir / f"outer_fold_{fold}_encoder.pt",
            transformer_dir / f"outer_fold_{fold}_representations.npz",
            normalizers / f"normalizer_holdout_fold_{fold}.json",
        ):
            if not required.exists():
                raise FileNotFoundError(required)
    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", "--filter=blob:none", REPO, repo])
    run(["git", "fetch", "origin", REVISION, "--depth", "1"], cwd=repo)
    run(["git", "checkout", "--detach", REVISION], cwd=repo)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != REVISION:
        raise RuntimeError(f"Source revision mismatch: {actual} != {REVISION}")
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", repo])
    run([sys.executable, "-m", "pytest", "-q", "tests/test_advanced_fusion.py", "tests/test_supervised_qubit_scaling.py"], cwd=repo)
    run([
        sys.executable, "run_advanced_quantum_fusion_screen.py",
        "--hdf5", hdf5,
        "--metadata", metadata,
        "--features", features,
        "--manifest", repo / "configs/approved_feature_manifest_v0_4.json",
        "--transformer-dir", transformer_dir,
        "--normalizers", normalizers,
        "--output", f"/kaggle/working/advanced-fusion-{FUSION}-q4",
        "--fusion", FUSION,
        "--fusion-per-class", "2000",
        "--quantum-per-class", "500",
        "--fusion-epochs", "10",
        "--quantum-epochs", "20",
        "--batch-size", "128",
        "--encoder-batch-size", "256",
        "--bootstrap-iterations", "2000",
        "--seed", "20260924",
    ], cwd=repo)
    print(f"Completed {FUSION} advanced q4 screen; folds 9 and 10 remained sealed", flush=True)


if __name__ == "__main__":
    main()
