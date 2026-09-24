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
    input_root = Path("/kaggle/input")
    hdf5 = unique(input_root, "primary_development_100hz.h5")
    metadata = unique(input_root, "processing_metadata_development.csv")
    features = unique(input_root, "deployable_features_with_clinical_composites.csv")
    transformer_dir = unique(input_root, "outer_fold_1_encoder.pt").parent
    normalizers = unique(input_root, "normalizer_holdout_fold_1.json").parent
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
