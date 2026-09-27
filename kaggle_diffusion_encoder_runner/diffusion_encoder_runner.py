"""Pinned Kaggle GPU runner for diffusion-pretrained ECG h128 and q4 VQC."""

from pathlib import Path
import subprocess
import sys


SOURCE_COMMIT = "5725c05e97188872e3d229f7c0a85597b35c8648"


def command(values, cwd=None):
    print(">>>", " ".join(str(value) for value in values), flush=True)
    subprocess.run([str(value) for value in values], cwd=cwd, check=True)


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Diffusion encoder experiment requires a Kaggle GPU")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    source = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    preprocessing = source / "aquire-med-preprocessing-pipeline"
    feature_source = source / "aquire-med-full-feature-repair"
    hdf5 = require(preprocessing / "aquire-artifacts/primary_development_100hz.h5")
    metadata = require(preprocessing / "aquire-artifacts/processing_metadata_development.csv")
    normalizers = require(preprocessing / "artifacts/g5/normalizers")
    features = require(
        feature_source
        / "full-feature-repair/feature-evidence-v0-4/deployable_features_with_clinical_composites.csv"
    )
    for fold in range(1, 9):
        require(normalizers / f"normalizer_holdout_fold_{fold}.json")

    repo = Path("/tmp/Quantum-ML-hybrid")
    command(["git", "init", str(repo)])
    command(
        ["git", "remote", "add", "origin", "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"],
        cwd=repo,
    )
    command(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=repo)
    command(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"Source mismatch: {actual} != {SOURCE_COMMIT}")
    command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_diffusion_ecg_encoder.py",
            "tests/test_train_reference_score_alignment.py",
        ],
        cwd=repo,
    )

    representation = Path("/kaggle/working/diffusion-encoder-representation-v1")
    command(
        [
            sys.executable,
            "run_diffusion_ecg_representation_export.py",
            "--hdf5", hdf5,
            "--metadata", metadata,
            "--normalizers", normalizers,
            "--output", representation,
            "--epochs", "8",
            "--batch-size", "128",
            "--lr", "0.0002",
            "--diffusion-steps", "1000",
            "--ema-decay", "0.999",
            "--min-snr-gamma", "5.0",
            "--seed", "20260927",
        ],
        cwd=repo,
    )
    command(
        [
            sys.executable,
            "run_train_reference_score_alignment.py",
            "--representations", representation,
            "--metadata", metadata,
            "--features", features,
            "--manifest", "configs/approved_feature_manifest_v0_4.json",
            "--output", "/kaggle/working/diffusion-encoder-q4-screen-v1",
            "--seeds", "42",
            "--per-class", "2000",
            "--epochs", "60",
            "--batch-size", "128",
            "--restarts", "3",
            "--bootstrap-iterations", "2000",
            "--device", "cuda",
        ],
        cwd=repo,
    )
    print("Diffusion-pretrained representation and retained VQC screen complete", flush=True)


if __name__ == "__main__":
    main()

