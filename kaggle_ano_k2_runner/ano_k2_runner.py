"""Pinned Kaggle GPU runner for the two-local ANO q4 scout."""

from pathlib import Path
import subprocess
import sys


SOURCE_COMMIT = "6f0aed03ebc9de42c7e41c89825c0dffc9742b7a"
LOCALITY = "2"


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
        raise RuntimeError("The ANO q4 scout requires a Kaggle GPU")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    source = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    representations = locate(
        source / "aquire-med-compact-ecg-transformer-representation/transformer-representation-v1",
        "outer_fold_1_representations.npz",
        directory=True,
    )
    metadata = locate(
        source / "aquire-med-preprocessing-pipeline/aquire-artifacts/processing_metadata_development.csv",
        "processing_metadata_development.csv",
    )
    features = locate(
        source / "aquire-med-full-feature-repair/full-feature-repair/feature-evidence-v0-4/deployable_features_with_clinical_composites.csv",
        "deployable_features_with_clinical_composites.csv",
    )
    reference = locate(
        source / "aquire-med-q4-score-alignment-screen/q4-score-alignment-screen-v1/seed_42/oof_predictions_and_observables.csv",
        "oof_predictions_and_observables.csv",
    )
    repo = Path("/tmp/Quantum-ML-hybrid")
    command(["git", "init", str(repo)])
    command(
        ["git", "remote", "add", "origin", "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"],
        cwd=repo,
    )
    command(["git", "fetch", "--depth", "1", "origin", SOURCE_COMMIT], cwd=repo)
    command(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if actual != SOURCE_COMMIT:
        raise RuntimeError(f"Source mismatch: {actual} != {SOURCE_COMMIT}")
    command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_adaptive_observables.py",
            "tests/test_adaptive_observable_q4_screen.py",
        ],
        cwd=repo,
    )
    base = [
        sys.executable,
        "run_adaptive_observable_q4_screen.py",
        "--locality",
        LOCALITY,
        "--representations",
        representations,
        "--metadata",
        metadata,
        "--features",
        features,
        "--manifest",
        "configs/approved_feature_manifest_v0_4.json",
        "--reference",
        reference,
        "--output",
        "/kaggle/working/ano-k2-q4-scout-v1",
        "--seed",
        "42",
        "--per-class",
        "2000",
        "--epochs",
        "40",
        "--batch-size",
        "128",
        "--restarts",
        "1",
        "--bootstrap-iterations",
        "2000",
        "--device",
        "cuda",
    ]
    command(base + ["--preflight-only"], cwd=repo)
    command(base, cwd=repo)


if __name__ == "__main__":
    main()
