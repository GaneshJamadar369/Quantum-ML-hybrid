"""Pinned Kaggle GPU entrypoint for the G6Q-KD2 divergence screen."""

from pathlib import Path
import subprocess
import sys


SOURCE_COMMIT = "9d128ad185fbf8779ed730612e3ac551b8afb894"


def run(command, cwd=None):
    print(">>>", " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], cwd=cwd, check=True)


def locate(preferred: Path, marker: str, *, directory: bool = False) -> Path:
    if preferred.exists():
        return preferred
    matches = sorted(Path("/kaggle/input").rglob(marker))
    print(f"Preferred input missing: {preferred}; matches={matches}", flush=True)
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {marker}, found {len(matches)}")
    return matches[0].parent if directory else matches[0]


def main():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("G6Q-KD2 requires a Kaggle GPU")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    source = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    representations = locate(
        source
        / "aquire-med-compact-ecg-transformer-representation"
        / "transformer-representation-v1",
        "outer_fold_1_representations.npz",
        directory=True,
    )
    metadata = locate(
        source
        / "aquire-med-preprocessing-pipeline"
        / "aquire-artifacts"
        / "processing_metadata_development.csv",
        "processing_metadata_development.csv",
    )
    features = locate(
        source
        / "aquire-med-full-feature-repair"
        / "full-feature-repair"
        / "feature-evidence-v0-4"
        / "deployable_features_with_clinical_composites.csv",
        "deployable_features_with_clinical_composites.csv",
    )

    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "init", repo])
    run(["git", "-C", repo, "remote", "add", "origin",
         "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"])
    run(["git", "-C", repo, "fetch", "--depth", "1", "origin", SOURCE_COMMIT])
    run(["git", "-C", repo, "checkout", "--detach", "FETCH_HEAD"])
    executed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if executed != SOURCE_COMMIT:
        raise RuntimeError(f"Source mismatch: expected {SOURCE_COMMIT}, got {executed}")
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", repo])
    run([
        sys.executable, "-m", "pytest", "-q",
        "tests/test_distillation.py",
        "tests/test_divergence_distillation_screen.py",
    ], cwd=repo)

    output = Path("/kaggle/working/divergence-distillation-q4-v1")
    command = [
        sys.executable, "run_divergence_distillation_screen.py",
        "--representations", representations,
        "--metadata", metadata,
        "--features", features,
        "--manifest", "configs/approved_feature_manifest_v0_4.json",
        "--output", output,
        "--per-class", "2000",
        "--epochs", "30",
        "--batch-size", "128",
        "--bootstrap-iterations", "2000",
        "--seed", "20260925",
    ]
    run(command + ["--preflight-only"], cwd=repo)
    run(command, cwd=repo)
    (output / "source_commit.txt").write_text(SOURCE_COMMIT + "\n")
    print("G6Q-KD2 divergence screen complete", flush=True)


if __name__ == "__main__":
    main()
