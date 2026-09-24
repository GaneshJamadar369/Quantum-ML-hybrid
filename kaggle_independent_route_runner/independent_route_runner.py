"""Private Kaggle GPU runner for independent dual-route fusion research."""

from pathlib import Path
import subprocess
import sys


EXPECTED_SOURCE_COMMIT = "76774c3fb160c8f6849baef454f66f9554f7bd54"


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
    run(["git", "fetch", "--depth", "1", "origin", EXPECTED_SOURCE_COMMIT], cwd=repo)
    run(["git", "checkout", "--detach", EXPECTED_SOURCE_COMMIT], cwd=repo)
    # The independent screen uses the repository's exact PyTorch statevector
    # simulator and Kaggle's existing NumPy/SciPy/sklearn/PyTorch stack.  Do
    # not mutate the environment with unrelated preprocessing dependencies.
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if actual_commit != EXPECTED_SOURCE_COMMIT:
        raise RuntimeError(f"Source commit mismatch: {actual_commit}")
    print("Pinned source commit:", actual_commit, flush=True)
    run([sys.executable, "-c",
         "import numpy,scipy,sklearn,torch; "
         "print(numpy.__version__,scipy.__version__,sklearn.__version__,torch.__version__)"],
        cwd=repo)

    # Verify route config
    route_config = repo / "configs" / "independent_route_v1.json"
    print(f"input: {route_config}", flush=True)
    if not route_config.exists():
        raise FileNotFoundError(route_config)

    command = [
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
    ]
    # Fail in under a minute if an upstream archive changes schema, record
    # ordering, labels, patients, or fold coverage.
    run(command + ["--preflight-only"], cwd=repo)
    run(command, cwd=repo)
    print("Independent dual-route fusion screen complete; folds 9 and 10 sealed", flush=True)


if __name__ == "__main__":
    main()
