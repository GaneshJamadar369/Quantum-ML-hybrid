"""Run the v0.4 signed-R measurement gate on unseen patients."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


CODE_COMMIT = "de6880fce7cd9ef6426dd4303e263674d5de5687"
REPOSITORY = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("\n>>> " + " ".join(command), flush=True)
    started = time.time()
    process = subprocess.Popen(
        command, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    print(f"[exit={process.returncode}, seconds={time.time() - started:.1f}]", flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


def locate(root: Path, filename: str, preferred_token: str | None = None) -> Path:
    matches = list(root.glob(f"**/{filename}"))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} below {root}")
    preferred = [path for path in matches if preferred_token and preferred_token in str(path)]
    selected = preferred[0] if preferred else matches[0]
    print(f"{filename}: {selected}", flush=True)
    return selected


def main() -> None:
    input_root = Path("/kaggle/input")
    output = Path("/kaggle/working/amplitude-holdout")
    output.mkdir(parents=True, exist_ok=True)
    primary = locate(input_root, "primary_development_100hz.h5", "preprocessing-pipeline")
    reference = locate(input_root, "ecgdeli_features.csv")
    prior_sample = locate(
        input_root, "deployable_features_v0_3_calibration.csv", "measurement-calibration"
    )

    repo = Path("/tmp/Quantum-ML-hybrid")
    run(["git", "clone", REPOSITORY, str(repo)])
    run(["git", "checkout", CODE_COMMIT], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])
    run([sys.executable, "-m", "pip", "install", "-q", "neurokit2==0.2.10"])

    local = output / "deployable_features_v0_4_holdout.csv"
    run([
        sys.executable, "run_feature_reextraction.py",
        "--hdf5", str(primary),
        "--output", str(local),
        "--sample-size", "1024",
        "--exclude-ecg-ids", str(prior_sample),
        "--shard-size", "128",
        "--seed", "20260918",
    ], cwd=repo)
    run([
        sys.executable, "run_measurement_calibration.py",
        "--local-features", str(local),
        "--reference-features", str(reference),
        "--feature-pairs", "configs/deployable_reference_pairs_v0_4.json",
        "--thresholds", "configs/measurement_acceptance_v0_4.json",
        "--output", str(output),
    ], cwd=repo)

    files = sorted(
        ({"path": str(path.relative_to(output)), "bytes": path.stat().st_size}
         for path in output.rglob("*") if path.is_file()),
        key=lambda item: item["path"],
    )
    (output / "artifact_manifest.json").write_text(json.dumps(files, indent=2))
    print("\nv0.4 unseen-patient artifacts:", flush=True)
    for item in files:
        print(f"  {item['path']} ({item['bytes'] / 1024:.1f} KiB)", flush=True)


if __name__ == "__main__":
    main()
