"""Run G5F feature evidence using the completed preprocessing kernel output."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time


CODE_COMMIT = "1e8aef0b90559ff97fa9d9d49e93773959d3ffba"
REPOSITORY = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("\n>>> " + " ".join(command), flush=True)
    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
    process.wait()
    print(f"[exit={process.returncode}, seconds={time.time() - started:.1f}]", flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


def locate(root: Path, filename: str, required: bool = True) -> Path | None:
    matches = list(root.glob(f"**/{filename}"))
    if not matches:
        if required:
            raise FileNotFoundError(f"Could not find {filename} below {root}")
        return None
    preferred = [path for path in matches if "aquire-artifacts" in str(path)]
    selected = preferred[0] if preferred else matches[0]
    print(f"{filename}: {selected}", flush=True)
    return selected


def main() -> None:
    input_root = Path("/kaggle/input")
    output = Path("/kaggle/working/feature-evidence")
    output.mkdir(parents=True, exist_ok=True)

    features = locate(input_root, "deployable_features_development.csv")
    metadata = locate(input_root, "processing_metadata_development.csv")
    manifest = locate(input_root, "patient_manifest.csv")
    primary = locate(input_root, "primary_development_100hz.h5")
    reference = locate(input_root, "ecgdeli_features.csv", required=False)

    repo = Path("/kaggle/working/Quantum-ML-hybrid")
    run(["git", "clone", REPOSITORY, str(repo)])
    run(["git", "checkout", CODE_COMMIT], cwd=repo)
    run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)])

    command = [
        sys.executable,
        "run_feature_evidence.py",
        "--features", str(features),
        "--metadata", str(metadata),
        "--manifest", str(manifest),
        "--primary-hdf5", str(primary),
        "--output", str(output),
    ]
    if reference is not None:
        command.extend(["--reference-features", str(reference)])
    run(command, cwd=repo)

    files = sorted(
        (
            {"path": str(path.relative_to(output)), "bytes": path.stat().st_size}
            for path in output.rglob("*") if path.is_file()
        ),
        key=lambda item: item["path"],
    )
    (output / "artifact_manifest.json").write_text(json.dumps(files, indent=2))
    print("\nFeature evidence artifacts:", flush=True)
    for item in files:
        print(f"  {item['path']} ({item['bytes'] / 1024:.1f} KiB)", flush=True)


if __name__ == "__main__":
    main()
