"""Kaggle GPU runner for Compression Ablation Screen."""
from pathlib import Path
import subprocess, sys

def run(cmd, cwd=None):
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)

def locate(preferred, marker, *, directory=False):
    p = Path(preferred)
    if p.exists(): return p
    matches = sorted(Path("/kaggle/input").rglob(marker))
    if len(matches) != 1: raise FileNotFoundError(preferred)
    return matches[0].parent if directory else matches[0]

def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Needs GPU")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    src = Path("/kaggle/input/notebooks/swayamjeetbhagat4")
    rep = locate(src/"aquire-med-compact-ecg-transformer-representation/transformer-representation-v1","outer_fold_1_representations.npz",directory=True)
    meta= locate(src/"aquire-med-preprocessing-pipeline/aquire-artifacts/processing_metadata_development.csv","processing_metadata_development.csv")

    repo = Path("/tmp/Quantum-ML-hybrid")
    if not repo.exists():
        run(["git","clone","--depth","1","https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git",str(repo)])
    run([sys.executable,"-m","pip","install","pennylane","pennylane-lightning"])

    run([sys.executable,"run_compression_ablation_screen.py",
         "--representations", str(rep),
         "--metadata", str(meta),
         "--output","/kaggle/working/compression-ablation-v1",
         "--epochs","20","--batch-size","128","--seed","42","--per-class","400"],
        cwd=repo)
    print("Compression ablation complete.", flush=True)

if __name__=="__main__":
    main()
