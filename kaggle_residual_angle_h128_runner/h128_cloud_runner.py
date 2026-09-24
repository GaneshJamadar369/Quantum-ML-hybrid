"""Pinned Kaggle runner for the h128 residual-angle adapter."""
from pathlib import Path
import subprocess
import sys

REPO = "https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git"
REVISION = "f6916245c14e364444610dd576cb34a151caf373"
MODE = "h128"


def run(command, cwd=None):
    print(">>>", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def locate(root, name):
    matches=list(root.rglob(name))
    if not matches: raise FileNotFoundError(f"{name} under {root}")
    return min(matches,key=lambda p:(len(p.parts),str(p)))


def main():
    root=Path('/kaggle/input')
    metadata=locate(root,'processing_metadata_development.csv')
    features=locate(root,'deployable_features_with_clinical_composites.csv')
    representations=locate(root,'outer_fold_1_representations.npz').parent
    repo=Path('/tmp/Quantum-ML-hybrid')
    run(['git','clone','--filter=blob:none',REPO,repo])
    run(['git','fetch','origin',REVISION,'--depth','1'],cwd=repo)
    run(['git','checkout','--detach',REVISION],cwd=repo)
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if actual != REVISION: raise RuntimeError(f"revision {actual} != {REVISION}")
    run([sys.executable,'-m','pip','install','-q','--no-deps','-e',repo])
    run([sys.executable,'-m','pytest','-q','tests/test_advanced_fusion.py','tests/test_supervised_qubit_scaling.py'],cwd=repo)
    run([
      sys.executable,'run_residual_angle_adapter_screen.py',
      '--representations',representations,'--metadata',metadata,'--features',features,
      '--manifest',repo/'configs/approved_feature_manifest_v0_4.json',
      '--output',f'/kaggle/working/residual-angle-{MODE}-q4','--mode',MODE,
      '--per-class','2000','--epochs','30','--batch-size','128',
      '--bootstrap-iterations','2000','--seed','20260924'
    ],cwd=repo)
    print(f'Completed residual angle adapter {MODE}',flush=True)


if __name__ == '__main__': main()
