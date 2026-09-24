from pathlib import Path
import subprocess,sys
REPO="https://github.com/GaneshJamadar369/Quantum-ML-hybrid.git";REVISION="1207946819ec9efc4385c2a3bb06e2990f343601";TOPOLOGY="ring"
def run(c,cwd=None): print(">>>"," ".join(map(str,c)),flush=True);subprocess.run(list(map(str,c)),cwd=cwd,check=True)
def locate(root,name):
 m=list(root.rglob(name))
 if not m: raise FileNotFoundError(name)
 return min(m,key=lambda p:(len(p.parts),str(p)))
def main():
 root=Path('/kaggle/input');metadata=locate(root,'processing_metadata_development.csv');representations=locate(root,'outer_fold_1_representations.npz').parent;repo=Path('/tmp/Quantum-ML-hybrid')
 run(['git','clone','--filter=blob:none',REPO,repo]);run(['git','fetch','origin',REVISION,'--depth','1'],repo);run(['git','checkout','--detach',REVISION],repo)
 if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=REVISION:raise RuntimeError('revision mismatch')
 run([sys.executable,'-m','pip','install','-q','--no-deps','-e',repo]);run([sys.executable,'-m','pytest','-q','tests/test_advanced_fusion.py','tests/test_supervised_qubit_scaling.py'],repo)
 run([sys.executable,'run_orthogonal_q4_vqc_screen.py','--representations',representations,'--metadata',metadata,'--output',f'/kaggle/working/orthogonal-q4-{TOPOLOGY}-l3','--topology',TOPOLOGY,'--layers','3','--per-class','2000','--epochs','40','--batch-size','128','--bootstrap-iterations','2000','--seed','20260924'],repo)
if __name__=='__main__':main()
