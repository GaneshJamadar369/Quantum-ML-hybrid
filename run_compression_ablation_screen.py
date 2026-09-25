"""Compression Ablation Screen: PLS vs NCA vs SupConAE vs Full Reuploading.

Tests four compression strategies on the h128 -> VQC pipeline:
  A. PLS-4  (current baseline)
  B. NCA-4  (neighborhood components analysis – task-aware linear)
  C. SupConAE-4  (supervised contrastive autoencoder – task-aware nonlinear)
  D. Full-128 data re-uploading (32 upload blocks x 4 qubits, no compression)

Each strategy is compared against an identical-input classical MLP control.
Folds 9 and 10 are sealed.
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from scipy.special import logit as scipy_logit


def _seed_all(s):
    import random; random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _safe_logit(p, eps=1e-7):
    return scipy_logit(np.clip(p, eps, 1 - eps))

def _auprc(y, s):
    m = ~np.isnan(s)
    return average_precision_score(y[m], s[m])

def _auroc(y, s):
    m = ~np.isnan(s)
    return roc_auc_score(y[m], s[m])


# ── Compression strategies ──────────────────────────────────────────────────

def _angle_scale(z_tr, z_va, seed):
    qt = QuantileTransformer(n_quantiles=min(256, len(z_tr)),
                             output_distribution="uniform", random_state=seed)
    z_tr = (2 * qt.fit_transform(z_tr) - 1) * (np.pi / 2)
    z_va = (2 * qt.transform(z_va) - 1) * (np.pi / 2)
    return z_tr.astype(np.float32), z_va.astype(np.float32)


def compress_pls(tr_h, tr_y, va_h, n=4, seed=42):
    sc = RobustScaler()
    x_tr = sc.fit_transform(tr_h); x_va = sc.transform(va_h)
    pls = PLSRegression(n_components=n, scale=False, max_iter=1000)
    q_tr = pls.fit_transform(x_tr, tr_y)[0]; q_va = pls.transform(x_va)
    for k in range(n):
        if np.corrcoef(q_tr[:, k], tr_y)[0, 1] < 0:
            q_tr[:, k] *= -1; q_va[:, k] *= -1
    return _angle_scale(q_tr, q_va, seed)


def compress_nca(tr_h, tr_y, va_h, n=4, seed=42):
    pipe = Pipeline([("sc", RobustScaler()),
                     ("nca", NeighborhoodComponentsAnalysis(
                         n_components=n, random_state=seed, max_iter=200))])
    q_tr = pipe.fit_transform(tr_h, tr_y); q_va = pipe.transform(va_h)
    return _angle_scale(q_tr, q_va, seed)


class _SupConAE(nn.Module):
    def __init__(self, in_d=128, lat=4):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_d,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,lat))
        self.dec = nn.Sequential(nn.Linear(lat,32),nn.GELU(),nn.Linear(32,64),nn.GELU(),nn.Linear(64,in_d))
        self.clf = nn.Linear(lat, 1)
    def forward(self, x):
        z = self.enc(x); return z, self.dec(z), self.clf(z).squeeze(-1)


def compress_supconae(tr_h, tr_y, va_h, lat=4, seed=42, epochs=60, bs=256,
                      device="cuda", lc=1.0, lr_=0.3, ls=0.3):
    _seed_all(seed)
    sc = RobustScaler()
    xt = torch.tensor(sc.fit_transform(tr_h), dtype=torch.float32).to(device)
    xv = torch.tensor(sc.transform(va_h), dtype=torch.float32).to(device)
    yt = torch.tensor(tr_y, dtype=torch.float32).to(device)
    model = _SupConAE(tr_h.shape[1], lat).to(device)
    pw = torch.tensor([(tr_y==0).sum()/max((tr_y==1).sum(),1)], dtype=torch.float32).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw); mse = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(tr_h))
    for _ in range(epochs):
        rng.shuffle(idx)
        model.train()
        for s in range(0, len(idx), bs):
            b = idx[s:s+bs]; opt.zero_grad()
            z, xhat, logit = model(xt[b])
            loss = lc*bce(logit, yt[b]) + lr_*mse(xhat, xt[b])
            if ls > 0 and len(b) > 2:
                z_n = nn.functional.normalize(z, dim=1)
                sim = torch.mm(z_n, z_n.T) / 0.1
                lb = yt[b].long()
                same = (lb.unsqueeze(0)==lb.unsqueeze(1)).float(); same.fill_diagonal_(0)
                exp = torch.exp(sim - sim.max(dim=1,keepdim=True).values)
                lp = sim - torch.log(exp.sum(dim=1,keepdim=True)+1e-9)
                loss = loss + ls*(-(same*lp).sum()/(same.sum()+1e-9))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        sch.step()
    model.eval()
    with torch.no_grad():
        z_tr = model.enc(xt).cpu().numpy()
        z_va = model.enc(xv).cpu().numpy()
    return _angle_scale(z_tr, z_va, seed)


# ── VQC models ──────────────────────────────────────────────────────────────

class _VQC4(nn.Module):
    def __init__(self, n_layers=2):
        super().__init__()
        import pennylane as qml
        self.n_qubits = 4; self.dev = qml.device("lightning.qubit", wires=4)
        self.weights = nn.Parameter(torch.randn(n_layers, 4, 3)*0.1)
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circ(x, w):
            for q in range(4): qml.RY(x[q], wires=q)
            for l in range(w.shape[0]):
                for q in range(4): qml.Rot(w[l,q,0],w[l,q,1],w[l,q,2],wires=q)
                for q in range(4): qml.CNOT(wires=[q,(q+1)%4])
            return qml.expval(qml.PauliZ(0))
        self._c = circ
    def forward(self, x): return torch.stack([self._c(xi, self.weights) for xi in x])


class _ReuploadVQC(nn.Module):
    def __init__(self, input_dim=128, block_size=4, n_lay=1):
        super().__init__()
        import pennylane as qml
        self.n_qubits = block_size
        self.n_blocks = input_dim // block_size
        self.dev = qml.device("lightning.qubit", wires=block_size)
        self.weights = nn.Parameter(torch.randn(self.n_blocks, n_lay, block_size, 3)*0.1)
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circ(x, w):
            for blk in range(self.n_blocks):
                chunk = x[blk*self.n_qubits:(blk+1)*self.n_qubits]
                for q in range(self.n_qubits): qml.RY(chunk[q], wires=q)
                for l in range(w.shape[1]):
                    for q in range(self.n_qubits): qml.Rot(w[blk,l,q,0],w[blk,l,q,1],w[blk,l,q,2],wires=q)
                    for q in range(self.n_qubits): qml.CNOT(wires=[q,(q+1)%self.n_qubits])
            return qml.expval(qml.PauliZ(0))
        self._c = circ
    def forward(self, x): return torch.stack([self._c(xi, self.weights) for xi in x])


# ── Training ─────────────────────────────────────────────────────────────────

def _train_vqc(model, x_tr, y_tr, y_soft, x_va, epochs, bs, seed, device, alpha=0.5):
    _seed_all(seed); model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
    pw = torch.tensor([(y_tr==0).sum()/max((y_tr==1).sum(),1)],dtype=torch.float32).to(device)
    bce_h = nn.BCEWithLogitsLoss(pos_weight=pw); bce_s = nn.BCEWithLogitsLoss()
    xt = torch.tensor(x_tr,dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr,dtype=torch.float32).to(device)
    ys = torch.tensor(y_soft,dtype=torch.float32).to(device)
    xv = torch.tensor(x_va,dtype=torch.float32).to(device)
    idx = np.arange(len(x_tr)); rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(idx); model.train()
        for s in range(0, len(idx), bs):
            b = idx[s:s+bs]; opt.zero_grad()
            lo = model(xt[b])
            loss = (1-alpha)*bce_h(lo,yt[b]) + alpha*bce_s(lo,ys[b])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
    model.eval()
    with torch.no_grad(): return model(xv).cpu().numpy()


def _train_mlp(x_tr, y_tr, x_va, seed, device):
    _seed_all(seed); in_d = x_tr.shape[1]
    mlp = nn.Sequential(nn.Linear(in_d,16),nn.GELU(),nn.Linear(16,1)).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-2)
    pw = torch.tensor([(y_tr==0).sum()/max((y_tr==1).sum(),1)],dtype=torch.float32).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    xt = torch.tensor(x_tr,dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr,dtype=torch.float32).to(device)
    xv = torch.tensor(x_va,dtype=torch.float32).to(device)
    for _ in range(40):
        mlp.train(); opt.zero_grad()
        bce(mlp(xt).squeeze(-1),yt).backward(); opt.step()
    mlp.eval()
    with torch.no_grad(): return mlp(xv).squeeze(-1).cpu().numpy()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--representations", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-class", type=int, default=400)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  CUDA: {torch.cuda.is_available()}", flush=True)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    meta = pd.read_csv(args.metadata)
    meta = meta[~meta["fold"].isin([9,10])].copy()
    record_ids = meta["ecg_id"].to_numpy(int)
    labels = meta["mi_label"].to_numpy(int)
    folds  = meta["fold"].to_numpy(int)
    patient_ids = meta["patient_id"].to_numpy(int)
    print(f"Records={len(record_ids)}  MI+={labels.sum()}  Folds={sorted(set(folds.tolist()))}", flush=True)

    # Load fold reps
    rep_dir = Path(args.representations)
    r2l = {int(r): int(l) for r,l in zip(record_ids, labels)}
    fold_data = {}
    for npz_path in sorted(rep_dir.glob("outer_fold_*_representations.npz")):
        fid = int(npz_path.stem.split("_")[2])
        if fid in (9,10): continue
        d = np.load(npz_path)
        val_ids = d["val_indices"].astype(int)
        val_h   = d["val_embeddings"].astype(np.float32)
        val_y   = np.array([r2l[r] for r in val_ids], dtype=int)
        is_val  = np.isin(record_ids, val_ids)
        # Build train embeddings from other folds
        train_hs = []
        for p2 in sorted(rep_dir.glob("outer_fold_*_representations.npz")):
            fid2 = int(p2.stem.split("_")[2])
            if fid2 == fid or fid2 in (9,10): continue
            train_hs.append(np.load(p2)["val_embeddings"].astype(np.float32))
        train_h = np.vstack(train_hs)
        train_y = labels[~is_val]
        fold_data[fid] = dict(train_h=train_h,train_y=train_y,
                               val_h=val_h,val_y=val_y,val_ids=val_ids)

    n = len(labels)
    ks = ["pls_vqc","pls_mlp","nca_vqc","nca_mlp","ae_vqc","ae_mlp","reup_vqc","reup_mlp"]
    scores = {k: np.full(n, np.nan) for k in ks}

    for fid in sorted(fold_data):
        rep = fold_data[fid]
        tr_h, tr_y, va_h = rep["train_h"], rep["train_y"], rep["val_h"]
        val_ids = rep["val_ids"]
        val_mask = np.isin(record_ids, val_ids)
        val_idx  = np.where(val_mask)[0]

        print(f"\n{'='*60}", flush=True)
        print(f"Fold {fid}  train={len(tr_h)}  val={len(va_h)}", flush=True)

        # Patient-balanced subsample
        train_pidx = np.where(~val_mask)[0]
        pmap = defaultdict(list)
        for i,idx in enumerate(train_pidx):
            pmap[patient_ids[idx]].append(i)
        rng = np.random.default_rng(args.seed + fid)
        spos=[]; sneg=[]
        for pid,idxs in pmap.items():
            pp=[i for i in idxs if tr_y[i]==1]
            pn=[i for i in idxs if tr_y[i]==0]
            if pp: spos.append(rng.choice(pp))
            if pn: sneg.append(rng.choice(pn))
        pc = min(args.per_class, len(spos), len(sneg))
        rng.shuffle(spos); rng.shuffle(sneg)
        sidx = np.array(spos[:pc]+sneg[:pc])
        q_tr_h = tr_h[sidx]; q_tr_y = tr_y[sidx]

        # Classical teacher soft labels
        clf = HistGradientBoostingClassifier(max_iter=300, class_weight="balanced", random_state=42)
        clf.fit(tr_h, tr_y)
        y_soft = clf.predict_proba(q_tr_h)[:,1].astype(np.float32)
        sf = args.seed + fid*100

        # A. PLS
        t0=time.time()
        ztr,zva = compress_pls(tr_h,tr_y,va_h,seed=sf)
        zq=ztr[sidx]
        scores["pls_vqc"][val_idx] = _train_vqc(_VQC4(2),zq,q_tr_y,y_soft,zva,args.epochs,args.batch_size,sf,device)
        scores["pls_mlp"][val_idx] = _train_mlp(zq,q_tr_y,zva,sf+1,device)
        print(f"  PLS   {time.time()-t0:.0f}s  VQC={_auprc(rep['val_y'],scores['pls_vqc'][val_idx]):.5f}  MLP={_auprc(rep['val_y'],scores['pls_mlp'][val_idx]):.5f}", flush=True)

        # B. NCA
        t0=time.time()
        ztr,zva = compress_nca(tr_h,tr_y,va_h,seed=sf)
        zq=ztr[sidx]
        scores["nca_vqc"][val_idx] = _train_vqc(_VQC4(2),zq,q_tr_y,y_soft,zva,args.epochs,args.batch_size,sf+10,device)
        scores["nca_mlp"][val_idx] = _train_mlp(zq,q_tr_y,zva,sf+11,device)
        print(f"  NCA   {time.time()-t0:.0f}s  VQC={_auprc(rep['val_y'],scores['nca_vqc'][val_idx]):.5f}  MLP={_auprc(rep['val_y'],scores['nca_mlp'][val_idx]):.5f}", flush=True)

        # C. SupConAE
        t0=time.time()
        ztr,zva = compress_supconae(tr_h,tr_y,va_h,seed=sf,device=device)
        zq=ztr[sidx]
        scores["ae_vqc"][val_idx] = _train_vqc(_VQC4(2),zq,q_tr_y,y_soft,zva,args.epochs,args.batch_size,sf+20,device)
        scores["ae_mlp"][val_idx] = _train_mlp(zq,q_tr_y,zva,sf+21,device)
        print(f"  AE    {time.time()-t0:.0f}s  VQC={_auprc(rep['val_y'],scores['ae_vqc'][val_idx]):.5f}  MLP={_auprc(rep['val_y'],scores['ae_mlp'][val_idx]):.5f}", flush=True)

        # D. Full 128-D reuploading
        t0=time.time()
        sc2 = RobustScaler()
        x128_tr = sc2.fit_transform(tr_h); x128_va = sc2.transform(va_h)
        qt128 = QuantileTransformer(n_quantiles=min(256,len(x128_tr)),output_distribution="uniform",random_state=sf)
        x128_tr = (2*qt128.fit_transform(x128_tr)-1)*(np.pi/2)
        x128_va = (2*qt128.transform(x128_va)-1)*(np.pi/2)
        zq128 = x128_tr[sidx].astype(np.float32)
        scores["reup_vqc"][val_idx] = _train_vqc(_ReuploadVQC(128,4,1),zq128,q_tr_y,y_soft,x128_va.astype(np.float32),args.epochs,args.batch_size,sf+30,device)
        scores["reup_mlp"][val_idx] = _train_mlp(zq128,q_tr_y,x128_va.astype(np.float32),sf+31,device)
        print(f"  REUP  {time.time()-t0:.0f}s  VQC={_auprc(rep['val_y'],scores['reup_vqc'][val_idx]):.5f}  MLP={_auprc(rep['val_y'],scores['reup_mlp'][val_idx]):.5f}", flush=True)

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n"+"="*70, flush=True)
    print("FINAL COMPRESSION ABLATION SUMMARY", flush=True)
    print("="*70, flush=True)
    results = {}
    for k in ks:
        ap = _auprc(labels, scores[k]); ar = _auroc(labels, scores[k])
        print(f"  {k:20s}  AUPRC={ap:.5f}  AUROC={ar:.5f}", flush=True)
        results[k] = {"auprc": ap, "auroc": ar}
    print("\n-- VQC vs MLP delta (same compression input) --", flush=True)
    for pref in ["pls","nca","ae","reup"]:
        d = results[f"{pref}_vqc"]["auprc"] - results[f"{pref}_mlp"]["auprc"]
        print(f"  {pref.upper():6s}  ΔAUPRC(VQC-MLP) = {d:+.5f}", flush=True)
    (out/"results.json").write_text(json.dumps(results,indent=2))
    print("Done.", flush=True)

if __name__=="__main__":
    main()
