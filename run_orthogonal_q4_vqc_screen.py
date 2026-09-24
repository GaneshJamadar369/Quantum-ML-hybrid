"""Identity-initialized orthogonal q4 mixer with ring or ladder VQC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from aquire_preprocessing.models_advanced_fusion import OrthogonalQ4Mixer
from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier
from run_advanced_quantum_fusion_screen import _fit_q4, _seed
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _patient_unique_sample


def _train(
    train_q, train_y, train_hard, val_q, *, topology, layers, use_mixer,
    epochs, batch_size, seed, device,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = OrthogonalQ4Mixer()
            if not use_mixer:
                self.mixer.skew_parameters.requires_grad_(False)
            self.vqc = TorchStatevectorQuantumClassifier(4, n_layers=layers, topology=topology)

        def forward(self, x):
            angles = self.mixer(x)
            return self.vqc(angles), angles

    _seed(seed)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    weights = np.ones(len(train_y), dtype=np.float32)
    weights[(train_y == 0) & train_hard] = 1.25
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_q), torch.from_numpy(train_y.astype(np.float32)),
            torch.from_numpy(weights),
        ), batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    losses=[]
    for _ in range(epochs):
        model.train(); total=0.0; seen=0
        for q, target, weight in loader:
            q,target,weight=q.to(device),target.to(device),weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits,_=model(q)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).mean()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
            total += float(loss.detach())*len(q); seen += len(q)
        scheduler.step(); losses.append(total/max(seen,1))

    def infer(q):
        model.eval(); scores=[]; angles=[]
        with torch.inference_mode():
            for start in range(0,len(q),batch_size):
                score,mixed=model(torch.from_numpy(q[start:start+batch_size]).to(device))
                scores.append(score.cpu().numpy());angles.append(mixed.cpu().numpy())
        return np.concatenate(scores),np.concatenate(angles)
    train_result=infer(train_q);val_result=infer(val_q)
    matrix=model.mixer.orthogonal_matrix().detach().cpu().numpy()
    return train_result,val_result,{
        'initial_loss':losses[0],'final_loss':losses[-1],
        'orthogonality_error':float(np.max(np.abs(matrix.T@matrix-np.eye(4)))),
        'mixer_parameters':model.mixer.skew_parameters.detach().cpu().numpy().tolist(),
        'trainable_parameters':int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def _metrics(y,logits):
    p=expit(logits)
    return {'auprc':float(average_precision_score(y,p)),'auroc':float(roc_auc_score(y,p)),'brier':float(brier_score_loss(y,p))}


def run(args):
    import torch
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metadata=pd.read_csv(args.metadata).set_index('ecg_id')
    labels_by_id=metadata.mi_label
    hard_by_id=metadata.hard_negative
    predictions={k:[] for k in ('ecg_id','patient_id','fold','label','mixed_vqc','base_vqc','same_q_logistic','same_q_mlp','same_q_rbf')}
    audits=[]
    for fold in range(1,9):
        d=np.load(args.representations/f'outer_fold_{fold}_representations.npz',allow_pickle=False)
        train_ids=d['train_record_ids'].astype(int);val_ids=d['val_record_ids'].astype(int)
        train_y=d['train_labels'].astype(int);val_y=d['val_labels'].astype(int)
        train_patients=d['train_patient_ids'].astype(int);val_patients=d['val_patient_ids'].astype(int)
        if set(train_patients)&set(val_patients):raise RuntimeError('patient leakage')
        if not np.array_equal(labels_by_id.loc[train_ids].to_numpy(int),train_y):raise RuntimeError('label mismatch')
        base_train,base_val,_=_fit_q4(d['train_embeddings'].astype('f4'),train_y,d['val_embeddings'].astype('f4'),args.seed+fold)
        hard=hard_by_id.loc[train_ids].to_numpy(bool)
        pick=_patient_unique_sample(train_y,hard,train_patients,per_class=args.per_class,seed=args.seed+fold)
        q=base_train[pick];y=train_y[pick];h=hard[pick]
        train_mixed,val_mixed,audit=_train(q,y,h,base_val,topology=args.topology,layers=args.layers,use_mixer=True,epochs=args.epochs,batch_size=args.batch_size,seed=args.seed+100*fold,device=device)
        _,val_base,_=_train(q,y,h,base_val,topology=args.topology,layers=args.layers,use_mixer=False,epochs=args.epochs,batch_size=args.batch_size,seed=args.seed+100*fold,device=device)
        q_train=train_mixed[1];q_val=val_mixed[1]
        lr=LogisticRegression(C=1,class_weight='balanced',max_iter=2000).fit(q_train,y)
        lr_score=lr.decision_function(q_val)
        mlp=MLPClassifier(hidden_layer_sizes=(7,),max_iter=500,random_state=args.seed+fold).fit(q_train,y)
        mp=mlp.predict_proba(q_val)[:,1];mlp_score=np.log(np.clip(mp,1e-5,1-1e-5)/np.clip(1-mp,1e-5,1-1e-5))
        rbf=SVC(C=1,kernel='rbf',class_weight='balanced').fit(q_train,y)
        values={'ecg_id':val_ids,'patient_id':val_patients,'fold':np.full(len(val_y),fold),'label':val_y,'mixed_vqc':val_mixed[0],'base_vqc':val_base[0],'same_q_logistic':lr_score,'same_q_mlp':mlp_score,'same_q_rbf':rbf.decision_function(q_val)}
        for key,value in values.items():predictions[key].append(value)
        audit['fold']=fold;audits.append(audit)
        print(f"fold {fold}: base={average_precision_score(val_y,expit(val_base[0])):.4f} mixed={average_precision_score(val_y,expit(val_mixed[0])):.4f}",flush=True)
    arrays={k:np.concatenate(v) for k,v in predictions.items()}; y=arrays['label'];patients=arrays['patient_id']
    metrics={name:_metrics(y,arrays[name]) for name in ('mixed_vqc','base_vqc','same_q_logistic','same_q_mlp','same_q_rbf')}
    comparisons={}
    for control in ('base_vqc','same_q_logistic','same_q_mlp','same_q_rbf'):
        comparisons[f'mixed_vqc_minus_{control}']=_paired_patient_bootstrap(y,patients,expit(arrays['mixed_vqc']),expit(arrays[control]),args.bootstrap_iterations,args.seed,f'mixed_vqc_minus_{control}')
    best=max(('same_q_logistic','same_q_mlp','same_q_rbf'),key=lambda x:metrics[x]['auprc']);delta=metrics['mixed_vqc']['auprc']-metrics[best]['auprc'];ci=comparisons[f'mixed_vqc_minus_{best}']['delta_auprc']
    verdict={'topology':args.topology,'layers':args.layers,'improves_base':metrics['mixed_vqc']['auprc']>=metrics['base_vqc']['auprc']+0.005,'quantum_gate':bool(delta>=0.005 and ci['ci95_low']>0),'best_same_q_control':best,'delta_auprc':delta,'fold_9_accessed':False,'fold_10_accessed':False}
    pd.DataFrame({k:v if k in ('ecg_id','patient_id','fold','label') else expit(v) for k,v in arrays.items()}).to_csv(args.output/'oof_predictions.csv',index=False)
    for name,obj in [('metrics.json',metrics),('bootstrap.json',comparisons),('audits.json',audits),('verdict.json',verdict)]: (args.output/name).write_text(json.dumps(obj,indent=2))
    print(json.dumps({'metrics':metrics,'verdict':verdict},indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--representations',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--topology',choices=['ring','ladder'],required=True);p.add_argument('--layers',type=int,default=3);p.add_argument('--per-class',type=int,default=2000);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--bootstrap-iterations',type=int,default=2000);p.add_argument('--seed',type=int,default=20260924);run(p.parse_args())


if __name__=='__main__':main()
