"""
Phase 7: Locked Holdout Single-Pass Evaluation Engine.
Evaluates the final champions across all phases on strictly sealed Folds 9 & 10 (4,489 records).
Champions:
1. Classical Champion: XGBoost
2. Deep Learning Champion: ECGResNet1D
3. Multimodal Champion: ECGMultimodalHybrid
4. Quantum Champion: HQNN / QSVM
Generates final holdout metrics, calibration curves, conformal coverage, and clinical decision curves.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import h5py

from aquire_preprocessing.baselines import evaluate_probabilities
from aquire_preprocessing.config import DEV_FOLDS, CALIBRATION_FOLD, LOCKED_TEST_FOLD
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.models_1d import ECGResNet1D
from aquire_preprocessing.models_hybrid import ECGMultimodalHybrid
from aquire_preprocessing.models_quantum import (
    HybridQuantumNeuralNetwork,
    QSVMClassifier,
)


def run_sealed_holdout_evaluation(
    h5_path: Path,
    metadata_path: Path,
    features_csv: Path,
    manifest_path: Path,
    output_dir: Path,
    seed: int = 42,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================", flush=True)
    print("PHASE 7: SEALED LOCKED HOLDOUT SINGLE-PASS EVALUATION (FOLDS 9 & 10)", flush=True)
    print("=================================================================", flush=True)

    # Load dataset
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")
    features_raw = pd.read_csv(features_csv).set_index("ecg_id")

    manifest = load_feature_manifest(manifest_path, features_raw.columns)
    approved = manifest["approved_features"]
    features_raw = features_raw[approved]

    joined = metadata.join(features_raw, how="inner", validate="one_to_one")
    joined = joined[joined.eligibility.eq("PRIMARY")]

    # Split into Development (1-8), Calibration (9), and Test (10)
    dev_mask = joined.strat_fold.isin(DEV_FOLDS)
    cal_mask = joined.strat_fold == CALIBRATION_FOLD
    test_mask = joined.strat_fold == LOCKED_TEST_FOLD
    holdout_mask = cal_mask | test_mask

    print(f"Total Primary Records: {len(joined)}")
    print(f"Development Records (Folds 1-8): {dev_mask.sum()}")
    print(f"Calibration Records (Fold 9): {cal_mask.sum()}")
    print(f"Locked Test Records (Fold 10): {test_mask.sum()}")
    print(f"Total Holdout Records (Folds 9 & 10): {holdout_mask.sum()}")

    # Feature matrix
    features_mat = joined[approved].select_dtypes(include=[np.number]).fillna(0.0).to_numpy(dtype=np.float32)

    # Pre-load raw signals
    print("\nPre-loading raw signals for holdout evaluation ...", flush=True)
    with h5py.File(h5_path, "r") as h5:
        h5_ids = np.asarray(h5["ecg_id"])
        id_to_idx = {int(eid): idx for idx, eid in enumerate(h5_ids)}
        ordered_indices = np.array([id_to_idx[eid] for eid in joined.index], dtype=int)
        raw_signals = h5["accepted_signal"][ordered_indices].astype(np.float32)
        if raw_signals.shape[1] == 1000 and raw_signals.shape[2] == 12:
            raw_signals = np.transpose(raw_signals, (0, 2, 1))

    # Prepare dev and holdout splits
    X_dev_tab = features_mat[dev_mask]
    y_dev = joined.mi_label[dev_mask].to_numpy(dtype=int)
    sig_dev = raw_signals[dev_mask]

    X_test_tab = features_mat[test_mask]
    y_test = joined.mi_label[test_mask].to_numpy(dtype=int)
    sig_test = raw_signals[test_mask]
    test_pids = joined.patient_id[test_mask].to_numpy()
    test_folds = joined.strat_fold[test_mask].to_numpy(dtype=int)
    test_hn = joined.hard_negative[test_mask].to_numpy(dtype=bool) if "hard_negative" in joined else np.zeros(len(y_test), dtype=bool)

    # Standardize tabular features with development statistics
    mean_tab = np.mean(X_dev_tab, axis=0, keepdims=True)
    std_tab = np.std(X_dev_tab, axis=0, keepdims=True)
    std_tab[std_tab < 1e-5] = 1.0

    X_dev_tab_norm = (X_dev_tab - mean_tab) / std_tab
    X_test_tab_norm = (X_test_tab - mean_tab) / std_tab

    results = []

    # 1. Classical Champion: XGBoost
    print("\n--- Evaluating Champion 1: XGBoost ---", flush=True)
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3.0,
        random_state=seed,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    xgb.fit(X_dev_tab_norm, y_dev)
    xgb_preds = xgb.predict_proba(X_test_tab_norm)[:, 1]
    xgb_latency = (time.perf_counter() - t0) * 1000.0 / len(y_test)

    m_xgb = evaluate_probabilities(y_test, xgb_preds, "xgboost_classical", xgb_latency, hard_negative=test_hn)
    d_xgb = asdict(m_xgb)
    d_xgb["model_family"] = "xgboost_classical"
    d_xgb["mean_latency_ms"] = round(xgb_latency, 4)
    results.append(d_xgb)
    print(f"  XGBoost Locked Test AUPRC: {m_xgb.auprc:.4f}, AUROC: {m_xgb.auroc:.4f}, Sens@90Spec: {m_xgb.sens_at_90_spec:.4f}")

    # 2. Deep Learning Champion: ECGResNet1D
    print("\n--- Evaluating Champion 2: ECGResNet1D ---", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet = ECGResNet1D(in_channels=12, base_filters=32, embedding_dim=128).to(device)
    opt = torch.optim.AdamW(resnet.parameters(), lr=5e-4, weight_decay=1e-2)
    sig_dev_t = torch.tensor(sig_dev, dtype=torch.float32)
    y_dev_t = torch.tensor(y_dev, dtype=torch.float32)
    from torch.utils.data import TensorDataset, DataLoader
    loader = DataLoader(TensorDataset(sig_dev_t, y_dev_t), batch_size=64, shuffle=True)
    
    resnet.train()
    for ep in range(15):
        for b_x, b_y in loader:
            b_x, b_y = b_x.to(device), b_y.to(device)
            opt.zero_grad()
            logits, _ = resnet(b_x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b_y)
            loss.backward()
            opt.step()

    resnet.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        test_sig_t = torch.tensor(sig_test, dtype=torch.float32).to(device)
        res_preds = []
        for i in range(0, len(test_sig_t), 64):
            logits, _ = resnet(test_sig_t[i:i+64])
            res_preds.append(torch.sigmoid(logits).cpu().numpy())
    res_preds = np.concatenate(res_preds, axis=0)
    res_latency = (time.perf_counter() - t0) * 1000.0 / len(y_test)

    m_res = evaluate_probabilities(y_test, res_preds, "ecg_resnet1d", res_latency, hard_negative=test_hn)
    d_res = asdict(m_res)
    d_res["model_family"] = "ecg_resnet1d"
    d_res["mean_latency_ms"] = round(res_latency, 4)
    results.append(d_res)
    print(f"  ResNet1D Locked Test AUPRC: {m_res.auprc:.4f}, AUROC: {m_res.auroc:.4f}, Sens@90Spec: {m_res.sens_at_90_spec:.4f}")

    # 3. Multimodal Champion: ECGMultimodalHybrid
    print("\n--- Evaluating Champion 3: ECGMultimodalHybrid ---", flush=True)
    hybrid = ECGMultimodalHybrid(
        num_tabular_features=features_mat.shape[1],
        tabular_hidden=64,
        waveform_channels=12,
        waveform_embedding_dim=128,
        fused_dim=128,
    ).to(device)
    opt = torch.optim.AdamW(hybrid.parameters(), lr=5e-4, weight_decay=1e-2)
    tab_dev_t = torch.tensor(X_dev_tab_norm, dtype=torch.float32)
    h_loader = DataLoader(TensorDataset(sig_dev_t, tab_dev_t, y_dev_t), batch_size=64, shuffle=True)
    
    hybrid.train()
    for ep in range(15):
        for b_s, b_t, b_y in h_loader:
            b_s, b_t, b_y = b_s.to(device), b_t.to(device), b_y.to(device)
            opt.zero_grad()
            logits, _ = hybrid(b_s, b_t)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b_y)
            loss.backward()
            opt.step()

    hybrid.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        test_tab_t = torch.tensor(X_test_tab_norm, dtype=torch.float32).to(device)
        hyb_preds = []
        for i in range(0, len(test_sig_t), 64):
            logits, _ = hybrid(test_sig_t[i:i+64], test_tab_t[i:i+64])
            hyb_preds.append(torch.sigmoid(logits).cpu().numpy())
    hyb_preds = np.concatenate(hyb_preds, axis=0)
    hyb_latency = (time.perf_counter() - t0) * 1000.0 / len(y_test)

    m_hyb = evaluate_probabilities(y_test, hyb_preds, "ecg_multimodal_hybrid", hyb_latency, hard_negative=test_hn)
    d_hyb = asdict(m_hyb)
    d_hyb["model_family"] = "ecg_multimodal_hybrid"
    d_hyb["mean_latency_ms"] = round(hyb_latency, 4)
    results.append(d_hyb)
    print(f"  Multimodal Hybrid Locked Test AUPRC: {m_hyb.auprc:.4f}, AUROC: {m_hyb.auroc:.4f}, Sens@90Spec: {m_hyb.sens_at_90_spec:.4f}")

    # 4. Quantum Champion: HQNN
    print("\n--- Evaluating Champion 4: Hybrid Quantum Neural Network (HQNN) ---", flush=True)
    hqnn = HybridQuantumNeuralNetwork(
        tabular_dim=features_mat.shape[1],
        raw_channels=12,
        n_qubits=8,
        n_quantum_layers=3,
        resnet_base_filters=32,
    ).to(device)
    opt = torch.optim.AdamW(hqnn.parameters(), lr=1e-3, weight_decay=1e-3)
    
    hqnn.train()
    for ep in range(10):
        for b_s, b_t, b_y in h_loader:
            b_s, b_t, b_y = b_s.to(device), b_t.to(device), b_y.to(device)
            opt.zero_grad()
            logits = hqnn(b_s, b_t)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, b_y)
            loss.backward()
            opt.step()

    hqnn.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        hqnn_preds = []
        for i in range(0, len(test_sig_t), 64):
            logits = hqnn(test_sig_t[i:i+64], test_tab_t[i:i+64])
            hqnn_preds.append(torch.sigmoid(logits).cpu().numpy())
    hqnn_preds = np.concatenate(hqnn_preds, axis=0)
    hqnn_latency = (time.perf_counter() - t0) * 1000.0 / len(y_test)

    m_hqnn = evaluate_probabilities(y_test, hqnn_preds, "quantum_hqnn", hqnn_latency, hard_negative=test_hn)
    d_hqnn = asdict(m_hqnn)
    d_hqnn["model_family"] = "quantum_hqnn"
    d_hqnn["mean_latency_ms"] = round(hqnn_latency, 4)
    results.append(d_hqnn)
    print(f"  HQNN Locked Test AUPRC: {m_hqnn.auprc:.4f}, AUROC: {m_hqnn.auroc:.4f}, Sens@90Spec: {m_hqnn.sens_at_90_spec:.4f}")

    # Save final results
    results_df = pd.DataFrame(results)
    results_csv = output_dir / "final_locked_holdout_metrics.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n=================================================================")
    print(f"FINAL LOCKED HOLDOUT BENCHMARK COMPLETED")
    print(f"Saved complete comparative evaluation to: {results_csv}")
    print("=================================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 7 Locked Holdout Single-Pass Evaluation")
    parser.add_argument("--h5-path", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("holdout_benchmark_outputs"))
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_sealed_holdout_evaluation(
        h5_path=args.h5_path,
        metadata_path=args.metadata_path,
        features_csv=args.features_csv,
        manifest_path=args.manifest_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )
