"""Phase 6Q-R: Quantum Residual Specialist.

This script takes the frozen 128D Out-Of-Fold (OOF) embeddings and logits
from the Phase 6C-R classical deep learning champion (1D-ResNet / Hybrid).
It trains a Projected Quantum Kernel (PQK) via Kernel Ridge Regression
to predict the residual errors (y - p) of the classical model, focusing
specifically on hard-negative clinical confounders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.models_quantum import ProjectedIQPFeatureMap
from run_quantum_baselines import _paired_patient_bootstrap


def _kernel_health(kernel: np.ndarray) -> dict:
    matrix = (np.asarray(kernel) + np.asarray(kernel).T) / 2.0
    eig = np.linalg.eigvalsh(matrix)
    positive = np.clip(eig, 0.0, None)
    probability = positive / positive.sum() if positive.sum() > 0 else positive
    probability = probability[probability > 0]
    effective_rank = float(np.exp(-np.sum(probability * np.log(probability)))) if len(probability) else 0.0
    offdiag = matrix[~np.eye(len(matrix), dtype=bool)]
    return {
        "effective_rank": effective_rank,
        "minimum_eigenvalue": float(eig.min()),
        "offdiagonal_mean": float(offdiag.mean()) if len(offdiag) else 0.0,
        "offdiagonal_std": float(offdiag.std()) if len(offdiag) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantum Residual Specialist")
    parser.add_argument("--hybrid-oof", type=Path, required=True, help="Path to hybrid_oof.npz")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Loading deep learning champion OOF: {args.hybrid_oof}", flush=True)
    oof_data = np.load(args.hybrid_oof)
    
    record_ids = oof_data["record_ids"]
    labels = oof_data["labels"]
    folds = oof_data["folds"]
    hard_neg = oof_data["hard_negative"]
    base_logits = oof_data["logits"]
    base_probs = oof_data["calibrated_prob"]
    embeddings = oof_data["embeddings"]  # Shape: (N, 128)
    
    # Calculate residuals (gradient of log loss)
    residuals = labels - base_probs
    
    # We will train fold-local residual models
    final_hybrid_logits = np.full(len(labels), np.nan)
    
    for held_out in sorted(np.unique(folds)):
        print(f"\nEvaluating Quantum Residual on Fold {held_out}...", flush=True)
        train_idx = np.where(folds != held_out)[0]
        val_idx = np.where(folds == held_out)[0]
        
        # 1. Project 128D embeddings to 8 dimensions using PLS on the residual!
        # This forces the PLS to find the subspace of the embedding that explains the classical model's errors.
        pls = PLSRegression(n_components=8, scale=True)
        z_train = pls.fit_transform(embeddings[train_idx], residuals[train_idx])[0]
        z_val = pls.transform(embeddings[val_idx])
        
        # 2. Quantile Angle transformation for the Quantum Map
        quantiles = QuantileTransformer(n_quantiles=256, output_distribution="uniform", random_state=args.seed)
        q_train = (2.0 * quantiles.fit_transform(z_train) - 1.0) * np.pi
        q_val = (2.0 * quantiles.transform(z_val) - 1.0) * np.pi
        
        # 3. Projected Quantum Kernel (PQK)
        print("  Constructing Projected Quantum Kernel...", flush=True)
        fmap = ProjectedIQPFeatureMap(
            n_qubits=8,
            n_layers=1,
            feature_scale=1.0,
            interaction_scale=0.5,
            topology="ladder",
            mixing_seed=args.seed,
        )
        
        # Extract observables
        obs_train = fmap.transform(q_train)
        obs_val = fmap.transform(q_val)
        
        # Compute Kernel
        from sklearn.metrics import pairwise_distances
        dist_train = pairwise_distances(obs_train, obs_train, metric="sqeuclidean")
        dist_val = pairwise_distances(obs_val, obs_train, metric="sqeuclidean")
        
        # Auto-bandwidth
        pos = dist_train[np.triu_indices_from(dist_train, k=1)]
        gamma = 1.0 / float(np.median(pos)) if len(pos) else 1.0
        
        k_train = np.exp(-gamma * dist_train)
        k_val = np.exp(-gamma * dist_val)
        
        print("  Kernel Health:", _kernel_health(k_train))
        
        # 4. Kernel Ridge Regression to predict residuals
        # Prioritize hard negatives
        weights = np.ones(len(train_idx))
        weights[(labels[train_idx] == 0) & hard_neg[train_idx]] = 2.0
        
        krr = KernelRidge(alpha=1.0, kernel="precomputed")
        krr.fit(k_train, residuals[train_idx], sample_weight=weights)
        
        pred_residuals = krr.predict(k_val)
        
        # 5. Blend residual prediction with base logit using a simple logistic regression mapping
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(class_weight="balanced")
        
        train_krr_preds = krr.predict(k_train)
        X_blend_train = np.column_stack([base_logits[train_idx], train_krr_preds])
        lr.fit(X_blend_train, labels[train_idx])
        
        X_blend_val = np.column_stack([base_logits[val_idx], pred_residuals])
        val_final_logits = lr.decision_function(X_blend_val)
        
        final_hybrid_logits[val_idx] = val_final_logits
        
        base_auprc = average_precision_score(labels[val_idx], base_probs[val_idx])
        new_auprc = average_precision_score(labels[val_idx], val_final_logits)
        print(f"  Fold {held_out} AUPRC - Base: {base_auprc:.4f} -> Quantum Residual: {new_auprc:.4f}", flush=True)

    # Save results
    final_probs = 1.0 / (1.0 + np.exp(-final_hybrid_logits))
    
    np.savez_compressed(
        args.output / "quantum_residual_oof.npz",
        record_ids=record_ids,
        labels=labels,
        folds=folds,
        base_probs=base_probs,
        final_probs=final_probs,
    )
    
    print("\n--- Final Quantum Residual Specialist Results ---", flush=True)
    base_auprc = average_precision_score(labels, base_probs)
    final_auprc = average_precision_score(labels, final_probs)
    print(f"Global OOF AUPRC - Base: {base_auprc:.4f} | Quantum: {final_auprc:.4f}")
    

if __name__ == "__main__":
    main()
