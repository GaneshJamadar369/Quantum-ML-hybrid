"""Zero-initialized classical residual adapter for a q4 VQC core.

The proven fold-local Transformer->PLS q4 coordinates remain the identity
path.  A small classical adapter may correct those angles using h128 alone or
h128 plus deployable clinical features.  The VQC and a matched classical
system are trained independently with the same data and optimization budget.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

from aquire_preprocessing.config import DEV_FOLDS
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.manifest import guard_fold_access
from aquire_preprocessing.models_advanced_fusion import ResidualAngleAdapter
from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier
from run_advanced_quantum_fusion_screen import _fit_q4, _seed, _train_vqc as _train_direct_vqc
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import _patient_unique_sample


class _SystemFactory:
    @staticmethod
    def build(mode: str, quantum: bool):
        from torch import nn

        clinical_dim = 106 if mode == "h128_clinical" else None

        class System(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = ResidualAngleAdapter(
                    waveform_dim=128, clinical_dim=clinical_dim,
                    hidden=16, max_residual=0.25, dropout=0.10,
                )
                if quantum:
                    self.head = TorchStatevectorQuantumClassifier(4, n_layers=2, topology="ring")
                else:
                    self.head = nn.Sequential(nn.Linear(4, 7), nn.ReLU(), nn.Linear(7, 1))

            def forward(self, base, waveform, clinical=None, observed=None):
                angles, residual = self.adapter(base, waveform, clinical, observed)
                return self.head(angles).squeeze(-1), angles, residual

        return System()


def _train_system(
    mode: str,
    quantum: bool,
    train_base: np.ndarray,
    train_h: np.ndarray,
    train_clinical: np.ndarray | None,
    train_observed: np.ndarray | None,
    train_y: np.ndarray,
    train_hard: np.ndarray,
    val_base: np.ndarray,
    val_h: np.ndarray,
    val_clinical: np.ndarray | None,
    val_observed: np.ndarray | None,
    *,
    epochs: int,
    batch_size: int,
    seed: int,
    device,
):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _seed(seed)
    model = _SystemFactory.build(mode, quantum).to(device)
    clinical_train = (
        train_clinical if train_clinical is not None else np.zeros((len(train_y), 0), np.float32)
    )
    observed_train = (
        train_observed if train_observed is not None else np.zeros((len(train_y), 0), np.float32)
    )
    weights = np.ones(len(train_y), dtype=np.float32)
    weights[(train_y == 0) & train_hard] = 1.25
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_base), torch.from_numpy(train_h),
            torch.from_numpy(clinical_train), torch.from_numpy(observed_train),
            torch.from_numpy(train_y.astype(np.float32)), torch.from_numpy(weights),
        ),
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        seen = 0
        for base, waveform, clinical, observed, target, weight in loader:
            base, waveform = base.to(device), waveform.to(device)
            target, weight = target.to(device), weight.to(device)
            clinical_arg = clinical.to(device) if mode == "h128_clinical" else None
            observed_arg = observed.to(device) if mode == "h128_clinical" else None
            optimizer.zero_grad(set_to_none=True)
            logits, _, residual = model(base, waveform, clinical_arg, observed_arg)
            prediction = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, target, reduction="none"
            )
            identity = residual.square().mean(dim=1)
            loss = (weight * (prediction + 0.10 * identity)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(base)
            seen += len(base)
        scheduler.step()
        losses.append(total / max(seen, 1))

    def infer(base, waveform, clinical, observed):
        model.eval()
        scores, angles, residuals = [], [], []
        with torch.inference_mode():
            for start in range(0, len(base), batch_size):
                b = torch.from_numpy(base[start : start + batch_size]).to(device)
                h = torch.from_numpy(waveform[start : start + batch_size]).to(device)
                c = None if clinical is None else torch.from_numpy(clinical[start : start + batch_size]).to(device)
                m = None if observed is None else torch.from_numpy(observed[start : start + batch_size]).to(device)
                score, q, delta = model(b, h, c, m)
                scores.append(score.cpu().numpy())
                angles.append(q.cpu().numpy())
                residuals.append(delta.cpu().numpy())
        return np.concatenate(scores), np.concatenate(angles), np.concatenate(residuals)

    train_result = infer(train_base, train_h, train_clinical, train_observed)
    val_result = infer(val_base, val_h, val_clinical, val_observed)
    shuffled = None
    if mode == "h128_clinical":
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(val_base))
        shuffled = infer(val_base, val_h, val_clinical[order], val_observed[order])[0]
    audit = {
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "initial_loss": float(losses[0]), "final_loss": float(losses[-1]),
        "loss_declined": bool(losses[-1] < losses[0]),
        "adapter_gate": float(model.adapter.gate_logit.sigmoid().detach().cpu()),
        "mean_abs_validation_angle_change": float(np.abs(val_result[2]).mean()),
    }
    return train_result, val_result, shuffled, audit


def _metrics(y, logits):
    probability = expit(logits)
    return {
        "auprc": float(average_precision_score(y, probability)),
        "auroc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
    }


def run(args) -> None:
    import torch

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(args.metadata).set_index("ecg_id")
    feature_frame = pd.read_csv(args.features).set_index("ecg_id")
    manifest = load_feature_manifest(args.manifest, feature_frame.columns)
    approved = manifest["approved_features"]
    joined = metadata.join(feature_frame[approved], how="inner", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")].copy()
    folds = joined.strat_fold.to_numpy(int)
    guard_fold_access(folds, purpose="tuning")
    labels = joined.mi_label.to_numpy(int)
    patients = joined.patient_id.to_numpy(int)
    hard = joined.hard_negative.to_numpy(bool)
    record_ids = joined.index.to_numpy(int)
    clinical_raw = joined[approved].to_numpy(np.float32)
    observed_all = np.isfinite(clinical_raw).astype(np.float32)
    row_for_id = {int(ecg_id): row for row, ecg_id in enumerate(record_ids)}
    names = [
        "base_vqc", "adapted_vqc", "adapted_mlp_system", "same_q_logistic",
        "same_q_mlp", "same_q_rbf", "clinical_shuffle",
    ]
    predictions = {name: np.full(len(joined), np.nan) for name in names}
    audits = []
    for held_out_value in sorted(np.unique(folds)):
        held_out = int(held_out_value)
        path = args.representations / f"outer_fold_{held_out}_representations.npz"
        with np.load(path, allow_pickle=False) as rep:
            train_ids = rep["train_record_ids"].astype(int)
            val_ids = rep["val_record_ids"].astype(int)
            train_h_raw = rep["train_embeddings"].astype(np.float32)
            val_h_raw = rep["val_embeddings"].astype(np.float32)
            train_y = rep["train_labels"].astype(int)
            train_patients = rep["train_patient_ids"].astype(int)
        train_rows = np.asarray([row_for_id[x] for x in train_ids])
        val_rows = np.asarray([row_for_id[x] for x in val_ids])
        if set(train_patients) & set(patients[val_rows]):
            raise RuntimeError("Patient leakage")
        base_train, base_val, _ = _fit_q4(train_h_raw, train_y, val_h_raw, args.seed + held_out)
        h_scaler = RobustScaler(quantile_range=(25, 75))
        train_h = h_scaler.fit_transform(train_h_raw).astype(np.float32)
        val_h = h_scaler.transform(val_h_raw).astype(np.float32)
        train_clinical = val_clinical = train_observed = val_observed = None
        if args.mode == "h128_clinical":
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            c_scaler = RobustScaler(quantile_range=(25, 75))
            train_clinical = c_scaler.fit_transform(
                imputer.fit_transform(clinical_raw[train_rows])
            ).astype(np.float32)
            val_clinical = c_scaler.transform(imputer.transform(clinical_raw[val_rows])).astype(np.float32)
            train_observed = observed_all[train_rows]
            val_observed = observed_all[val_rows]
        sample = _patient_unique_sample(
            train_y, hard[train_rows], train_patients,
            per_class=args.per_class, seed=args.seed + held_out,
        )
        common = dict(
            mode=args.mode,
            train_base=base_train[sample], train_h=train_h[sample],
            train_clinical=None if train_clinical is None else train_clinical[sample],
            train_observed=None if train_observed is None else train_observed[sample],
            train_y=train_y[sample], train_hard=hard[train_rows][sample],
            val_base=base_val, val_h=val_h, val_clinical=val_clinical,
            val_observed=val_observed, epochs=args.epochs, batch_size=args.batch_size,
            device=device,
        )
        q_train_result, q_val_result, shuffled, q_audit = _train_system(
            quantum=True, seed=args.seed + 100 * held_out, **common
        )
        _, classical_val_result, _, classical_audit = _train_system(
            quantum=False, seed=args.seed + 100 * held_out, **common
        )
        q_train_angles = q_train_result[1]
        q_val_logits, q_val_angles, _ = q_val_result
        predictions["adapted_vqc"][val_rows] = q_val_logits
        predictions["adapted_mlp_system"][val_rows] = classical_val_result[0]
        predictions["clinical_shuffle"][val_rows] = q_val_logits if shuffled is None else shuffled
        # Same learned angles: strict head-only controls.
        y_sample = train_y[sample]
        lr = LogisticRegression(C=1, class_weight="balanced", max_iter=2000).fit(q_train_angles, y_sample)
        predictions["same_q_logistic"][val_rows] = lr.decision_function(q_val_angles)
        mlp = MLPClassifier(hidden_layer_sizes=(7,), max_iter=500, random_state=args.seed + held_out).fit(q_train_angles, y_sample)
        probability = mlp.predict_proba(q_val_angles)[:, 1]
        predictions["same_q_mlp"][val_rows] = np.log(np.clip(probability, 1e-5, 1-1e-5) / np.clip(1-probability, 1e-5, 1-1e-5))
        rbf = SVC(C=1, kernel="rbf", class_weight="balanced").fit(q_train_angles, y_sample)
        predictions["same_q_rbf"][val_rows] = rbf.decision_function(q_val_angles)
        # Identity-path VQC with the same records, epochs, batch size and seed.
        base_logits, _ = _train_direct_vqc(
            base_train[sample], y_sample, base_val,
            epochs=args.epochs, batch_size=args.batch_size,
            seed=args.seed + 100 * held_out, device=device, entanglement=True,
        )
        predictions["base_vqc"][val_rows] = base_logits
        audits.append({"fold": held_out, "quantum": q_audit, "classical": classical_audit})
        print(
            f"fold {held_out}: base={average_precision_score(labels[val_rows], expit(base_logits)):.4f} "
            f"adapted={average_precision_score(labels[val_rows], expit(q_val_logits)):.4f}", flush=True,
        )

    metrics = {name: _metrics(labels, logits) for name, logits in predictions.items()}
    comparisons = {}
    for control in ("base_vqc", "adapted_mlp_system", "same_q_logistic", "same_q_mlp", "same_q_rbf", "clinical_shuffle"):
        comparisons[f"adapted_vqc_minus_{control}"] = _paired_patient_bootstrap(
            labels, patients, expit(predictions["adapted_vqc"]), expit(predictions[control]),
            iterations=args.bootstrap_iterations, seed=args.seed,
            comparison=f"adapted_vqc_minus_{control}",
        )
    best_control = max(("same_q_logistic", "same_q_mlp", "same_q_rbf"), key=lambda x: metrics[x]["auprc"])
    delta = metrics["adapted_vqc"]["auprc"] - metrics[best_control]["auprc"]
    ci = comparisons[f"adapted_vqc_minus_{best_control}"]["delta_auprc"]
    verdict = {
        "mode": args.mode,
        "improves_base_vqc": metrics["adapted_vqc"]["auprc"] > metrics["base_vqc"]["auprc"] + 0.005,
        "quantum_gate": bool(delta >= 0.005 and ci["ci95_low"] > 0),
        "best_same_q_control": best_control,
        "vqc_delta_auprc": delta,
        "fold_9_accessed": False, "fold_10_accessed": False,
    }
    pd.DataFrame({
        "ecg_id": record_ids, "patient_id": patients, "strat_fold": folds, "y_true": labels,
        **{name: expit(value) for name, value in predictions.items()},
    }).to_csv(output / "oof_predictions.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (output / "bootstrap.json").write_text(json.dumps(comparisons, indent=2))
    (output / "audits.json").write_text(json.dumps(audits, indent=2))
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"metrics": metrics, "verdict": verdict}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["h128", "h128_clinical"], required=True)
    parser.add_argument("--per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260924)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
