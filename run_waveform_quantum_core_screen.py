"""Direct quantum-core screen on fold-coherent waveform representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.svm import SVC

from aquire_preprocessing.models_quantum import DirectQuantumClassifier
from run_quantum_baselines import _paired_patient_bootstrap
from run_quantum_core_screen import (
    _atomic_npz,
    _json_default,
    _matched_mlp,
    _patient_unique_sample,
    _train_torch_head,
)


def _embedding_to_angles(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    fit_relative: np.ndarray,
    val_embeddings: np.ndarray,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    fit_scaled = scaler.fit_transform(train_embeddings[fit_relative])
    val_scaled = scaler.transform(val_embeddings)
    pls = PLSRegression(n_components=n_components, scale=False, max_iter=1000)
    q_fit = pls.fit_transform(fit_scaled, train_labels[fit_relative])[0]
    q_val = pls.transform(val_scaled)
    quantile = QuantileTransformer(
        n_quantiles=min(256, len(q_fit)),
        output_distribution="uniform",
        random_state=seed,
    )
    q_fit = (2.0 * quantile.fit_transform(q_fit) - 1.0) * np.pi
    q_val = (2.0 * quantile.transform(q_val) - 1.0) * np.pi
    return q_fit.astype(np.float32), q_val.astype(np.float32), {
        "source_dim": int(train_embeddings.shape[1]),
        "target_dim": int(n_components),
        "fit_records": int(len(fit_relative)),
        "method": "same-encoder robust scale + supervised PLS + quantile angles",
    }


def run_screen(
    representation_dir: Path,
    metadata_path: Path,
    output_dir: Path,
    n_qubits: int = 4,
    per_class: int = 500,
    epochs: int = 20,
    batch_size: int = 64,
    seed: int = 20260922,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(metadata_path).set_index("ecg_id")
    model_names = ["waveform_direct_vqc", "waveform_matched_mlp", "waveform_rbf_svc"]
    accumulated = {
        "record_id": [],
        "patient_id": [],
        "fold": [],
        "label": [],
        "hard_negative": [],
        **{name: [] for name in model_names},
    }
    audits = []

    for held_out in range(1, 9):
        source = representation_dir / f"outer_fold_{held_out}_representations.npz"
        if not source.exists():
            raise FileNotFoundError(source)
        checkpoint = output_dir / f"fold_{held_out}.npz"
        if checkpoint.exists():
            saved = np.load(checkpoint, allow_pickle=False)
            fold_output = {key: saved[key] for key in accumulated}
            audit = json.loads(str(saved["audit"].item()))
            print(f"fold {held_out}: resumed", flush=True)
        else:
            data = np.load(source, allow_pickle=False)
            train_embeddings = data["train_embeddings"].astype(np.float32)
            val_embeddings = data["val_embeddings"].astype(np.float32)
            train_labels = data["train_labels"].astype(int)
            val_labels = data["val_labels"].astype(int)
            train_patients = data["train_patient_ids"]
            val_patients = data["val_patient_ids"]
            if set(train_patients) & set(val_patients):
                raise ValueError(f"Patient overlap in outer fold {held_out}")
            if not np.isfinite(train_embeddings).all() or not np.isfinite(val_embeddings).all():
                raise ValueError(f"Non-finite embeddings in outer fold {held_out}")
            train_record_ids = data["train_record_ids"].astype(int)
            val_record_ids = data["val_record_ids"].astype(int)
            train_meta = metadata.loc[train_record_ids]
            val_meta = metadata.loc[val_record_ids]
            if not np.array_equal(train_meta.mi_label.to_numpy(dtype=int), train_labels):
                raise ValueError(f"Training-label mismatch in outer fold {held_out}")
            if not np.array_equal(val_meta.mi_label.to_numpy(dtype=int), val_labels):
                raise ValueError(f"Validation-label mismatch in outer fold {held_out}")
            train_hard = train_meta.hard_negative.to_numpy(dtype=bool)
            val_hard = val_meta.hard_negative.to_numpy(dtype=bool)
            fit_relative = _patient_unique_sample(
                train_labels,
                train_hard,
                train_patients,
                per_class,
                seed + held_out,
            )
            q_train, q_val, representation_audit = _embedding_to_angles(
                train_embeddings,
                train_labels,
                fit_relative,
                val_embeddings,
                n_qubits,
                seed + held_out,
            )
            y_train = train_labels[fit_relative]
            print(
                f"fold {held_out}: waveform q{n_qubits}, train={len(y_train)}, "
                f"val={len(val_labels)}",
                flush=True,
            )
            vqc = DirectQuantumClassifier(n_qubits=n_qubits, n_layers=2, topology="ring")
            vqc_logits, vqc_audit = _train_torch_head(
                vqc,
                q_train,
                y_train,
                q_val,
                epochs,
                batch_size,
                learning_rate=0.01,
                seed=seed + held_out,
            )
            mlp_logits, mlp_audit = _train_torch_head(
                _matched_mlp(n_qubits),
                q_train,
                y_train,
                q_val,
                epochs,
                batch_size,
                learning_rate=0.01,
                seed=seed + held_out,
            )
            rbf = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
            rbf.fit(q_train, y_train)
            rbf_logits = rbf.decision_function(q_val)
            fold_output = {
                "record_id": val_record_ids,
                "patient_id": val_patients.astype(int),
                "fold": np.full(len(val_labels), held_out, dtype=int),
                "label": val_labels,
                "hard_negative": val_hard,
                "waveform_direct_vqc": expit(vqc_logits),
                "waveform_matched_mlp": expit(mlp_logits),
                "waveform_rbf_svc": expit(rbf_logits),
            }
            audit = {
                "held_out_fold": held_out,
                "source_artifact": source.name,
                "training_records": int(len(y_train)),
                "training_patients": int(len(np.unique(train_patients[fit_relative]))),
                "representation": representation_audit,
                "vqc": vqc_audit,
                "matched_mlp": mlp_audit,
                "coordinate_contract_verified": True,
                "probability_status": "uncalibrated_monotonic_sigmoid_of_margin",
            }
            _atomic_npz(
                checkpoint,
                **fold_output,
                audit=json.dumps(audit, default=_json_default, sort_keys=True),
            )
        for key in accumulated:
            accumulated[key].append(fold_output[key])
        audits.append(audit)

    arrays = {key: np.concatenate(values) for key, values in accumulated.items()}
    metrics = []
    rows = []
    for name in model_names:
        probability = arrays[name]
        metrics.append(
            {
                "model": name,
                "representation": f"waveform_128_pls_q{n_qubits}",
                "auprc": float(average_precision_score(arrays["label"], probability)),
                "auroc": float(roc_auc_score(arrays["label"], probability)),
                "probability_status": "uncalibrated_monotonic_sigmoid_of_margin",
            }
        )
        for index in range(len(probability)):
            rows.append(
                {
                    "ecg_id": int(arrays["record_id"][index]),
                    "patient_id": int(arrays["patient_id"][index]),
                    "fold": int(arrays["fold"][index]),
                    "label": int(arrays["label"][index]),
                    "model": name,
                    "score": float(probability[index]),
                }
            )
    metric_frame = pd.DataFrame(metrics).sort_values("auprc", ascending=False)
    metric_frame.to_csv(output_dir / "waveform_quantum_screen_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(output_dir / "waveform_quantum_screen_predictions.csv", index=False)
    (output_dir / "waveform_quantum_fold_audits.json").write_text(
        json.dumps(audits, indent=2, default=_json_default)
    )
    for control in ("waveform_matched_mlp", "waveform_rbf_svc"):
        report = _paired_patient_bootstrap(
            arrays["label"],
            arrays["patient_id"],
            arrays["waveform_direct_vqc"],
            arrays[control],
            iterations=2000,
            seed=seed,
        )
        report["comparison"] = f"waveform_direct_vqc_minus_{control}"
        (output_dir / f"paired_waveform_vqc_vs_{control}.json").write_text(
            json.dumps(report, indent=2, default=_json_default)
        )
    print(metric_frame.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representations", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260922)
    args = parser.parse_args()
    run_screen(
        args.representations,
        args.metadata,
        args.output,
        n_qubits=args.qubits,
        per_class=args.per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
