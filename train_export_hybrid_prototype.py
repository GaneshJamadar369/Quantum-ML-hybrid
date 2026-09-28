"""Train and export the fixed parallel q4-VQC + morphology-HGB prototype.

Folds 1--8 fit every predictive component. Fold 9 is used once for final
Platt calibration and the operating threshold. Fold 10 is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.preprocessing import QuantileTransformer, RobustScaler
from sklearn.cross_decomposition import PLSRegression

from aquire_preprocessing.config import CANONICAL_LEAD_ORDER, DEV_FOLDS
from aquire_preprocessing.feature_evidence import derive_clinical_composites
from aquire_preprocessing.feature_manifest import load_feature_manifest
from aquire_preprocessing.normalization import LeadRobustScaler
from aquire_preprocessing.prototype_bundle import artifact_records, verify_bundle
from run_independent_dual_route_screen import _fit_nonnegative_logistic
from run_nested_q4_optimization import Candidate, _combined_loss, _model
from run_quantum_core_screen import _patient_unique_sample
from run_transformer_representation_export import _fit_epoch_schedule
from run_waveform_representation_export import _embed, _normalise


# Kept local to make the frozen production choice explicit and auditable.
NARROW_JS = Candidate("narrow_js", 5e-3, 0.5, 0.35, 0.0)


def _atomic_json(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_split(hdf5_path: Path, metadata_path: Path, allowed_folds: set[int]):
    import h5py

    metadata = pd.read_csv(metadata_path)
    metadata = metadata[metadata.eligibility.eq("PRIMARY")].copy()
    observed = set(metadata.strat_fold.astype(int).unique())
    if not observed or not observed.issubset(allowed_folds):
        raise PermissionError(f"Unexpected folds {sorted(observed)}; allowed={sorted(allowed_folds)}")
    metadata = metadata.set_index("ecg_id", drop=False)
    with h5py.File(hdf5_path, "r") as handle:
        h5_ids = handle["ecg_id"][:].astype(int)
        lookup = {value: index for index, value in enumerate(h5_ids)}
        missing = sorted(set(metadata.ecg_id.astype(int)) - set(lookup))
        if missing:
            raise ValueError(f"HDF5 misses metadata ECG IDs; first={missing[0]}")
        rows = np.asarray([lookup[int(value)] for value in metadata.ecg_id])
        order = np.argsort(rows)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        def read(name):
            return np.asarray(handle[name][rows[order]])[inverse]
        signals = read("accepted_signal").astype(np.float32)
        sample_masks = read("sample_mask").astype(bool)
        lead_masks = read("lead_mask").astype(bool)
    if signals.shape[1:] == (1000, 12):
        signals = signals.transpose(0, 2, 1)
        sample_masks = sample_masks.transpose(0, 2, 1)
    if signals.shape[1:] != (12, 1000) or not np.isfinite(signals).all():
        raise ValueError(f"Invalid signal tensor {signals.shape}")
    return metadata.reset_index(drop=True), signals, sample_masks, lead_masks


def _load_features(path: Path, metadata: pd.DataFrame, approved: list[str]) -> np.ndarray:
    frame = pd.read_csv(path).set_index("ecg_id")
    # Calibration extraction produces base features. Derive exactly the same
    # deterministic clinical composites used by the frozen development file.
    if not set(approved).issubset(frame.columns):
        derived, _ = derive_clinical_composites(frame)
        frame = pd.concat([frame, derived], axis=1)
    missing = sorted(set(approved) - set(frame.columns))
    if missing:
        raise ValueError(f"Feature file omits approved deployable features: {missing}")
    if not frame.index.is_unique:
        raise ValueError("Feature ecg_id must be unique")
    return frame.loc[metadata.ecg_id.astype(int), approved].to_numpy(np.float32)


def _teacher(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=180, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=1.0, random_state=int(seed),
    )


def _fit_clinical(dev_x, y, folds, cal_x, seed):
    raw_oof = np.full(len(y), np.nan)
    for held_out in sorted(np.unique(folds)):
        fit, val = folds != held_out, folds == held_out
        imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(dev_x[fit])
        model = _teacher(seed + int(held_out)).fit(imputer.transform(dev_x[fit]), y[fit])
        p = model.predict_proba(imputer.transform(dev_x[val]))[:, 1]
        raw_oof[val] = logit(np.clip(p, 1e-6, 1 - 1e-6))
    calibrator = LogisticRegression(C=1.0, max_iter=2000).fit(raw_oof[:, None], y)
    oof = calibrator.predict_proba(raw_oof[:, None])[:, 1]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(dev_x)
    model = _teacher(seed).fit(imputer.transform(dev_x), y)
    cal_raw = logit(np.clip(model.predict_proba(imputer.transform(cal_x))[:, 1], 1e-6, 1 - 1e-6))
    cal = calibrator.predict_proba(cal_raw[:, None])[:, 1]
    return oof.astype(np.float32), cal.astype(np.float32), imputer, model, calibrator


def _fit_q4_artifacts(train_h, train_y, cal_h, seed):
    imputer = SimpleImputer(strategy="median").fit(train_h)
    scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(imputer.transform(train_h))
    train_x = scaler.transform(imputer.transform(train_h))
    cal_x = scaler.transform(imputer.transform(cal_h))
    pls = PLSRegression(n_components=4, scale=False, max_iter=1000).fit(train_x, train_y)
    train_q, cal_q = pls.transform(train_x), pls.transform(cal_x)
    signs = np.ones(4)
    for component in range(4):
        correlation = np.corrcoef(train_q[:, component], train_y)[0, 1]
        if np.isfinite(correlation) and correlation < 0:
            signs[component] = -1.0
    train_q *= signs
    cal_q *= signs
    quantile = QuantileTransformer(
        n_quantiles=min(256, len(train_q)), output_distribution="uniform", random_state=seed,
    ).fit(train_q)
    train_q = (2 * quantile.transform(train_q) - 1) * (np.pi / 2)
    cal_q = (2 * quantile.transform(cal_q) - 1) * (np.pi / 2)
    return train_q.astype(np.float32), cal_q.astype(np.float32), imputer, scaler, pls, signs, quantile


def _fit_vqc(train_q, y, teacher, patients, hard, *, seed, epochs, batch_size, device):
    import torch

    model = _model(NARROW_JS, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=2e-4)
    sample = _patient_unique_sample(y, hard, patients, per_class=2000, seed=seed)
    x = torch.from_numpy(train_q[sample]).to(device)
    target = torch.from_numpy(y[sample].astype(np.float32)).to(device)
    soft = torch.from_numpy(teacher[sample].astype(np.float32)).to(device)
    order, rng = np.arange(len(sample)), np.random.default_rng(seed)
    history = []
    for epoch in range(epochs):
        rng.shuffle(order)
        running = 0.0
        model.train()
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = _combined_loss(model(x[batch]), target[batch], soft[batch], NARROW_JS)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite VQC loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += float(loss.detach()) * len(batch)
        scheduler.step()
        history.append(running / len(order))
    model.eval()
    with torch.inference_mode():
        reference = model(torch.from_numpy(train_q).to(device)).cpu().numpy()
    return model, np.sort(reference.astype(np.float32)), history, sample


def _select_fusion_c(s_c, s_q, y, folds):
    raw = np.column_stack([logit(np.clip(s_c, 1e-6, 1-1e-6)), logit(np.clip(s_q, 1e-6, 1-1e-6))])
    losses = {}
    for c in (0.1, 1.0, 10.0):
        values = []
        for fold in sorted(np.unique(folds)):
            fit, val = folds != fold, folds == fold
            mean, scale = raw[fit].mean(0), raw[fit].std(0).clip(1e-6)
            parameters = _fit_nonnegative_logistic((raw[fit]-mean)/scale, y[fit], c)
            probability = expit(parameters[0] + ((raw[val]-mean)/scale) @ parameters[1:])
            values.append(float(-np.mean(y[val]*np.log(np.clip(probability,1e-7,1)) + (1-y[val])*np.log(np.clip(1-probability,1e-7,1)))))
        losses[c] = float(np.mean(values))
    return min(losses, key=losses.get), losses, raw


def _threshold_at_specificity(y, probability, target=0.90):
    fpr, tpr, thresholds = roc_curve(y, probability)
    valid = np.flatnonzero((1.0 - fpr >= target) & np.isfinite(thresholds))
    if not len(valid):
        raise RuntimeError("No finite threshold satisfies target specificity")
    best = valid[np.argmax(tpr[valid])]
    return float(thresholds[best]), float(tpr[best]), float(1.0-fpr[best])


def run(args) -> None:
    import joblib
    import sklearn
    import torch

    args.bundle.mkdir(parents=True, exist_ok=True)
    dev_meta, dev_signal, dev_sample_mask, dev_lead_mask = _load_split(
        args.development_hdf5, args.development_metadata, set(DEV_FOLDS)
    )
    cal_meta, cal_signal, _, _ = _load_split(
        args.calibration_hdf5, args.calibration_metadata, {9}
    )
    if 10 in set(dev_meta.strat_fold) | set(cal_meta.strat_fold):
        raise PermissionError("Fold 10 is sealed")
    manifest = load_feature_manifest(args.feature_manifest, pd.read_csv(args.development_features, nrows=1).columns)
    approved = list(manifest["approved_features"])
    dev_features = _load_features(args.development_features, dev_meta, approved)
    cal_features = _load_features(args.calibration_features, cal_meta, approved)
    y = dev_meta.mi_label.to_numpy(int)
    cal_y = cal_meta.mi_label.to_numpy(int)
    folds = dev_meta.strat_fold.to_numpy(int)

    scaler = LeadRobustScaler().fit(
        dev_signal, folds=folds, allowed_folds=list(DEV_FOLDS),
        sample_masks=dev_sample_mask, lead_masks=dev_lead_mask,
        patient_ids=dev_meta.patient_id.to_numpy(),
    )
    normalizer_path = args.bundle / "waveform_normalizer.json"
    scaler.params.to_json(normalizer_path)
    dev_norm = scaler.transform(dev_signal)
    cal_norm = scaler.transform(cal_signal)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    transformer, transformer_history, _ = _fit_epoch_schedule(
        dev_norm, dev_meta, np.arange(len(dev_meta)), None,
        max_epochs=args.transformer_epochs, fixed_epochs=args.transformer_epochs,
        batch_size=args.transformer_batch_size, learning_rate=3e-4,
        seed=args.seed, device=device,
    )
    transformer_path = args.bundle / "transformer.pt"
    torch.save({"state_dict": {k:v.detach().cpu() for k,v in transformer.state_dict().items()},
                "epochs": args.transformer_epochs}, transformer_path)
    dev_h = _embed(transformer, dev_norm, device, args.transformer_batch_size)[0]
    cal_h = _embed(transformer, cal_norm, device, args.transformer_batch_size)[0]
    train_q, cal_q, h_imputer, h_scaler, pls, signs, quantile = _fit_q4_artifacts(dev_h, y, cal_h, args.seed)

    clinical_seed = int(args.vqc_seeds[0])
    clinical_oof, clinical_cal, morphology_imputer, hgb, morphology_calibrator = _fit_clinical(
        dev_features, y, folds, cal_features, clinical_seed
    )
    models, references, histories = [], [], []
    for restart, restart_seed in enumerate(args.vqc_seeds):
        model, reference, history, sample = _fit_vqc(
            train_q, y, clinical_oof, dev_meta.patient_id.to_numpy(int),
            dev_meta.hard_negative.to_numpy(bool), seed=restart_seed,
            epochs=args.vqc_epochs, batch_size=args.vqc_batch_size, device=device,
        )
        path = args.bundle / f"vqc_restart_{restart}.pt"
        torch.save({"state_dict": {k:v.detach().cpu() for k,v in model.state_dict().items()},
                    "seed": restart_seed, "sample_ecg_ids": dev_meta.ecg_id.to_numpy()[sample]}, path)
        models.append(model); references.append(reference); histories.append(history)

    oof = pd.read_csv(args.oof_scores).set_index("ecg_id").loc[dev_meta.ecg_id]
    if not np.array_equal(oof.y_true.to_numpy(int), y) or not np.array_equal(oof.strat_fold.to_numpy(int), folds):
        raise ValueError("OOF score artifact does not align with development records")
    q_calibrator = LogisticRegression(C=1.0, max_iter=2000).fit(
        oof.score_vqc_train_cdf.to_numpy()[:, None], y
    )
    oof_q = q_calibrator.predict_proba(oof.score_vqc_train_cdf.to_numpy()[:, None])[:, 1]
    oof_c = oof.clinical_teacher.to_numpy(float)
    selected_c, c_losses, fusion_raw = _select_fusion_c(oof_c, oof_q, y, folds)
    fusion_mean, fusion_scale = fusion_raw.mean(0), fusion_raw.std(0).clip(1e-6)
    fusion_parameters = _fit_nonnegative_logistic(
        (fusion_raw-fusion_mean)/fusion_scale, y, selected_c
    )

    with torch.inference_mode():
        cal_tensor = torch.from_numpy(cal_q).to(device)
        cal_cdfs = []
        for model, reference in zip(models, references):
            raw = model(cal_tensor).cpu().numpy()
            cal_cdfs.append((np.searchsorted(reference, raw, side="right") + 0.5) / (len(reference)+1.0))
    cal_q_score = q_calibrator.predict_proba(np.mean(cal_cdfs, axis=0)[:, None])[:, 1]
    cal_raw = np.column_stack([
        logit(np.clip(clinical_cal,1e-6,1-1e-6)),
        logit(np.clip(cal_q_score,1e-6,1-1e-6)),
    ])
    cal_fusion_logit = fusion_parameters[0] + ((cal_raw-fusion_mean)/fusion_scale) @ fusion_parameters[1:]
    final_calibrator = LogisticRegression(C=1.0, max_iter=2000).fit(cal_fusion_logit[:, None], cal_y)
    cal_probability = final_calibrator.predict_proba(cal_fusion_logit[:, None])[:, 1]
    threshold, sensitivity, specificity = _threshold_at_specificity(cal_y, cal_probability)

    joblib.dump(h_imputer, args.bundle / "h128_imputer.joblib")
    joblib.dump(h_scaler, args.bundle / "h128_scaler.joblib")
    joblib.dump({"model": pls, "signs": signs}, args.bundle / "pls_q4.joblib")
    joblib.dump(quantile, args.bundle / "angle_quantiles.joblib")
    joblib.dump({"train_reference_logits": references, "calibrator": q_calibrator}, args.bundle / "vqc_score_alignment.joblib")
    shutil.copy2(args.feature_manifest, args.bundle / "morphology_feature_manifest.json")
    joblib.dump({"imputer": morphology_imputer, "calibrator": morphology_calibrator}, args.bundle / "morphology_conditioner.joblib")
    joblib.dump(hgb, args.bundle / "morphology_hgb.joblib")
    _atomic_json({"mean": fusion_mean.tolist(), "scale": fusion_scale.tolist(),
                  "intercept": float(fusion_parameters[0]), "weights": fusion_parameters[1:].tolist(),
                  "selected_c": selected_c, "candidate_logloss": c_losses}, args.bundle / "fusion.json")
    joblib.dump(final_calibrator, args.bundle / "platt_calibrator.joblib")
    _atomic_json({"threshold": threshold, "target_specificity": 0.90,
                  "observed_sensitivity": sensitivity, "observed_specificity": specificity}, args.bundle / "decision_threshold.json")
    np.savez_compressed(args.bundle / "golden_fold9_cases.npz",
                        signal_mv=cal_signal[:8], ecg_id=cal_meta.ecg_id.to_numpy()[:8],
                        y_true=cal_y[:8], probability=cal_probability[:8])

    feature_hash = hashlib.sha256(args.feature_manifest.read_bytes()).hexdigest()
    entries = [
        ("transformer", "transformer.pt"), ("waveform_normalizer", "waveform_normalizer.json"),
        ("h128_imputer", "h128_imputer.joblib"), ("h128_scaler", "h128_scaler.joblib"),
        ("pls_q4", "pls_q4.joblib"), ("angle_quantiles", "angle_quantiles.joblib"),
        ("vqc_score_alignment", "vqc_score_alignment.joblib"),
        ("morphology_feature_manifest", "morphology_feature_manifest.json"),
        ("morphology_conditioner", "morphology_conditioner.joblib"),
        ("morphology_hgb", "morphology_hgb.joblib"), ("fusion", "fusion.json"),
        ("platt_calibrator", "platt_calibrator.joblib"), ("decision_threshold", "decision_threshold.json"),
        ("training_report", "training_report.json"), ("golden_fixture", "golden_fold9_cases.npz"),
    ] + [("vqc_model", f"vqc_restart_{i}.pt") for i in range(len(models))]
    report = {
        "development_records": len(dev_meta), "calibration_records": len(cal_meta),
        "development_folds": sorted(np.unique(folds).tolist()), "calibration_fold": 9,
        "fold_10_accessed": False, "transformer_epochs": args.transformer_epochs,
        "vqc_epochs": args.vqc_epochs, "vqc_seeds": args.vqc_seeds,
        "clinical_seed": clinical_seed,
        "fold9_auprc": float(average_precision_score(cal_y, cal_probability)),
        "fold9_auroc": float(roc_auc_score(cal_y, cal_probability)),
        "fold9_brier": float(brier_score_loss(cal_y, cal_probability)),
        "threshold": threshold, "sensitivity": sensitivity, "specificity": specificity,
        "transformer_training_history": transformer_history,
        "vqc_training_history": histories,
        "claim_boundary": "MI-pattern research screening; no quantum-advantage or clinical-deployment claim",
    }
    _atomic_json(report, args.bundle / "training_report.json")
    manifest_payload = {
        "schema_version": 1, "model_version": args.model_version,
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": _git_commit(),
        "task": "mi_pattern_vs_non_mi_pattern", "prediction_routing": "fixed_parallel_all_eligible_inputs",
        "input": {"shape": [12,1000], "sampling_rate_hz": 100, "physical_units": "mV"},
        "routes": {
            "quantum": {"active": True, "model": "transformer_pls_q4_vqc_ensemble"},
            "classical": {"active": True, "model": "morphology_106_hgb"},
            "fusion": {"active": True, "model": "nonnegative_logistic"},
        },
        "calibration_state": "fold9_frozen",
        "artifact_runtime": {
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "joblib": joblib.__version__,
        },
        "training_patient_sha256": scaler.params.training_patient_checksum,
        "feature_manifest_sha256": feature_hash,
        "artifacts": artifact_records(args.bundle, entries),
    }
    _atomic_json(manifest_payload, args.bundle / "manifest.json")
    verify_bundle(args.bundle)
    print(json.dumps(report, indent=2, default=str), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-hdf5", type=Path, required=True)
    parser.add_argument("--development-metadata", type=Path, required=True)
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--calibration-hdf5", type=Path, required=True)
    parser.add_argument("--calibration-metadata", type=Path, required=True)
    parser.add_argument("--calibration-features", type=Path, required=True)
    parser.add_argument("--oof-scores", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--model-version", default="aquire-hybrid-q4-v1")
    parser.add_argument("--transformer-epochs", type=int, default=13)
    parser.add_argument("--transformer-batch-size", type=int, default=128)
    parser.add_argument("--vqc-epochs", type=int, default=60)
    parser.add_argument("--vqc-batch-size", type=int, default=128)
    parser.add_argument("--vqc-seeds", type=int, nargs="+", default=[42,31415,27182])
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--device", choices=("auto","cpu","cuda"), default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
