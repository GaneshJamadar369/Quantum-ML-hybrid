"""Aggregate the frozen two-seed train-reference score-alignment screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_divergence_distillation_screen import _metrics
from run_quantum_baselines import _paired_patient_bootstrap


MODELS = (
    "vqc_raw",
    "vqc_train_z",
    "vqc_train_cdf",
    "vqc_no_entanglement_cdf",
    "q4_logistic",
    "q4_mlp_ensemble",
    "clinical_teacher",
    "fusion_vqc_train_cdf",
    "fusion_q4_mlp",
)


def _discover(roots):
    found = {}
    for root in roots:
        for path in root.rglob("oof_predictions_and_observables.csv"):
            seed = int(path.parent.name.removeprefix("seed_"))
            if seed in found:
                raise ValueError(f"Duplicate seed {seed}")
            found[seed] = path
    return found


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    paths = _discover(args.inputs)
    if set(paths) != set(args.expected_seeds):
        raise ValueError(f"Expected {sorted(args.expected_seeds)}, found {sorted(paths)}")
    frames = {
        seed: pd.read_csv(path).sort_values("ecg_id").reset_index(drop=True)
        for seed, path in paths.items()
    }
    identity = ["ecg_id", "patient_id", "strat_fold", "y_true", "hard_negative"]
    reference = frames[min(frames)]
    for seed, frame in frames.items():
        if not frame[identity].equals(reference[identity]):
            raise ValueError(f"Seed {seed} cohort differs")
        if frame[list(MODELS)].isna().any().any():
            raise ValueError(f"Seed {seed} is incomplete")
    labels = reference.y_true.to_numpy(int)
    patients = reference.patient_id.to_numpy(int)
    seed_rows = []
    for seed, frame in sorted(frames.items()):
        metrics = {name: _metrics(labels, frame[name].to_numpy(float)) for name in MODELS}
        strongest = max(("q4_logistic", "q4_mlp_ensemble"), key=lambda name: metrics[name]["auprc"])
        seed_rows.append(
            {
                "seed": seed,
                "raw_vqc_auprc": metrics["vqc_raw"]["auprc"],
                "cdf_vqc_auprc": metrics["vqc_train_cdf"]["auprc"],
                "alignment_delta": metrics["vqc_train_cdf"]["auprc"] - metrics["vqc_raw"]["auprc"],
                "strongest_control": strongest,
                "control_auprc": metrics[strongest]["auprc"],
                "quantum_delta": metrics["vqc_train_cdf"]["auprc"] - metrics[strongest]["auprc"],
                "fusion_vqc_auprc": metrics["fusion_vqc_train_cdf"]["auprc"],
                "fusion_classical_auprc": metrics["fusion_q4_mlp"]["auprc"],
            }
        )
    averaged = reference[identity].copy()
    for name in MODELS:
        averaged[name] = np.mean(
            [frame[name].to_numpy(float) for frame in frames.values()], axis=0
        )
    metrics = {name: _metrics(labels, averaged[name].to_numpy(float)) for name in MODELS}
    strongest = max(("q4_logistic", "q4_mlp_ensemble"), key=lambda name: metrics[name]["auprc"])
    comparisons = {}
    for left, right in (
        ("vqc_train_cdf", "vqc_raw"),
        ("vqc_train_cdf", strongest),
        ("vqc_train_cdf", "vqc_no_entanglement_cdf"),
        ("fusion_vqc_train_cdf", "fusion_q4_mlp"),
    ):
        key = f"{left}_minus_{right}"
        comparisons[key] = _paired_patient_bootstrap(
            labels,
            patients,
            averaged[left].to_numpy(float),
            averaged[right].to_numpy(float),
            iterations=args.bootstrap_iterations,
            seed=args.seed,
            comparison=key,
        )
    alignment = comparisons["vqc_train_cdf_minus_vqc_raw"]["delta_auprc"]
    quantum = comparisons[f"vqc_train_cdf_minus_{strongest}"]["delta_auprc"]
    entanglement = comparisons[
        "vqc_train_cdf_minus_vqc_no_entanglement_cdf"
    ]["delta_auprc"]
    system = comparisons[
        "fusion_vqc_train_cdf_minus_fusion_q4_mlp"
    ]["delta_auprc"]
    seed_frame = pd.DataFrame(seed_rows)
    verdict = {
        "seeds": sorted(paths),
        "strongest_identical_q4_control": strongest,
        "alignment_gate": bool(
            alignment["mean"] >= 0.003
            and alignment["ci95_low"] > 0
            and (seed_frame.alignment_delta > 0).all()
        ),
        "quantum_gate": bool(quantum["ci95_low"] > 0 and (seed_frame.quantum_delta > 0).all()),
        "entanglement_gate": bool(entanglement["ci95_low"] > 0),
        "system_gate": bool(
            metrics["fusion_vqc_train_cdf"]["auprc"] > 0.8380154031265051
            and system["ci95_low"] > 0
        ),
        "expand_to_five_seeds": False,
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    verdict["expand_to_five_seeds"] = bool(
        verdict["alignment_gate"] and verdict["quantum_gate"] and verdict["system_gate"]
    )
    seed_frame.to_csv(args.output / "seed_metrics.csv", index=False)
    averaged.to_csv(args.output / "mean_oof_predictions.csv", index=False)
    (args.output / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "bootstrap.json").write_text(json.dumps(comparisons, indent=2))
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"seed_metrics": seed_rows, "metrics": metrics, "verdict": verdict}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[42, 31415])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=420)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
