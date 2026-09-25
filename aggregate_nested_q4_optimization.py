"""Aggregate two nested-q4 Kaggle shards without selecting on seed outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_divergence_distillation_screen import _metrics
from run_nested_q4_optimization import _operating_point
from run_quantum_baselines import _paired_patient_bootstrap


MODELS = (
    "vqc", "vqc_no_entanglement", "q4_logistic", "q4_mlp", "q4_rbf",
    "clinical_teacher", "fusion_vqc", "fusion_q4_mlp",
)


def _discover(roots: list[Path]) -> dict[int, Path]:
    found = {}
    for root in roots:
        for path in root.rglob("oof_predictions_and_observables.csv"):
            seed = int(path.parent.name.removeprefix("seed_"))
            if seed in found:
                raise ValueError(f"Duplicate seed {seed}: {found[seed]} and {path}")
            found[seed] = path
    return found


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    paths = _discover(args.inputs)
    expected = set(args.expected_seeds)
    if set(paths) != expected:
        raise ValueError(f"Expected seeds {sorted(expected)}, found {sorted(paths)}")
    frames = {seed: pd.read_csv(path).sort_values("ecg_id").reset_index(drop=True) for seed, path in paths.items()}
    reference = frames[min(frames)]
    identity = ["ecg_id", "patient_id", "strat_fold", "y_true", "hard_negative"]
    for seed, frame in frames.items():
        if not frame[identity].equals(reference[identity]):
            raise ValueError(f"Seed {seed} cohort/order differs")
        if frame[list(MODELS)].isna().any().any():
            raise ValueError(f"Seed {seed} has incomplete predictions")
    y = reference.y_true.to_numpy(int)
    patients = reference.patient_id.to_numpy(int)
    hard = reference.hard_negative.to_numpy(bool)
    seed_rows = []
    for seed, frame in sorted(frames.items()):
        metrics = {name: _metrics(y, frame[name].to_numpy(float)) for name in MODELS}
        strongest = max(("q4_logistic", "q4_mlp", "q4_rbf"), key=lambda name: metrics[name]["auprc"])
        seed_rows.append({
            "seed": seed,
            "vqc_auprc": metrics["vqc"]["auprc"],
            "strongest_control": strongest,
            "strongest_control_auprc": metrics[strongest]["auprc"],
            "delta_vqc_control": metrics["vqc"]["auprc"] - metrics[strongest]["auprc"],
            "delta_vqc_no_entanglement": metrics["vqc"]["auprc"] - metrics["vqc_no_entanglement"]["auprc"],
            "fusion_vqc_auprc": metrics["fusion_vqc"]["auprc"],
            "fusion_classical_auprc": metrics["fusion_q4_mlp"]["auprc"],
            "delta_fusion": metrics["fusion_vqc"]["auprc"] - metrics["fusion_q4_mlp"]["auprc"],
        })
    seed_metrics = pd.DataFrame(seed_rows)
    averaged = reference[identity].copy()
    for name in MODELS:
        averaged[name] = np.mean([frame[name].to_numpy(float) for frame in frames.values()], axis=0)
    ensemble_metrics = {name: _metrics(y, averaged[name]) for name in MODELS}
    strongest = max(("q4_logistic", "q4_mlp", "q4_rbf"), key=lambda name: ensemble_metrics[name]["auprc"])
    comparisons = {}
    for left, right in (("vqc", strongest), ("vqc", "vqc_no_entanglement"), ("fusion_vqc", "fusion_q4_mlp")):
        key = f"{left}_minus_{right}"
        comparisons[key] = _paired_patient_bootstrap(
            y, patients, averaged[left], averaged[right],
            iterations=args.bootstrap_iterations, seed=args.seed, comparison=key,
        )
    operating = {name: _operating_point(y, averaged[name].to_numpy(float), hard) for name in MODELS}
    quantum_delta = comparisons[f"vqc_minus_{strongest}"]["delta_auprc"]
    entanglement_delta = comparisons["vqc_minus_vqc_no_entanglement"]["delta_auprc"]
    fusion_delta = comparisons["fusion_vqc_minus_fusion_q4_mlp"]["delta_auprc"]
    wins = int((seed_metrics.delta_vqc_control > 0).sum())
    verdict = {
        "seeds": sorted(paths),
        "strongest_identical_q4_control": strongest,
        "seed_wins": wins,
        "quantum_gate": bool(
            quantum_delta["mean"] >= 0.005
            and quantum_delta["ci95_low"] > 0.0
            and wins >= 4
        ),
        "entanglement_gate": bool(entanglement_delta["ci95_low"] > 0.0),
        "system_gate": bool(
            ensemble_metrics["fusion_vqc"]["auprc"] > 0.83802
            and fusion_delta["ci95_low"] > 0.0
        ),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
    }
    seed_metrics.to_csv(args.output / "seed_metrics.csv", index=False)
    averaged.to_csv(args.output / "mean_oof_predictions.csv", index=False)
    (args.output / "ensemble_metrics.json").write_text(json.dumps(ensemble_metrics, indent=2))
    (args.output / "bootstrap.json").write_text(json.dumps(comparisons, indent=2))
    (args.output / "operating_points.json").write_text(json.dumps(operating, indent=2))
    (args.output / "verdict.json").write_text(json.dumps(verdict, indent=2))
    print(json.dumps({"seed_metrics": seed_rows, "ensemble_metrics": ensemble_metrics, "verdict": verdict}, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[42, 31415, 27182, 16180, 14142])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=420)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
