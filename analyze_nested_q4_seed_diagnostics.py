"""Post-hoc diagnostics for completed nested-q4 seeds (descriptive only)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


def _effective_rank(values: np.ndarray) -> float:
    singular = np.linalg.svd(values - values.mean(0), compute_uv=False)
    probability = singular**2 / np.sum(singular**2)
    probability = probability[probability > 0]
    return float(np.exp(-np.sum(probability * np.log(probability))))


def _cross_fitted_observable_logistic(frame: pd.DataFrame) -> np.ndarray:
    columns = [column for column in frame if column.startswith("quantum_observable_")]
    prediction = np.full(len(frame), np.nan)
    for held_out in sorted(frame.strat_fold.unique()):
        train = frame.strat_fold.ne(held_out)
        model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        model.fit(frame.loc[train, columns], frame.loc[train, "y_true"])
        prediction[~train] = model.predict_proba(frame.loc[~train, columns])[:, 1]
    return prediction


def run(inputs: list[Path], output: Path):
    rows, selection_rows = [], []
    for root in inputs:
        for path in sorted(root.rglob("oof_predictions_and_observables.csv")):
            seed = int(path.parent.name.removeprefix("seed_"))
            frame = pd.read_csv(path)
            labels = frame.y_true.to_numpy(int)
            q4 = frame[[f"q4_{index}" for index in range(4)]].to_numpy(float)
            observables = frame[[f"quantum_observable_{index}" for index in range(8)]].to_numpy(float)
            observable_probability = _cross_fitted_observable_logistic(frame)
            selections = json.loads((path.parent / "selections.json").read_text())
            counts = Counter(item["selected_candidate"]["candidate"]["name"] for item in selections)
            epochs = [item["selected_candidate"]["best_epoch"] for item in selections]
            inner_scores = [item["selected_candidate"]["best_inner_auprc"] for item in selections]
            audits = json.loads((path.parent / "fold_audits.json").read_text())
            gradient_min = [item["vqc"]["gradient_min"] for item in audits]
            gradient_max = [item["vqc"]["gradient_max"] for item in audits]
            rows.append(
                {
                    "seed": seed,
                    "q4_effective_rank": _effective_rank(q4),
                    "observable_effective_rank": _effective_rank(observables),
                    "vqc_auprc": average_precision_score(labels, frame.vqc),
                    "observable_meta_logistic_auprc_descriptive": average_precision_score(
                        labels, observable_probability
                    ),
                    "q4_logistic_auprc": average_precision_score(labels, frame.q4_logistic),
                    "q4_mlp_auprc": average_precision_score(labels, frame.q4_mlp),
                    "score_spearman_vqc_q4_logistic": spearmanr(frame.vqc, frame.q4_logistic).statistic,
                    "score_spearman_vqc_q4_mlp": spearmanr(frame.vqc, frame.q4_mlp).statistic,
                    "selected_epoch_mean": float(np.mean(epochs)),
                    "selected_epoch_at_cap": int(np.sum(np.asarray(epochs) == 30)),
                    "selected_inner_auprc_mean": float(np.mean(inner_scores)),
                    "gradient_min_across_folds": float(np.min(gradient_min)),
                    "gradient_max_across_folds": float(np.max(gradient_max)),
                    **{f"selected_{name}": counts[name] for name in sorted(counts)},
                }
            )
            for item in selections:
                selection_rows.append(
                    {
                        "seed": seed,
                        "outer_fold": item["outer_fold"],
                        "winner": item["selected_candidate"]["candidate"]["name"],
                        "epoch": item["selected_candidate"]["best_epoch"],
                        "inner_auprc": item["selected_candidate"]["best_inner_auprc"],
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("seed").to_csv(output / "seed_diagnostics.csv", index=False)
    pd.DataFrame(selection_rows).sort_values(["seed", "outer_fold"]).to_csv(
        output / "selection_diagnostics.csv", index=False
    )
    print(pd.DataFrame(rows).sort_values("seed").to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.inputs, arguments.output)
