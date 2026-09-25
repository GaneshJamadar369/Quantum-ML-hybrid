from argparse import Namespace

import numpy as np
import pandas as pd

from aggregate_score_alignment_screen import MODELS, run


def _write(root, seed):
    rng = np.random.default_rng(seed)
    n = 80
    labels = np.tile([0, 0, 0, 1], n // 4)
    frame = pd.DataFrame(
        {
            "ecg_id": np.arange(n),
            "patient_id": np.repeat(np.arange(n // 2), 2),
            "strat_fold": np.tile(np.arange(1, 9), n // 8),
            "y_true": labels,
            "hard_negative": np.tile([False, True, False, False], n // 4),
        }
    )
    for index, name in enumerate(MODELS):
        frame[name] = np.clip(
            np.where(labels == 1, 0.75, 0.15) + rng.normal(0, 0.02 + index * 0.001, n),
            0.01,
            0.99,
        )
    target = root / f"seed_{seed}"
    target.mkdir(parents=True)
    frame.to_csv(target / "oof_predictions_and_observables.csv", index=False)


def test_score_alignment_aggregate_contract(tmp_path):
    root = tmp_path / "inputs"
    _write(root, 42)
    _write(root, 31415)
    output = tmp_path / "output"
    run(
        Namespace(
            inputs=[root],
            expected_seeds=[42, 31415],
            output=output,
            bootstrap_iterations=25,
            seed=3,
        )
    )
    assert len(pd.read_csv(output / "seed_metrics.csv")) == 2
    assert len(pd.read_csv(output / "mean_oof_predictions.csv")) == 80
    assert (output / "verdict.json").exists()
