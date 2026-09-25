from argparse import Namespace

import numpy as np
import pandas as pd

from aggregate_nested_q4_optimization import MODELS, _discover, run


SEEDS = [42, 31415, 27182, 16180, 14142]


def _write_seed(root, seed, *, reverse=False):
    n = 80
    y = np.tile([0, 0, 0, 1], n // 4)
    base = np.where(y == 1, 0.76, 0.18).astype(float)
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "ecg_id": np.arange(n),
            "patient_id": np.repeat(np.arange(n // 2), 2),
            "strat_fold": np.tile(np.arange(1, 9), n // 8),
            "y_true": y,
            "hard_negative": np.tile([False, True, False, False], n // 4),
        }
    )
    for index, name in enumerate(MODELS):
        noise = rng.normal(0.0, 0.025 + index * 0.001, n)
        frame[name] = np.clip(base + noise, 0.01, 0.99)
    if reverse:
        frame = frame.iloc[::-1]
    output = root / f"seed_{seed}"
    output.mkdir(parents=True)
    frame.to_csv(output / "oof_predictions_and_observables.csv", index=False)


def test_five_seed_aggregation_contract(tmp_path):
    shard_a, shard_b = tmp_path / "a", tmp_path / "b"
    for seed in SEEDS[:3]:
        _write_seed(shard_a, seed, reverse=seed == SEEDS[1])
    for seed in SEEDS[3:]:
        _write_seed(shard_b, seed)

    assert set(_discover([shard_a, shard_b])) == set(SEEDS)
    output = tmp_path / "aggregate"
    run(
        Namespace(
            inputs=[shard_a, shard_b],
            expected_seeds=SEEDS,
            output=output,
            bootstrap_iterations=25,
            seed=7,
        )
    )
    assert len(pd.read_csv(output / "seed_metrics.csv")) == 5
    assert len(pd.read_csv(output / "mean_oof_predictions.csv")) == 80
    assert (output / "verdict.json").exists()
