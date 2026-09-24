import numpy as np
import torch

from aquire_preprocessing.distillation import FROZEN_DIVERGENCE_SCREEN
from run_divergence_distillation_screen import (
    _oof_calibrated_teacher,
    _train_candidate,
)


def test_oof_teacher_and_reverse_kl_vqc_smoke():
    rng = np.random.default_rng(17)
    labels = np.tile(np.array([0, 1]), 18)
    folds = np.repeat(np.array([1, 2, 3]), 12)
    features = rng.normal(size=(36, 6)).astype(np.float32)
    features[:, 0] += 0.8 * labels
    validation = rng.normal(size=(8, 6)).astype(np.float32)
    train_probability, val_probability, audit = _oof_calibrated_teacher(
        features, labels, folds, validation, seed=9
    )
    assert train_probability.shape == (36,)
    assert val_probability.shape == (8,)
    assert np.isfinite(train_probability).all()
    assert np.isfinite(val_probability).all()
    assert audit["inner_folds"] == [1, 2, 3]

    q_train = rng.uniform(-1.2, 1.2, size=(36, 4)).astype(np.float32)
    q_val = rng.uniform(-1.2, 1.2, size=(8, 4)).astype(np.float32)
    reverse = next(spec for spec in FROZEN_DIVERGENCE_SCREEN if spec.name == "reverse_t2")
    logits, training_audit = _train_candidate(
        q_train, labels, train_probability, q_val, reverse,
        epochs=1, batch_size=12, seed=21, device=torch.device("cpu"),
    )
    assert logits.shape == (8,)
    assert np.isfinite(logits).all()
    assert training_audit["parameters"] == 45
    assert np.isfinite(training_audit["final_loss"])
