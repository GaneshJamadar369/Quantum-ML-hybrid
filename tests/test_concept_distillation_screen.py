import numpy as np

from run_concept_rationale_distillation_screen import _oof_q4_assistant


def test_q4_assistant_is_inner_fold_oof_and_finite():
    rng = np.random.default_rng(17)
    train_q = rng.normal(size=(72, 4)).astype(np.float32)
    train_y = np.tile([0, 1], 36)
    train_folds = np.repeat([1, 2, 3], 24)
    validation_q = rng.normal(size=(11, 4)).astype(np.float32)
    train_probability, validation_probability, reliability, audit = _oof_q4_assistant(
        train_q, train_y, train_folds, validation_q, seed=23
    )
    assert train_probability.shape == (72,)
    assert validation_probability.shape == (11,)
    assert reliability.shape == (72,)
    assert np.isfinite(train_probability).all()
    assert np.isfinite(validation_probability).all()
    assert set(audit["inner_folds"]) == {1, 2, 3}
    assert 0.0 <= reliability.min() <= reliability.max() <= 1.0
