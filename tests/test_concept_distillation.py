import numpy as np
import pytest

from aquire_preprocessing.concept_distillation import (
    CONCEPT_COLUMNS,
    FROZEN_CONCEPT_SCREEN,
    FoldLocalConceptTransform,
    assistant_reliability_weights,
    bernoulli_js_per_record,
    masked_concept_loss,
)


def test_frozen_screen_has_equal_epoch_budget_and_hard_control():
    assert FROZEN_CONCEPT_SCREEN[0].name == "hard"
    assert all(spec.stage1_epochs + spec.stage2_epochs == 30 for spec in FROZEN_CONCEPT_SCREEN)
    assert len({spec.name for spec in FROZEN_CONCEPT_SCREEN}) == len(FROZEN_CONCEPT_SCREEN)


def test_concept_transform_preserves_missing_mask_and_uses_train_statistics():
    train = np.arange(40, dtype=float).reshape(4, len(CONCEPT_COLUMNS))
    train[0, 2] = np.nan
    validation = np.full((2, len(CONCEPT_COLUMNS)), 1000.0)
    validation[1, 4] = np.nan
    transform = FoldLocalConceptTransform()
    train_scaled, train_mask = transform.fit_transform(train)
    validation_scaled, validation_mask = transform.transform(validation)
    assert train_scaled.shape == train.shape
    assert not bool(train_mask[0, 2])
    assert not bool(validation_mask[1, 4])
    assert np.isfinite(train_scaled).all() and np.isfinite(validation_scaled).all()
    assert np.max(np.abs(validation_scaled)) <= 1.0
    assert np.max(transform.center_) < 1000.0


def test_reliability_penalizes_confident_miscalibration():
    probability = np.array([0.95, 0.95, 0.05, 0.05, 0.55, 0.45])
    calibrated_labels = np.array([1, 1, 0, 0, 1, 0])
    wrong_labels = 1 - calibrated_labels
    calibrated = assistant_reliability_weights(probability, calibrated_labels, bins=4)
    wrong = assistant_reliability_weights(probability, wrong_labels, bins=4)
    assert calibrated.mean() > wrong.mean()
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_losses_are_finite_and_differentiable():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([-0.4, 0.2, 1.1], requires_grad=True)
    teacher = torch.tensor([0.1, 0.6, 0.9])
    js = bernoulli_js_per_record(logits, teacher)
    prediction = torch.zeros((3, len(CONCEPT_COLUMNS)), requires_grad=True)
    target = torch.ones_like(prediction) * 0.25
    observed = torch.ones_like(prediction)
    loss = js.mean() + masked_concept_loss(prediction, target, observed)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_masked_concept_loss_rejects_shape_mismatch():
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError):
        masked_concept_loss(torch.zeros(2, 3), torch.zeros(2, 2), torch.zeros(2, 2))
