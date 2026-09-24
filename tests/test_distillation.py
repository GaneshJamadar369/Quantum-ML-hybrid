import pytest
import torch

from aquire_preprocessing.distillation import (
    DistillationSpec,
    FROZEN_DIVERGENCE_SCREEN,
    distillation_loss,
)


def test_frozen_screen_names_are_unique_and_contains_controls():
    names = [spec.name for spec in FROZEN_DIVERGENCE_SCREEN]
    assert len(names) == len(set(names))
    assert {"hard", "forward_t2", "mixed_rkl25_t2", "js_t2", "reverse_t2"} <= set(names)


def test_hard_control_matches_binary_cross_entropy():
    logits = torch.tensor([-1.0, 0.2, 1.4], requires_grad=True)
    labels = torch.tensor([0.0, 1.0, 1.0])
    teacher = torch.tensor([0.2, 0.8, 0.9])
    spec = DistillationSpec("hard", "hard", alpha=0.0, temperature=1.0)
    actual = distillation_loss(logits, labels, teacher, spec)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize("spec", FROZEN_DIVERGENCE_SCREEN[1:])
def test_all_distillation_losses_are_finite_and_differentiable(spec):
    logits = torch.tensor([-2.0, -0.2, 0.3, 2.1], requires_grad=True)
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    teacher = torch.tensor([0.05, 0.35, 0.7, 0.95])
    loss = distillation_loss(logits, labels, teacher, spec)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_forward_reverse_and_js_vanish_when_distributions_match_without_hard_term():
    probability = torch.tensor([0.1, 0.35, 0.8])
    logits = torch.logit(probability).requires_grad_()
    labels = torch.tensor([0.0, 0.0, 1.0])
    for kind, reverse in (("bidirectional_kl", 0.0), ("bidirectional_kl", 1.0), ("js", 0.0)):
        spec = DistillationSpec("soft", kind, alpha=1.0, temperature=2.0, reverse_weight=reverse)
        loss = distillation_loss(logits, labels, probability, spec)
        assert float(loss.detach()) == pytest.approx(0.0, abs=2e-6)


def test_invalid_teacher_probability_is_rejected():
    with pytest.raises(ValueError):
        distillation_loss(
            torch.zeros(2), torch.zeros(2), torch.tensor([0.5, 1.2]),
            FROZEN_DIVERGENCE_SCREEN[1],
        )
