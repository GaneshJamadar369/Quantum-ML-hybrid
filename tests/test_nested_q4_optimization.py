import numpy as np
import torch

from run_nested_q4_optimization import (
    CANDIDATES,
    _combined_loss,
    _model,
    _ranking_loss,
)


def test_search_candidates_are_bounded_and_unique():
    assert len(CANDIDATES) == 6
    assert len({value.name for value in CANDIDATES}) == len(CANDIDATES)
    assert all(value.js_weight + value.ranking_weight < 1 for value in CANDIDATES)


def test_bandwidth_initialization_matches_frozen_candidate():
    candidate = next(value for value in CANDIDATES if value.name == "narrow_js")
    model = _model(candidate, torch.device("cpu"))
    multiplier = 2.0 * torch.sigmoid(model.feature_scales)
    assert torch.allclose(multiplier, torch.full((4,), candidate.bandwidth))


def test_pairwise_ranking_prefers_correct_order_and_combined_loss_backpropagates():
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
    correct = torch.tensor([2.0, 1.0, -1.0, -2.0])
    reversed_score = -correct
    assert _ranking_loss(correct, labels) < _ranking_loss(reversed_score, labels)
    logits = torch.tensor([0.2, -0.3, 0.1, -0.2], requires_grad=True)
    teacher = torch.tensor([0.8, 0.7, 0.3, 0.2])
    candidate = next(value for value in CANDIDATES if value.name == "js_rank15")
    loss = _combined_loss(logits, labels, teacher, candidate)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
