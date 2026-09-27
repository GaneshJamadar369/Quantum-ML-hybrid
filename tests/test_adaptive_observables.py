import pytest
import torch

from aquire_preprocessing.adaptive_observables import (
    AdaptiveObservableQuantumClassifier,
    ParameterMatchedQ4MLP,
)


@pytest.mark.parametrize("locality,outputs", [(2, 6), (3, 4)])
def test_adaptive_observable_shapes_hermiticity_and_gradients(locality, outputs):
    model = AdaptiveObservableQuantumClassifier(locality)
    x = torch.randn(5, 4, requires_grad=True)
    observables = model.quantum_observables(x)
    logits = model(x)
    assert observables.shape == (5, outputs)
    assert logits.shape == (5,)
    assert model.operator_bank.shape == (outputs, 4**locality, 16, 16)
    assert torch.allclose(
        model.operator_bank,
        model.operator_bank.conj().transpose(-1, -2),
    )
    assert torch.allclose(
        torch.linalg.vector_norm(model.pauli_weights(), dim=1),
        torch.ones(outputs),
    )
    logits.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.observable_coefficients.grad is not None
    assert torch.isfinite(model.observable_coefficients.grad).all()


def test_fixed_pauli_ablation_freezes_only_observable_coefficients():
    model = AdaptiveObservableQuantumClassifier(2, trainable_observables=False)
    assert not model.observable_coefficients.requires_grad
    assert model.rotations.requires_grad and model.interactions.requires_grad
    assert torch.allclose(model.pauli_weights()[:, -1], torch.ones(6))
    assert torch.count_nonzero(model.pauli_weights()[:, :-1]) == 0


@pytest.mark.parametrize("locality", [2, 3])
def test_parameter_matched_mlp_is_within_one_percent(locality):
    quantum = AdaptiveObservableQuantumClassifier(locality)
    target = sum(value.numel() for value in quantum.parameters() if value.requires_grad)
    mlp = ParameterMatchedQ4MLP(target)
    difference = abs(mlp.parameter_difference)
    assert difference / target < 0.01
    output = mlp(torch.randn(8, 4))
    assert output.shape == (8,) and torch.isfinite(output).all()


def test_adaptive_observable_rejects_unsupported_locality_and_shape():
    with pytest.raises(ValueError):
        AdaptiveObservableQuantumClassifier(4)
    model = AdaptiveObservableQuantumClassifier(2)
    with pytest.raises(ValueError):
        model(torch.randn(3, 8))
