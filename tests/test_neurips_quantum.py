import pytest
import torch

from aquire_preprocessing.neurips_quantum import (
    LayerwiseFrequencyQuantumClassifier,
    TiedEquilibriumQuantumClassifier,
)


@pytest.mark.parametrize(
    "model",
    [
        LayerwiseFrequencyQuantumClassifier(),
        TiedEquilibriumQuantumClassifier(steps=3, damping=0.5),
    ],
)
def test_neurips_q4_variants_have_finite_gradients_and_restricted_readout(model):
    x = torch.randn(7, 4, requires_grad=True)
    output = model(x)
    observables = model.quantum_observables(x)
    assert output.shape == (7,)
    assert observables.shape == (7, 8)
    assert torch.isfinite(output).all() and torch.isfinite(observables).all()
    assert model.readout.in_features == 8 and model.readout.out_features == 1
    output.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(torch.isfinite(value.grad).all() for value in model.parameters() if value.grad is not None)


def test_layerwise_frequency_scaling_is_bounded_and_not_shared_between_layers():
    model = LayerwiseFrequencyQuantumClassifier(initial_scale=0.5)
    assert model.feature_scales.shape == (2, 4)
    assert torch.allclose(model.scale_values(), torch.full((2, 4), 0.5), atol=1e-6)
    with torch.no_grad():
        model.feature_scales[1].add_(1.0)
    assert not torch.allclose(model.scale_values()[0], model.scale_values()[1])
    assert torch.all((model.scale_values() >= 0.25) & (model.scale_values() <= 2.0))


def test_equilibrium_cell_reuses_parameters_and_exposes_convergence_residual():
    model = TiedEquilibriumQuantumClassifier(steps=4, damping=0.5)
    x = torch.randn(5, 4)
    observables, trace = model.quantum_observables(x, return_trace=True)
    assert observables.shape == (5, 8)
    assert trace.shape == (5, 4)
    assert torch.all(trace >= 0)
    assert torch.allclose(model.convergence_residual(x), trace[:, -1])
    # Parameter count is independent of the number of tied refinement steps.
    deeper = TiedEquilibriumQuantumClassifier(steps=6, damping=0.5)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in deeper.parameters())


def test_neurips_q4_variants_reject_wrong_shapes():
    for model in (LayerwiseFrequencyQuantumClassifier(), TiedEquilibriumQuantumClassifier()):
        with pytest.raises(ValueError):
            model(torch.randn(3, 8))
