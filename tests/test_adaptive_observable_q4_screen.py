import pytest

from aquire_preprocessing.adaptive_observables import (
    AdaptiveObservableQuantumClassifier,
    ParameterMatchedQ4MLP,
)
from run_adaptive_observable_q4_screen import _model


def test_screen_factory_builds_isolated_measurement_and_classical_arms():
    adaptive = _model(
        "adaptive", locality=2, matched_parameters=None, device="cpu"
    )
    target = sum(value.numel() for value in adaptive.parameters() if value.requires_grad)
    fixed = _model(
        "fixed_pauli", locality=2, matched_parameters=None, device="cpu"
    )
    matched = _model(
        "matched_mlp", locality=2, matched_parameters=target, device="cpu"
    )
    assert isinstance(adaptive, AdaptiveObservableQuantumClassifier)
    assert adaptive.observable_coefficients.requires_grad
    assert isinstance(fixed, AdaptiveObservableQuantumClassifier)
    assert not fixed.observable_coefficients.requires_grad
    assert isinstance(matched, ParameterMatchedQ4MLP)
    assert abs(matched.parameter_difference) / target < 0.01


def test_screen_factory_rejects_unknown_arm_and_missing_match_budget():
    with pytest.raises(ValueError, match="Unknown arm"):
        _model("other", locality=2, matched_parameters=None, device="cpu")
    with pytest.raises(ValueError, match="matched_parameters"):
        _model("matched_mlp", locality=2, matched_parameters=None, device="cpu")
