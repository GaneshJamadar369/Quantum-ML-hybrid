"""Controlled q4 circuit variants motivated by recent NeurIPS research.

The classes in this module deliberately keep the existing fold-local PLS-q4
representation and linear observable readout.  They change only the quantum
head, which permits paired comparison with the retained q4 VQC.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .models_quantum import TorchStatevectorQuantumClassifier


def _bounded_scale(parameter: torch.Tensor) -> torch.Tensor:
    """Map unconstrained parameters to the frozen [0.25, 2.0] bandwidth range."""

    return 0.25 + 1.75 * torch.sigmoid(parameter)


def _scale_initial_value(value: float) -> float:
    if not 0.25 < value < 2.0:
        raise ValueError("initial scale must lie strictly inside [0.25, 2.0]")
    probability = (value - 0.25) / 1.75
    return math.log(probability / (1.0 - probability))


class LayerwiseFrequencyQuantumClassifier(TorchStatevectorQuantumClassifier):
    """q4 VQC with a separate input frequency and phase in each upload layer.

    The retained VQC learns one scale for each q4 coordinate and reuses that
    scale in both layers.  This arm learns a bounded scale and phase per
    coordinate *and* layer, a supervised-classification adaptation of the
    trainable input-scaling idea used in parametrized quantum policies.
    """

    def __init__(self, initial_scale: float = 0.5):
        super().__init__(n_qubits=4, n_layers=2, topology="ring", input_dim=4)
        initial = _scale_initial_value(initial_scale)
        self.feature_scales = nn.Parameter(torch.full((2, 4), initial))
        self.feature_phases = nn.Parameter(torch.zeros(2, 4))

    def quantum_observables(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 4 or not torch.isfinite(x).all():
            raise ValueError(f"Expected finite q4 input (batch, 4), got {tuple(x.shape)}")
        state = torch.zeros(
            len(x), 1 << self.n_qubits,
            dtype=self._complex_dtype(x.dtype),
            device=x.device,
        )
        state[:, 0] = 1.0
        scales = _bounded_scale(self.feature_scales)
        for layer in range(2):
            angles = x * scales[layer] + self.feature_phases[layer]
            for wire in range(4):
                state = self._apply_ry(state, angles[:, wire], wire)
                state = self._apply_rz(state, self.rotations[layer, wire, 0], wire)
                state = self._apply_ry(state, self.rotations[layer, wire, 1], wire)
                state = self._apply_rz(state, self.rotations[layer, wire, 2], wire)
            for edge_index in range(len(self.edges)):
                phase = torch.exp(
                    (
                        -0.5j
                        * self.interactions[layer, edge_index]
                        * self.edge_signs[edge_index]
                    ).to(state.dtype)
                )
                state = state * phase
        probabilities = state.real.square() + state.imag.square()
        return probabilities @ self.observable_signs.T.to(probabilities.dtype)

    def scale_values(self) -> torch.Tensor:
        return _bounded_scale(self.feature_scales)


class TiedEquilibriumQuantumClassifier(TorchStatevectorQuantumClassifier):
    """Finite, tied recurrent quantum cell inspired by quantum DEQs.

    Each refinement starts a fresh four-qubit circuit.  The first upload is
    the fixed q4 input and the second upload is the previous four local-Z
    expectations.  The same circuit parameters are reused at every step.
    This is a transparent finite equilibrium screen, not a claim of free
    infinite physical depth: every refinement is another circuit execution.
    """

    def __init__(
        self,
        steps: int = 3,
        damping: float = 0.5,
        initial_scale: float = 0.5,
    ):
        if steps < 2:
            raise ValueError("equilibrium screen requires at least two tied steps")
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        super().__init__(n_qubits=4, n_layers=2, topology="ring", input_dim=4)
        self.steps = int(steps)
        self.damping = float(damping)
        initial = _scale_initial_value(initial_scale)
        # Row 0 scales the immutable q4 input; row 1 scales recurrent memory.
        self.feature_scales = nn.Parameter(torch.full((2, 4), initial))

    def _cell(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        state = torch.zeros(
            len(x), 1 << self.n_qubits,
            dtype=self._complex_dtype(x.dtype),
            device=x.device,
        )
        state[:, 0] = 1.0
        scales = _bounded_scale(self.feature_scales)
        for layer, values in enumerate((x, memory)):
            angles = values * scales[layer]
            for wire in range(4):
                state = self._apply_ry(state, angles[:, wire], wire)
                state = self._apply_rz(state, self.rotations[layer, wire, 0], wire)
                state = self._apply_ry(state, self.rotations[layer, wire, 1], wire)
                state = self._apply_rz(state, self.rotations[layer, wire, 2], wire)
            for edge_index in range(len(self.edges)):
                phase = torch.exp(
                    (
                        -0.5j
                        * self.interactions[layer, edge_index]
                        * self.edge_signs[edge_index]
                    ).to(state.dtype)
                )
                state = state * phase
        probabilities = state.real.square() + state.imag.square()
        return probabilities @ self.observable_signs.T.to(probabilities.dtype)

    def quantum_observables(
        self, x: torch.Tensor, *, return_trace: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[1] != 4 or not torch.isfinite(x).all():
            raise ValueError(f"Expected finite q4 input (batch, 4), got {tuple(x.shape)}")
        memory = torch.zeros_like(x)
        changes = []
        observables = None
        for _ in range(self.steps):
            observables = self._cell(x, memory)
            proposal = observables[:, :4]
            updated = (1.0 - self.damping) * memory + self.damping * proposal
            changes.append(torch.linalg.vector_norm(updated - memory, dim=1))
            memory = updated
        assert observables is not None
        if return_trace:
            return observables, torch.stack(changes, dim=1)
        return observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.quantum_observables(x)).squeeze(-1)

    def convergence_residual(self, x: torch.Tensor) -> torch.Tensor:
        _, changes = self.quantum_observables(x, return_trace=True)
        return changes[:, -1]

    def scale_values(self) -> torch.Tensor:
        return _bounded_scale(self.feature_scales)
