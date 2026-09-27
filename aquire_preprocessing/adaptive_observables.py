"""Four-qubit VQC with trainable k-local Hermitian measurements.

The circuit is kept identical to the retained q4 statevector VQC.  Only the
measurement family changes from fixed local Z/ring-ZZ expectations to a
Pauli-basis parameterization of an arbitrary Hermitian observable on every
k-qubit subset.
"""

from __future__ import annotations

from itertools import combinations, product

import torch
from torch import nn

from .models_quantum import TorchStatevectorQuantumClassifier


def _pauli_matrices(dtype: torch.dtype = torch.complex64) -> tuple[torch.Tensor, ...]:
    identity = torch.eye(2, dtype=dtype)
    x = torch.tensor([[0, 1], [1, 0]], dtype=dtype)
    y = torch.tensor([[0, -1j], [1j, 0]], dtype=dtype)
    z = torch.tensor([[1, 0], [0, -1]], dtype=dtype)
    return identity, x, y, z


def _operator_bank(n_qubits: int, locality: int) -> tuple[tuple[tuple[int, ...], ...], torch.Tensor]:
    subsets = tuple(combinations(range(n_qubits), locality))
    paulis = _pauli_matrices()
    operators = []
    for subset in subsets:
        subset_operators = []
        for local_indices in product(range(4), repeat=locality):
            assignment = dict(zip(subset, local_indices))
            operator = torch.ones(1, 1, dtype=torch.complex64)
            for wire in range(n_qubits):
                operator = torch.kron(operator, paulis[assignment.get(wire, 0)])
            subset_operators.append(operator)
        operators.append(torch.stack(subset_operators, dim=0))
    return subsets, torch.stack(operators, dim=0)


class AdaptiveObservableQuantumClassifier(TorchStatevectorQuantumClassifier):
    """Retained q4 circuit followed by normalized trainable k-local observables."""

    def __init__(
        self,
        locality: int,
        *,
        trainable_observables: bool = True,
        initial_bandwidth: float = 0.5,
    ):
        if locality not in (2, 3):
            raise ValueError("The frozen q4 ANO screen supports locality 2 or 3")
        if not 0.0 < initial_bandwidth < 2.0:
            raise ValueError("initial bandwidth must lie in (0, 2)")
        super().__init__(n_qubits=4, n_layers=2, topology="ring", input_dim=4)
        self.locality = int(locality)
        self.trainable_observables = bool(trainable_observables)
        subsets, bank = _operator_bank(self.n_qubits, self.locality)
        self.subsets = subsets
        self.register_buffer("operator_bank", bank)
        coefficients = torch.zeros(len(subsets), 4**self.locality)
        # Start from the matching fixed Z...Z string for each subset. All
        # remaining Pauli coefficients are available to the adaptive arm.
        coefficients[:, -1] = 1.0
        self.observable_coefficients = nn.Parameter(
            coefficients, requires_grad=trainable_observables
        )
        # Parent uses multiplier 2*sigmoid(parameter).
        initial = torch.logit(torch.tensor(initial_bandwidth / 2.0)).item()
        with torch.no_grad():
            self.feature_scales.fill_(initial)
        self.readout = nn.Linear(len(subsets), 1)

    def statevector(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 4 or not torch.isfinite(x).all():
            raise ValueError(f"Expected finite q4 input (batch, 4), got {tuple(x.shape)}")
        state = torch.zeros(
            len(x),
            1 << self.n_qubits,
            dtype=self._complex_dtype(x.dtype),
            device=x.device,
        )
        state[:, 0] = 1.0
        scaled = x * (2.0 * torch.sigmoid(self.feature_scales))
        for layer in range(self.n_layers):
            for wire in range(self.n_qubits):
                state = self._apply_ry(state, scaled[:, wire], wire)
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
        return state

    def pauli_weights(self) -> torch.Tensor:
        norm = torch.linalg.vector_norm(
            self.observable_coefficients, dim=1, keepdim=True
        ).clamp_min(1e-8)
        return self.observable_coefficients / norm

    def quantum_observables(self, x: torch.Tensor) -> torch.Tensor:
        state = self.statevector(x)
        bank = self.operator_bank.to(state.dtype)
        pauli_expectations = torch.einsum(
            "bi,ctij,bj->bct", state.conj(), bank, state
        ).real
        return torch.einsum("bct,ct->bc", pauli_expectations, self.pauli_weights())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.quantum_observables(x)).squeeze(-1)

    def scale_values(self) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.feature_scales)


class ParameterMatchedQ4MLP(nn.Module):
    """One-hidden-layer tanh control matched to the ANO trainable parameter count."""

    def __init__(self, target_parameters: int):
        super().__init__()
        if target_parameters < 7:
            raise ValueError("target parameter count is too small")
        # A 4 -> h -> 1 MLP has exactly 6h+1 trainable parameters.
        hidden = max(1, round((target_parameters - 1) / 6))
        self.network = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.target_parameters = int(target_parameters)

    @property
    def parameter_difference(self) -> int:
        actual = sum(value.numel() for value in self.parameters())
        return int(actual - self.target_parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 4 or not torch.isfinite(x).all():
            raise ValueError(f"Expected finite q4 input (batch, 4), got {tuple(x.shape)}")
        return self.network(x).squeeze(-1)
