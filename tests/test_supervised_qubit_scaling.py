import numpy as np
import pytest

torch = pytest.importorskip("torch")
qml = pytest.importorskip("pennylane")

from aquire_preprocessing.models_quantum import TorchStatevectorQuantumClassifier
from run_supervised_qubit_scaling_screen import _matched_mlp, _metric_row


def test_torch_statevector_matches_pennylane_and_backpropagates():
    torch.manual_seed(19)
    model = TorchStatevectorQuantumClassifier(n_qubits=3, n_layers=2).double()
    inputs = torch.tensor(
        [[0.2, -0.4, 0.7], [-0.8, 0.1, 0.3]], dtype=torch.float64,
        requires_grad=True,
    )
    observed = model.quantum_observables(inputs)

    device = qml.device("default.qubit", wires=3)
    rotations = model.rotations.detach().numpy()
    interactions = model.interactions.detach().numpy()
    scales = model.feature_scales.detach().numpy()

    @qml.qnode(device)
    def reference(values):
        scaled = values * (2.0 / (1.0 + np.exp(-scales)))
        for layer in range(2):
            qml.AngleEmbedding(scaled, wires=range(3), rotation="Y")
            for wire in range(3):
                qml.Rot(*rotations[layer, wire], wires=wire)
            for edge_index, (left, right) in enumerate(model.edges):
                qml.IsingZZ(interactions[layer, edge_index], wires=[left, right])
        result = [qml.expval(qml.PauliZ(wire)) for wire in range(3)]
        result.extend(
            qml.expval(qml.PauliZ(left) @ qml.PauliZ(right))
            for left, right in model.edges
        )
        return result

    expected = np.asarray([reference(row) for row in inputs.detach().numpy()])
    assert np.allclose(observed.detach().numpy(), expected, atol=1e-10)
    model(inputs).square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_scalable_statevector_contract_and_parameter_matched_control():
    model = TorchStatevectorQuantumClassifier(n_qubits=8, n_layers=2)
    x = torch.empty(4, 8).uniform_(-np.pi, np.pi)
    observables = model.quantum_observables(x)
    output = model(x)
    assert observables.shape == (4, 16)
    assert output.shape == (4,)
    assert torch.isfinite(observables).all() and torch.isfinite(output).all()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    mlp, hidden = _matched_mlp(8, parameters)
    mlp_parameters = sum(parameter.numel() for parameter in mlp.parameters())
    assert hidden > 0
    assert abs(mlp_parameters - parameters) <= 8 + 2


def test_q4_replication_drives_all_sixteen_qubits_and_backpropagates():
    model = TorchStatevectorQuantumClassifier(
        n_qubits=16, input_dim=4, n_layers=1, topology="ring"
    )
    assert model.wire_feature_indices.tolist() == [0, 1, 2, 3] * 4
    inputs = torch.randn(2, 4, requires_grad=True)
    observables = model.quantum_observables(inputs)
    output = model(inputs)
    assert observables.shape == (2, 32) and output.shape == (2,)
    output.square().mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert model.feature_scales.grad is not None
    assert model.feature_scales.grad.shape == (16,)
    assert torch.isfinite(model.feature_scales.grad).all()


def test_metric_row_reports_hard_cohort_and_sensitivity():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    hard = np.array([True, False, True, False, False, False, False, False])
    row = _metric_row("q8_vqc", labels, scores, hard, 8)
    assert row["qubits"] == 8 and row["input_dim"] == 8
    assert row["auprc"] == pytest.approx(1.0)
    assert row["hard_negative_auroc"] == pytest.approx(1.0)
    assert row["sensitivity_at_90_specificity"] == pytest.approx(1.0)
