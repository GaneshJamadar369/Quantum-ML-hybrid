import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("pennylane")

from aquire_preprocessing.models_quantum import (
    HybridQuantumNeuralNetwork,
    ProjectedIQPFeatureMap,
    QSVMClassifier,
    QuantumKernelEstimator,
    VariationalQuantumClassifier,
)
from run_quantum_baselines import _paired_patient_bootstrap


def test_iqp_kernel_is_psd_and_not_old_separable_angle_kernel():
    rng = np.random.default_rng(7)
    x = rng.normal(0.0, 0.5, size=(8, 4))
    estimator = QuantumKernelEstimator(n_qubits=4, n_layers=2)
    kernel = estimator.compute_kernel_matrix(x)
    diagnostics = estimator.diagnostics(kernel)
    assert np.allclose(kernel, kernel.T, atol=1e-10)
    assert np.allclose(np.diag(kernel), 1.0, atol=1e-10)
    assert diagnostics["negative_eigenvalue_count"] == 0

    # This was the exact functional form of the former AngleEmbedding plus
    # data-independent CNOT-ring kernel.  IQP data-dependent interactions must
    # not collapse to it.
    delta = x[:, None, :] - x[None, :, :]
    old_product_kernel = np.prod(np.cos(delta / 2.0) ** 2, axis=-1)
    assert not np.allclose(kernel, old_product_kernel, atol=1e-6)


def test_qsvm_end_to_end_probability_contract():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(32, 4))
    y = (x[:, 0] * x[:, 1] > 0).astype(int)
    model = QSVMClassifier(
        n_qubits=4, n_layers=2, calibration_splits=2, seed=11
    ).fit(x, y)
    probability = model.predict_proba(x[:5])
    assert probability.shape == (5, 2)
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert model.kernel_diagnostics_["negative_eigenvalue_count"] == 0


def test_paired_patient_bootstrap_detects_better_scores():
    labels = np.tile([0, 1], 20)
    patients = np.repeat(np.arange(20), 2)
    quantum = labels * 0.8 + (1 - labels) * 0.2
    classical = np.full(len(labels), 0.5)
    report = _paired_patient_bootstrap(
        labels, patients, quantum, classical, iterations=100, seed=5
    )
    assert report["delta_auprc"]["ci95_low"] > 0
    assert report["matched_kernel_accuracy_gate"] == "PASS_MATCHED_KERNEL_ACCURACY_DELTA"
    assert "does not establish computational quantum advantage" in report["claim_boundary"]


def test_angle_controls_are_finite_and_product_cosine_is_psd():
    from run_quantum_baselines import _classical_kernel, _quantum_angle_coordinates

    rng = np.random.default_rng(17)
    train = rng.normal(size=(20, 8))
    val = rng.normal(size=(5, 8))
    angle_train, angle_val = _quantum_angle_coordinates(train, val)
    assert angle_train.shape == train.shape
    assert angle_val.shape == val.shape
    assert np.isfinite(angle_train).all()
    kernel = _classical_kernel("product_cosine")(angle_train, angle_train)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.linalg.eigvalsh(kernel).min() > -1e-8


def test_vqc_batched_backward_pass():
    import torch

    model = VariationalQuantumClassifier(in_features=8, n_qubits=4, n_layers=1)
    features = torch.randn(2, 8)
    output = model(features)
    output.square().mean().backward()
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    assert model.encoder[1].weight.grad is not None
    assert model.qnode.weights.grad is not None


def test_hqnn_batched_backward_pass():
    import torch

    model = HybridQuantumNeuralNetwork(
        tabular_dim=8,
        raw_channels=12,
        n_qubits=4,
        n_quantum_layers=1,
        resnet_base_filters=4,
    )
    signal = torch.randn(2, 12, 128)
    tabular = torch.randn(2, 8)
    output = model(signal, tabular)
    output.square().mean().backward()
    assert output.shape == (2,)
    assert torch.isfinite(output).all()
    assert model.waveform_stem[0].weight.grad is not None
    assert model.tabular_encoder[1].weight.grad is not None
    assert model.quantum_layer.weights.grad is not None


def test_projected_iqp_features_and_kernel_are_finite_psd():
    rng = np.random.default_rng(23)
    x = rng.uniform(-1.0, 1.0, size=(12, 4))
    feature_map = ProjectedIQPFeatureMap(
        n_qubits=4,
        n_layers=2,
        feature_scale=0.5,
        interaction_scale=0.25,
        topology="ring",
        mixing_seed=23,
    )
    projected = feature_map.transform(x)
    kernel, gamma = feature_map.rbf_kernel(projected)
    assert projected.shape == (12, 16)
    assert np.isfinite(projected).all()
    assert gamma > 0
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), 1.0)
    assert np.linalg.eigvalsh(kernel).min() > -1e-8
