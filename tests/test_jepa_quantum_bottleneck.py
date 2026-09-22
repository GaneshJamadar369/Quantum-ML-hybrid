import numpy as np
import pytest

torch = pytest.importorskip("torch")

from run_jepa_quantum_bottleneck_screen import embedding_to_pca_angles


def test_label_free_pca_angles_are_finite_and_train_local():
    rng = np.random.default_rng(8)
    train = rng.normal(size=(40, 16)).astype(np.float32)
    val = rng.normal(size=(11, 16)).astype(np.float32)
    fit = np.arange(30)
    q_train, q_val, audit = embedding_to_pca_angles(train, fit, val, 8, 31)
    assert q_train.shape == (30, 8) and q_val.shape == (11, 8)
    assert np.isfinite(q_train).all() and np.isfinite(q_val).all()
    assert np.max(np.abs(q_train)) <= np.pi + 1e-6
    assert audit["labels_used_by_transform"] is False
    changed = train.copy()
    changed[30:] += 1000.0
    q_train_2, q_val_2, _ = embedding_to_pca_angles(changed, fit, val, 8, 31)
    assert np.allclose(q_train, q_train_2)
    assert np.allclose(q_val, q_val_2)


def test_four_qubit_q8_reupload_forward_and_gradient():
    pytest.importorskip("pennylane")
    from aquire_preprocessing.models_quantum import ReuploadingQuantumClassifier

    model = ReuploadingQuantumClassifier(input_dim=8, n_qubits=4)
    assert model.uploads == 2 and model.n_qubits == 4
    x = torch.randn(3, 8, requires_grad=True)
    output = model(x)
    assert output.shape == (3,) and torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
