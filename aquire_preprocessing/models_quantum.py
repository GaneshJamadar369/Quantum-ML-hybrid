"""
AQUIRE-Med Quantum Machine Learning Models (Phase 6Q).
Optimized Vectorized & Batched PennyLane + PyTorch Implementation:
1. QuantumKernelClassifier (QSVM / QKE): Batch-computed statevector overlap for ultra-fast kernel matrix generation.
2. VariationalQuantumClassifier (VQC): Parameterized quantum circuit (PQC) with AngleEmbedding + StronglyEntanglingLayers.
3. HybridQuantumNeuralNetwork (HQNN): Multimodal 1D-ResNet + MLP + Quantum Bottleneck (AngleEmbedding + Entanglement) + Readout.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

try:
    import pennylane as qml
except ImportError:
    qml = None


def get_quantum_device(n_qubits: int):
    """Select a deterministic analytic state-vector simulator.

    ``lightning.qubit`` is a CPU simulator.  Its presence must never be
    reported as quantum-hardware or GPU execution.
    """
    if qml is None:
        raise ImportError("PennyLane is required for quantum models")
    try:
        return qml.device("lightning.qubit", wires=n_qubits)
    except Exception:
        return qml.device("default.qubit", wires=n_qubits)


# =====================================================================
# 1. Quantum Feature Mapping & Fast Vectorized Kernel Estimation (QSVM)
# =====================================================================

class QuantumKernelEstimator:
    """
    Vectorized Quantum Kernel Estimator using PennyLane.
    Maps clinical feature vectors into statevectors in 2^n Hilbert space
    and calculates fidelity matrix via batch matrix multiplication:
    K(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2 = |Psi_1 @ Psi_2^H|^2
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 2):
        if qml is None:
            raise ImportError("PennyLane is required for QuantumKernelEstimator.")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = get_quantum_device(n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev)
        def state_circuit(x):
            # IQPEmbedding contains data-dependent one- and two-qubit phase
            # terms.  The former implementation appended a fixed CNOT ring
            # after AngleEmbedding; a shared data-independent unitary cancels
            # from <psi(x)|psi(y)> and therefore added no kernel structure.
            qml.IQPEmbedding(
                features=x,
                wires=range(self.n_qubits),
                n_repeats=self.n_layers,
                pattern=None,
            )
            return qml.state()

        self.state_circuit = state_circuit

    def get_statevectors(self, X: np.ndarray) -> np.ndarray:
        """Compute statevector for each sample in X."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_qubits:
            raise ValueError(
                f"Quantum kernel expects shape (n, {self.n_qubits}), got {X.shape}"
            )
        X_scaled = np.clip(X, -np.pi, np.pi)
        states = []
        for row in X_scaled:
            states.append(self.state_circuit(row))
        return np.array(states, dtype=np.complex128)

    def compute_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes pairwise quantum kernel matrix using vectorized state overlaps.
        Runs in seconds instead of hours.
        """
        psi1 = self.get_statevectors(X1)
        if X2 is None or X2 is X1:
            overlap = np.abs(np.dot(psi1, psi1.conj().T)) ** 2
            overlap = (overlap + overlap.T) / 2.0
            np.fill_diagonal(overlap, 1.0)
            return overlap.astype(np.float64)
        else:
            psi2 = self.get_statevectors(X2)
            overlap = np.abs(np.dot(psi1, psi2.conj().T)) ** 2
            return overlap.astype(np.float64)

    @staticmethod
    def diagnostics(kernel: np.ndarray) -> Dict[str, float]:
        """Return numerical checks required before fitting an SVM."""
        matrix = np.asarray(kernel, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Kernel diagnostics require a square matrix")
        symmetric = (matrix + matrix.T) / 2.0
        eigenvalues = np.linalg.eigvalsh(symmetric)
        return {
            "max_asymmetry": float(np.max(np.abs(matrix - matrix.T))),
            "max_diagonal_error": float(np.max(np.abs(np.diag(matrix) - 1.0))),
            "minimum_eigenvalue": float(eigenvalues.min()),
            "negative_eigenvalue_count": int((eigenvalues < -1e-8).sum()),
            "condition_number": float(np.linalg.cond(symmetric + 1e-8 * np.eye(len(matrix)))),
        }


class ProjectedIQPFeatureMap:
    """Shallow sparse IQP map exposing local quantum observables.

    The returned feature vector contains one-qubit X/Y/Z expectations and ZZ
    expectations on the selected entanglement edges.  A classical kernel may
    then be applied to these locally projected quantum features.  This avoids
    relying exclusively on a global state-fidelity measurement.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 1,
        feature_scale: Union[float, np.ndarray] = 0.5,
        interaction_scale: float = 0.5,
        topology: str = "ring",
        mixing_seed: int = 42,
    ):
        if qml is None:
            raise ImportError("PennyLane is required for ProjectedIQPFeatureMap")
        if n_qubits < 2:
            raise ValueError("Projected IQP map requires at least two qubits")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        scale = np.asarray(feature_scale, dtype=float)
        if scale.ndim == 0:
            scale = np.repeat(scale, self.n_qubits)
        if scale.shape != (self.n_qubits,) or not np.isfinite(scale).all():
            raise ValueError(f"feature_scale must be finite shape ({self.n_qubits},)")
        self.feature_scale = scale
        self.interaction_scale = float(interaction_scale)
        self.topology = topology
        self.edges = self._make_edges(topology)
        rng = np.random.default_rng(mixing_seed)
        # Small non-commuting rotations prevent repeated diagonal layers from
        # collapsing into one rescaled layer while keeping the circuit shallow.
        self.mixing_angles = rng.uniform(-np.pi / 4, np.pi / 4, (self.n_layers, self.n_qubits))
        self.dev = get_quantum_device(self.n_qubits)
        self._build_circuit()

    def _make_edges(self, topology: str) -> list[tuple[int, int]]:
        if topology == "ring":
            return [(idx, (idx + 1) % self.n_qubits) for idx in range(self.n_qubits)]
        if topology == "ladder":
            edges = {(idx, idx + 1) for idx in range(self.n_qubits - 1)}
            edges.update((idx, idx + 2) for idx in range(self.n_qubits - 2))
            return sorted(edges)
        raise ValueError("topology must be 'ring' or 'ladder'")

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev)
        def projected_circuit(x):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
            for layer in range(self.n_layers):
                for wire in range(self.n_qubits):
                    angle = self.feature_scale[wire] * x[wire]
                    qml.RZ(angle, wires=wire)
                    qml.RX(0.5 * angle, wires=wire)
                for left, right in self.edges:
                    angle = self.interaction_scale * x[left] * x[right]
                    qml.IsingZZ(angle, wires=[left, right])
                for wire in range(self.n_qubits):
                    qml.RY(self.mixing_angles[layer, wire], wires=wire)
            observables = []
            for wire in range(self.n_qubits):
                observables.extend(
                    [
                        qml.expval(qml.PauliX(wire)),
                        qml.expval(qml.PauliY(wire)),
                        qml.expval(qml.PauliZ(wire)),
                    ]
                )
            observables.extend(
                qml.expval(qml.PauliZ(left) @ qml.PauliZ(right))
                for left, right in self.edges
            )
            return observables

        self.projected_circuit = projected_circuit

    def transform(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim != 2 or array.shape[1] != self.n_qubits:
            raise ValueError(
                f"Projected IQP map expects shape (n, {self.n_qubits}), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("Projected IQP inputs must be finite")
        return np.asarray([self.projected_circuit(row) for row in array], dtype=float)

    @staticmethod
    def rbf_kernel(
        left_features: np.ndarray,
        right_features: Optional[np.ndarray] = None,
        gamma: Optional[float] = None,
    ) -> tuple[np.ndarray, float]:
        from sklearn.metrics import pairwise_distances

        left = np.asarray(left_features, dtype=float)
        right = left if right_features is None else np.asarray(right_features, dtype=float)
        distances = pairwise_distances(left, right, metric="sqeuclidean")
        if gamma is None:
            reference = distances[np.triu_indices_from(distances, k=1)] if right_features is None else distances.ravel()
            positive = reference[reference > 1e-12]
            median = float(np.median(positive)) if len(positive) else 1.0
            gamma = 1.0 / median
        kernel = np.exp(-float(gamma) * distances)
        return kernel, float(gamma)


class QSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Quantum Support Vector Classifier wrapping QuantumKernelEstimator with scikit-learn SVC.
    """
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        C: float = 1.0,
        calibration_splits: int = 5,
        seed: int = 42,
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.C = C
        self.calibration_splits = calibration_splits
        self.seed = seed
        self.qke = None
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="precomputed", C=self.C, class_weight="balanced")
        self.X_train_ = None
        self.calibrator_ = None
        self.kernel_diagnostics_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.qke is None:
            self.qke = QuantumKernelEstimator(n_qubits=self.n_qubits, n_layers=self.n_layers)
        
        X_sub = X[:, :self.n_qubits]
        X_scaled = self.scaler.fit_transform(X_sub)
        X_scaled = np.tanh(X_scaled) * np.pi
        self.X_train_ = X_scaled

        K_train = self.qke.compute_kernel_matrix(self.X_train_)
        self.kernel_diagnostics_ = self.qke.diagnostics(K_train)
        if self.kernel_diagnostics_["negative_eigenvalue_count"]:
            raise ValueError(
                "Quantum training kernel is not positive semidefinite within tolerance: "
                f"{self.kernel_diagnostics_}"
            )

        # Train-only, cross-fitted Platt calibration.  This avoids both the
        # deprecated SVC(probability=True) path and the former second layer of
        # calibration applied after outer-fold prediction.
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold

        y = np.asarray(y, dtype=int)
        class_counts = np.bincount(y, minlength=2)
        splits = min(int(self.calibration_splits), int(class_counts.min()))
        if splits < 2:
            raise ValueError("At least two examples per class are needed for calibration")
        calibration_score = np.full(len(y), np.nan, dtype=float)
        cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=self.seed)
        for inner_train, inner_cal in cv.split(K_train, y):
            inner_svm = SVC(
                kernel="precomputed", C=self.C, class_weight="balanced"
            )
            inner_svm.fit(K_train[np.ix_(inner_train, inner_train)], y[inner_train])
            calibration_score[inner_cal] = inner_svm.decision_function(
                K_train[np.ix_(inner_cal, inner_train)]
            )
        if not np.isfinite(calibration_score).all():
            raise RuntimeError("Incomplete train-only calibration scores")
        self.calibrator_ = LogisticRegression(solver="lbfgs", max_iter=500)
        self.calibrator_.fit(calibration_score.reshape(-1, 1), y)
        self.svm.fit(K_train, y)
        self.classes_ = self.svm.classes_
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.qke is None or self.X_train_ is None:
            raise RuntimeError("QSVMClassifier is not fitted")
        X_sub = X[:, :self.n_qubits]
        X_scaled = self.scaler.transform(X_sub)
        X_scaled = np.tanh(X_scaled) * np.pi
        K_test = self.qke.compute_kernel_matrix(X_scaled, self.X_train_)
        return np.asarray(self.svm.decision_function(K_test), dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.calibrator_ is None:
            raise RuntimeError("QSVMClassifier is not fitted")
        score = self.decision_function(X)
        return self.calibrator_.predict_proba(score.reshape(-1, 1))

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =====================================================================
# 2. Variational Quantum Classifier (VQC / PQC)
# =====================================================================

class VariationalQuantumClassifier(nn.Module):
    """
    Variational Quantum Classifier using PyTorch + PennyLane.
    Uses AngleEmbedding for native multi-sample batch support.
    """
    def __init__(self, in_features: int = 106, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required for VariationalQuantumClassifier.")
        
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 32),
            nn.SiLU(),
            nn.Linear(32, n_qubits),
            nn.Tanh()
        )

        self.dev = get_quantum_device(n_qubits)

        # ``parameter-shift`` cannot differentiate the broadcasted trainable
        # inputs produced by TorchLayer.  Adjoint differentiation on
        # lightning.qubit supports batched inputs and is substantially cheaper
        # for this analytic simulator; default.qubit uses backpropagation.
        diff_method = "adjoint" if self.dev.name == "lightning.qubit" else "backprop"

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        q_out = self.qnode(z)
        logits = self.classifier(q_out)
        return logits.squeeze(-1)


class DirectQuantumClassifier(nn.Module):
    """Direct-input VQC for an already conditioned ``q_d`` representation.

    Unlike :class:`VariationalQuantumClassifier`, this model does not place a
    high-capacity classical MLP before the circuit.  The input is expected to
    be the fold-local quantum representation in ``[-pi, pi]``.  Trainable
    feature scales, data re-uploading, sparse Ising interactions and local
    observables provide the expressive part of the predictive head.  The
    classical readout is deliberately linear so the quantum transformation is
    identifiable in matched ablations.
    """

    def __init__(
        self,
        n_qubits: int,
        n_layers: int = 2,
        topology: str = "ring",
    ):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required for DirectQuantumClassifier")
        if n_qubits < 2:
            raise ValueError("DirectQuantumClassifier requires at least two qubits")
        if n_layers < 1:
            raise ValueError("n_layers must be positive")
        if topology == "ring":
            edges = [(index, (index + 1) % n_qubits) for index in range(n_qubits)]
        elif topology == "ladder":
            edge_set = {(index, index + 1) for index in range(n_qubits - 1)}
            edge_set.update((index, index + 2) for index in range(n_qubits - 2))
            edges = sorted(edge_set)
        else:
            raise ValueError("topology must be 'ring' or 'ladder'")

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.topology = topology
        self.edges = edges
        self.dev = get_quantum_device(self.n_qubits)
        diff_method = "adjoint" if self.dev.name == "lightning.qubit" else "backprop"

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, rotations, interactions, feature_scales):
            # Bounded trainable bandwidth.  Re-uploading lets shallow circuits
            # build nonlinear feature interactions without a classical encoder.
            scaled = inputs * (2.0 * torch.sigmoid(feature_scales))
            for layer in range(self.n_layers):
                qml.AngleEmbedding(scaled, wires=range(self.n_qubits), rotation="Y")
                for wire in range(self.n_qubits):
                    qml.Rot(
                        rotations[layer, wire, 0],
                        rotations[layer, wire, 1],
                        rotations[layer, wire, 2],
                        wires=wire,
                    )
                for edge_index, (left, right) in enumerate(self.edges):
                    qml.IsingZZ(interactions[layer, edge_index], wires=[left, right])
            observables = [qml.expval(qml.PauliZ(wire)) for wire in range(self.n_qubits)]
            observables.extend(
                qml.expval(qml.PauliZ(left) @ qml.PauliZ(right))
                for left, right in self.edges
            )
            return observables

        weight_shapes = {
            "rotations": (self.n_layers, self.n_qubits, 3),
            "interactions": (self.n_layers, len(self.edges)),
            "feature_scales": (self.n_qubits,),
        }
        # Near-identity initialization avoids beginning the search in a highly
        # random circuit where gradients can be weak.  A zero feature-scale
        # parameter corresponds to a unit multiplier because 2*sigmoid(0)=1.
        init_method = {
            "rotations": lambda tensor: nn.init.normal_(tensor, mean=0.0, std=0.1),
            "interactions": lambda tensor: nn.init.normal_(tensor, mean=0.0, std=0.1),
            "feature_scales": nn.init.zeros_,
        }
        self.quantum_layer = qml.qnn.TorchLayer(
            circuit, weight_shapes, init_method=init_method
        )
        self.readout = nn.Linear(self.n_qubits + len(self.edges), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.n_qubits:
            raise ValueError(
                f"Expected (batch, {self.n_qubits}) quantum input, got {tuple(x.shape)}"
            )
        q_out = self.quantum_layer(x)
        return self.readout(q_out).squeeze(-1)


# =====================================================================
# 3. Hybrid Quantum-Classical Deep Neural Network (HQNN)
# =====================================================================

class HybridQuantumNeuralNetwork(nn.Module):
    """
    Multimodal Hybrid Quantum Neural Network.
    Fuses raw 12-lead ECG waveforms (via 1D-ResNet) with handcrafted clinical features (via MLP),
    processes joint representation through an 8-qubit quantum bottleneck with AngleEmbedding.
    """
    def __init__(
        self,
        tabular_dim: int = 106,
        raw_channels: int = 12,
        n_qubits: int = 8,
        n_quantum_layers: int = 3,
        resnet_base_filters: int = 32
    ):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required for HybridQuantumNeuralNetwork.")
        
        self.n_qubits = n_qubits
        self.n_quantum_layers = n_quantum_layers

        self.waveform_stem = nn.Sequential(
            nn.Conv1d(raw_channels, resnet_base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(resnet_base_filters),
            nn.SiLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.waveform_conv = nn.Sequential(
            nn.Conv1d(resnet_base_filters, resnet_base_filters * 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(resnet_base_filters * 2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.waveform_fc = nn.Linear(resnet_base_filters * 2, n_qubits // 2)

        self.tabular_encoder = nn.Sequential(
            nn.BatchNorm1d(tabular_dim),
            nn.Linear(tabular_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_qubits // 2)
        )

        self.dev = get_quantum_device(n_qubits)

        diff_method = "adjoint" if self.dev.name == "lightning.qubit" else "backprop"

        @qml.qnode(self.dev, interface="torch", diff_method=diff_method)
        def hybrid_quantum_circuit(inputs, weights):
            qml.AngleEmbedding(inputs * np.pi, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_quantum_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(hybrid_quantum_circuit, weight_shapes)

        self.head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x_signal: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        wf = self.waveform_stem(x_signal)
        wf = self.waveform_conv(wf).flatten(1)
        wf_lat = torch.tanh(self.waveform_fc(wf))

        tab_lat = torch.tanh(self.tabular_encoder(x_tab))
        joint_lat = torch.cat([wf_lat, tab_lat], dim=1)

        q_out = self.quantum_layer(joint_lat)
        logits = self.head(q_out)
        return logits.squeeze(-1)
