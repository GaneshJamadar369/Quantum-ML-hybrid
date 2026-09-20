"""
AQUIRE-Med Quantum Machine Learning Models (Phase 6Q).
Implements:
1. QuantumKernelClassifier: Computes quantum fidelity kernel K(x_i, x_j) with dual SVM.
2. VariationalQuantumClassifier: Parameterized quantum circuit (PQC) with trainable variational angles.
3. HybridQuantumNeuralNetwork: End-to-end 1D-ResNet/MLP backbone + PennyLane QNode bottleneck + Readout.
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


# =====================================================================
# 1. Quantum Feature Mapping & Kernel Estimation (QSVM / QKE)
# =====================================================================

class QuantumKernelEstimator:
    """
    Quantum Kernel Estimator using PennyLane.
    Maps clinical feature vectors into a 2^n Hilbert space using Angle + Entanglement embedding
    and calculates fidelity |<psi(x_i)|psi(x_j)>|^2.
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 2, dev_name: str = "default.qubit"):
        if qml is None:
            raise ImportError("PennyLane is required for QuantumKernelEstimator. Please install pennylane.")
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(dev_name, wires=n_qubits)
        self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev)
        def kernel_circuit(x1, x2):
            # Encode x1
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(x1[i], wires=i)
                    qml.RZ(x1[i], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Invert encoding for x2 (adjoint)
            for layer in reversed(range(self.n_layers)):
                if self.n_qubits > 2:
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
                for i in reversed(range(self.n_qubits - 1)):
                    qml.CNOT(wires=[i, i + 1])
                for i in reversed(range(self.n_qubits)):
                    qml.RZ(-x2[i], wires=i)
                    qml.RY(-x2[i], wires=i)

            return qml.probs(wires=range(self.n_qubits))

        self.kernel_circuit = kernel_circuit

    def compute_kernel_matrix(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes pairwise quantum kernel matrix.
        If X2 is None, computes symmetric self-kernel K(X1, X1).
        """
        is_symmetric = X2 is None
        if is_symmetric:
            X2 = X1

        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2), dtype=np.float32)

        # Scale features to [-pi, pi] for angle embedding
        X1_scaled = np.clip(X1[:, :self.n_qubits], -np.pi, np.pi)
        X2_scaled = np.clip(X2[:, :self.n_qubits], -np.pi, np.pi)

        if is_symmetric:
            for i in range(n1):
                K[i, i] = 1.0
                for j in range(i + 1, n1):
                    prob = self.kernel_circuit(X1_scaled[i], X2_scaled[j])[0]
                    K[i, j] = prob
                    K[j, i] = prob
        else:
            for i in range(n1):
                for j in range(n2):
                    prob = self.kernel_circuit(X1_scaled[i], X2_scaled[j])[0]
                    K[i, j] = prob

        return K


class QSVMClassifier(BaseEstimator, ClassifierMixin):
    """
    Quantum Support Vector Classifier wrapping QuantumKernelEstimator with scikit-learn SVC.
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 2, C: float = 1.0):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.C = C
        self.qke = None
        self.scaler = StandardScaler()
        self.svm = SVC(kernel="precomputed", C=self.C, probability=True)
        self.X_train_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        if self.qke is None:
            self.qke = QuantumKernelEstimator(n_qubits=self.n_qubits, n_layers=self.n_layers)
        
        # Select top features and scale to [-pi, pi]
        X_sub = X[:, :self.n_qubits]
        X_scaled = self.scaler.fit_transform(X_sub)
        X_scaled = np.tanh(X_scaled) * np.pi  # non-linear bounding to [-pi, pi]
        self.X_train_ = X_scaled

        K_train = self.qke.compute_kernel_matrix(self.X_train_)
        self.svm.fit(K_train, y)
        self.classes_ = self.svm.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_sub = X[:, :self.n_qubits]
        X_scaled = self.scaler.transform(X_sub)
        X_scaled = np.tanh(X_scaled) * np.pi
        K_test = self.qke.compute_kernel_matrix(X_scaled, self.X_train_)
        return self.svm.predict_proba(K_test)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# =====================================================================
# 2. Variational Quantum Classifier (VQC / PQC)
# =====================================================================

class VariationalQuantumClassifier(nn.Module):
    """
    Variational Quantum Classifier using PyTorch + PennyLane.
    Architecture:
    Input (Features) -> Linear Encoder -> AngleEmbedding -> StronglyEntanglingLayers -> Expectation Values (PauliZ) -> Linear Readout
    """
    def __init__(self, in_features: int = 106, n_qubits: int = 8, n_layers: int = 3):
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane is required for VariationalQuantumClassifier.")
        
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Pre-encoder to project input features to n_qubits
        self.encoder = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 32),
            nn.SiLU(),
            nn.Linear(32, n_qubits),
            nn.Tanh()  # Output range [-1, 1], scaled by pi in quantum circuit
        )

        # Quantum Device & Circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint" if hasattr(self.dev, "adjoint_jacobian") else "backprop")
        def quantum_circuit(inputs, weights):
            # inputs shape: (n_qubits,)
            # Angle embedding
            for i in range(self.n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            # Parameterized Entangling layers
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            # Readout PauliZ expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Variational parameters shape: (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.qnode = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Readout classification head
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.SiLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, in_features)
        z = self.encoder(x)  # (Batch, n_qubits)
        q_out = self.qnode(z)  # (Batch, n_qubits)
        logits = self.classifier(q_out)  # (Batch, 1)
        return logits.squeeze(-1)


# =====================================================================
# 3. Hybrid Quantum-Classical Deep Neural Network (HQNN)
# =====================================================================

class HybridQuantumNeuralNetwork(nn.Module):
    """
    Multimodal Hybrid Quantum Neural Network.
    Fuses raw 12-lead ECG waveforms (via 1D-ResNet) with handcrafted clinical features (via MLP),
    processes the joint representation through a Quantum Bottleneck circuit, and predicts MI probability.
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

        # 1. Classical 1D Waveform Stream (Lightweight ResNet Encoder)
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

        # 2. Classical Tabular Feature Stream (MLP Encoder)
        self.tabular_encoder = nn.Sequential(
            nn.BatchNorm1d(tabular_dim),
            nn.Linear(tabular_dim, 64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_qubits // 2)
        )

        # 3. Quantum Bottleneck Circuit
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint" if hasattr(self.dev, "adjoint_jacobian") else "backprop")
        def hybrid_quantum_circuit(inputs, weights):
            # inputs shape: (n_qubits,)
            for i in range(self.n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)
            
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        weight_shapes = {"weights": (n_quantum_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(hybrid_quantum_circuit, weight_shapes)

        # 4. Final Classification Head
        self.head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 1)
        )

    def forward(self, x_signal: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        # Waveform stream
        wf = self.waveform_stem(x_signal)
        wf = self.waveform_conv(wf).flatten(1)
        wf_lat = torch.tanh(self.waveform_fc(wf))  # (Batch, n_qubits/2)

        # Tabular stream
        tab_lat = torch.tanh(self.tabular_encoder(x_tab))  # (Batch, n_qubits/2)

        # Concatenate joint latent representation
        joint_lat = torch.cat([wf_lat, tab_lat], dim=1)  # (Batch, n_qubits)

        # Pass through quantum bottleneck
        q_out = self.quantum_layer(joint_lat)  # (Batch, n_qubits)

        # Readout head
        logits = self.head(q_out)
        return logits.squeeze(-1)
