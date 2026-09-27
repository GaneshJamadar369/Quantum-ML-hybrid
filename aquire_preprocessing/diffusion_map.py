"""Fold-local landmark diffusion maps for compact ECG representations.

The implementation is deliberately inductive: landmarks, robust scaling,
kernel bandwidth, graph normalization, eigenvectors, and output quantiles are
fit from an outer-training partition only.  Validation examples are embedded
with a Nystrom extension and never enter the fitted graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist
from sklearn.preprocessing import QuantileTransformer, RobustScaler


def select_patient_unique_landmarks(
    patient_ids: np.ndarray, n_landmarks: int, seed: int
) -> np.ndarray:
    """Select at most one record per patient without using labels."""
    patient_ids = np.asarray(patient_ids)
    if patient_ids.ndim != 1:
        raise ValueError("patient_ids must be one-dimensional")
    if n_landmarks < 8:
        raise ValueError("n_landmarks must be at least 8")
    rng = np.random.default_rng(seed)
    unique = np.unique(patient_ids)
    chosen_patients = rng.choice(unique, size=min(n_landmarks, len(unique)), replace=False)
    indices = []
    for patient in chosen_patients:
        candidates = np.flatnonzero(patient_ids == patient)
        indices.append(int(rng.choice(candidates)))
    return np.asarray(indices, dtype=int)


def _effective_rank(values: np.ndarray) -> float:
    singular = np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
    energy = np.square(singular)
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    probability = energy / total
    probability = probability[probability > 0]
    return float(np.exp(-(probability * np.log(probability)).sum()))


@dataclass
class LandmarkDiffusionMap:
    """Sparse anisotropic diffusion map with an out-of-sample extension."""

    n_components: int = 4
    n_landmarks: int = 1024
    graph_neighbors: int = 32
    bandwidth_neighbors: int = 15
    alpha: float = 1.0
    random_state: int = 42

    def _validate(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] < self.n_components:
            raise ValueError("values must be a 2D matrix with enough columns")
        if not np.isfinite(values).all():
            raise ValueError("diffusion-map inputs must be finite")
        return values

    def fit(self, values: np.ndarray, patient_ids: np.ndarray):
        values = self._validate(values)
        patient_ids = np.asarray(patient_ids)
        if len(patient_ids) != len(values):
            raise ValueError("patient_ids length differs from values")
        self.scaler_ = RobustScaler(quantile_range=(25.0, 75.0)).fit(values)
        scaled = self.scaler_.transform(values)
        self.landmark_indices_ = select_patient_unique_landmarks(
            patient_ids, self.n_landmarks, self.random_state
        )
        landmarks = scaled[self.landmark_indices_]
        if len(landmarks) <= self.n_components + 1:
            raise ValueError("too few landmarks for requested components")
        squared = cdist(landmarks, landmarks, metric="sqeuclidean")
        k_band = min(max(2, self.bandwidth_neighbors), len(landmarks) - 1)
        kth = np.partition(squared, k_band, axis=1)[:, k_band]
        positive = kth[np.isfinite(kth) & (kth > 0)]
        if len(positive) == 0:
            raise ValueError("landmark geometry has zero bandwidth")
        self.bandwidth_ = float(np.median(positive))
        weights = np.exp(-squared / self.bandwidth_)
        k_graph = min(max(self.n_components + 2, self.graph_neighbors), len(landmarks) - 1)
        keep = np.zeros_like(weights, dtype=bool)
        nearest = np.argpartition(squared, k_graph, axis=1)[:, : k_graph + 1]
        keep[np.arange(len(landmarks))[:, None], nearest] = True
        keep = keep | keep.T
        weights *= keep
        np.fill_diagonal(weights, 1.0)

        density = np.maximum(weights.sum(axis=1), 1e-12)
        anisotropic = weights / (
            np.power(density[:, None], self.alpha)
            * np.power(density[None, :], self.alpha)
        )
        degree = np.maximum(anisotropic.sum(axis=1), 1e-12)
        symmetric = anisotropic / np.sqrt(degree[:, None] * degree[None, :])
        start = len(landmarks) - (self.n_components + 1)
        eigenvalues, eigenvectors = eigh(
            symmetric, subset_by_index=(start, len(landmarks) - 1), check_finite=False
        )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        # Drop the stationary eigenvector.  The remaining coordinates are the
        # one-step diffusion coordinates lambda_k * phi_k.
        self.eigenvalues_ = eigenvalues[1 : self.n_components + 1]
        self.eigenvectors_ = eigenvectors[:, 1 : self.n_components + 1]
        if np.min(np.abs(self.eigenvalues_)) < 1e-8:
            raise ValueError("diffusion graph has degenerate retained eigenvalues")
        self.landmarks_ = landmarks
        self.landmark_density_ = density
        self.landmark_degree_ = degree
        raw_train = self._transform_scaled(scaled)
        self.quantile_ = QuantileTransformer(
            n_quantiles=min(256, len(raw_train)),
            output_distribution="uniform",
            random_state=self.random_state,
        ).fit(raw_train)
        self.fit_audit_ = {
            "n_training_records": int(len(values)),
            "n_training_patients": int(len(np.unique(patient_ids))),
            "n_landmarks": int(len(landmarks)),
            "bandwidth": self.bandwidth_,
            "graph_neighbors": int(k_graph),
            "bandwidth_neighbors": int(k_band),
            "alpha": float(self.alpha),
            "eigenvalues": self.eigenvalues_.tolist(),
            "raw_effective_rank": _effective_rank(raw_train),
        }
        return self

    def _transform_scaled(self, scaled: np.ndarray, chunk_size: int = 2048) -> np.ndarray:
        outputs = []
        k_graph = min(max(self.n_components + 2, self.graph_neighbors), len(self.landmarks_))
        for start in range(0, len(scaled), chunk_size):
            squared = cdist(
                scaled[start : start + chunk_size], self.landmarks_, metric="sqeuclidean"
            )
            weights = np.exp(-squared / self.bandwidth_)
            if k_graph < len(self.landmarks_):
                nearest = np.argpartition(squared, k_graph - 1, axis=1)[:, :k_graph]
                sparse = np.zeros_like(weights)
                rows = np.arange(len(weights))[:, None]
                sparse[rows, nearest] = weights[rows, nearest]
                weights = sparse
            query_density = np.maximum(weights.sum(axis=1), 1e-12)
            anisotropic = weights / (
                np.power(query_density[:, None], self.alpha)
                * np.power(self.landmark_density_[None, :], self.alpha)
            )
            query_degree = np.maximum(anisotropic.sum(axis=1), 1e-12)
            symmetric_rows = anisotropic / np.sqrt(
                query_degree[:, None] * self.landmark_degree_[None, :]
            )
            outputs.append(symmetric_rows @ self.eigenvectors_)
        return np.vstack(outputs)

    def transform_raw(self, values: np.ndarray) -> np.ndarray:
        values = self._validate(values)
        return self._transform_scaled(self.scaler_.transform(values))

    def transform(self, values: np.ndarray) -> np.ndarray:
        raw = self.transform_raw(values)
        angles = (2.0 * self.quantile_.transform(raw) - 1.0) * (np.pi / 2.0)
        if not np.isfinite(angles).all():
            raise RuntimeError("non-finite diffusion-map angles")
        return angles.astype(np.float32)

    def fit_transform(self, values: np.ndarray, patient_ids: np.ndarray) -> np.ndarray:
        return self.fit(values, patient_ids).transform(values)

