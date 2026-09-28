"""Fixed parallel hybrid inference orchestration.

This module contains no route selection. Both predictors must succeed before
fusion can run; failure of either route fails the whole prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


class HybridInferenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class HybridScores:
    quantum_score: float
    classical_score: float
    fusion_logit: float
    fused_probability: float
    calibrated_probability: float
    threshold: float

    @property
    def label(self) -> str:
        return "MI_PATTERN" if self.calibrated_probability >= self.threshold else "NON_MI_PATTERN"


ScoreRoute = Callable[[np.ndarray], float]
FusionRoute = Callable[[float, float], tuple[float, float]]
CalibrationRoute = Callable[[float], float]


class FixedParallelHybrid:
    """Run the quantum and classical routes for every eligible input."""

    def __init__(
        self,
        *,
        quantum_route: ScoreRoute,
        classical_route: ScoreRoute,
        fusion_route: FusionRoute,
        calibrator: CalibrationRoute,
        threshold: float,
    ) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("Decision threshold must lie strictly between zero and one")
        self.quantum_route = quantum_route
        self.classical_route = classical_route
        self.fusion_route = fusion_route
        self.calibrator = calibrator
        self.threshold = float(threshold)

    @staticmethod
    def _finite(name: str, value: float) -> float:
        result = float(value)
        if not np.isfinite(result):
            raise HybridInferenceError(f"{name} returned a non-finite value")
        return result

    def predict(self, signal_mv: np.ndarray) -> HybridScores:
        signal = np.asarray(signal_mv, dtype=np.float32)
        if signal.shape != (12, 1000) or not np.isfinite(signal).all():
            raise HybridInferenceError("Hybrid inference requires finite float32[12,1000]")

        # Neither branch is optional. Exceptions deliberately propagate as a
        # failed hybrid prediction instead of activating a fallback route.
        quantum_score = self._finite("quantum route", self.quantum_route(signal))
        classical_score = self._finite("classical route", self.classical_route(signal))
        fusion_logit, fused_probability = self.fusion_route(quantum_score, classical_score)
        fusion_logit = self._finite("fusion logit", fusion_logit)
        fused_probability = self._finite("fusion probability", fused_probability)
        calibrated = self._finite("calibrator", self.calibrator(fused_probability))
        if not 0.0 <= fused_probability <= 1.0 or not 0.0 <= calibrated <= 1.0:
            raise HybridInferenceError("Fusion and calibrated outputs must be probabilities")
        return HybridScores(
            quantum_score=quantum_score,
            classical_score=classical_score,
            fusion_logit=fusion_logit,
            fused_probability=fused_probability,
            calibrated_probability=calibrated,
            threshold=self.threshold,
        )
