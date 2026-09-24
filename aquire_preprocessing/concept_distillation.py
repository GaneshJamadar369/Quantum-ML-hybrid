"""Clinically grounded auxiliary supervision for the retained q4 VQC.

The concept targets are deployable waveform measurements from the signed
v0.4 feature manifest.  They are deliberately free of diagnostic codes,
reports and MI labels.  Every transform is fit on an outer-training partition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


CONCEPT_COLUMNS = (
    "rr_cv",
    "clinical__global_st_rms_mv",
    "clinical__global_st_positive_count",
    "clinical__global_st_negative_count",
    "clinical__global_t_inversion_count",
    "clinical__inferior__st_mean_mv",
    "clinical__high_lateral__st_mean_mv",
    "clinical__anterior__st_mean_mv",
    "clinical__precordial_transition_lead",
    "clinical__frontal_axis_proxy_deg",
)


@dataclass(frozen=True)
class ConceptDistillationSpec:
    """One prespecified training schedule for the q4 student."""

    name: str
    stage1_epochs: int
    stage2_epochs: int
    hard_weight: float
    assistant_weight: float
    concept_weight: float
    reliability_weighted: bool = False

    def __post_init__(self) -> None:
        if not self.name or self.stage1_epochs < 0 or self.stage2_epochs < 0:
            raise ValueError("Invalid concept-distillation schedule")
        if self.stage1_epochs + self.stage2_epochs != 30:
            raise ValueError("Every screen arm must use exactly 30 epochs")
        weights = (self.hard_weight, self.assistant_weight, self.concept_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Loss weights must be non-negative")
        if self.stage1_epochs and not np.isclose(sum(weights), 1.0):
            raise ValueError("Stage-one loss weights must sum to one")

    def to_dict(self) -> dict:
        return asdict(self)


FROZEN_CONCEPT_SCREEN = (
    ConceptDistillationSpec("hard", 30, 0, 1.0, 0.0, 0.0),
    ConceptDistillationSpec("assistant_js_full", 30, 0, 0.5, 0.5, 0.0),
    ConceptDistillationSpec("assistant_two_stage", 10, 20, 0.5, 0.5, 0.0),
    ConceptDistillationSpec("concept_two_stage", 10, 20, 0.5, 0.0, 0.5),
    ConceptDistillationSpec(
        "concept_assistant_two_stage", 10, 20, 0.4, 0.3, 0.3
    ),
    ConceptDistillationSpec(
        "reliable_concept_assistant", 10, 20, 0.4, 0.3, 0.3, True
    ),
)


class FoldLocalConceptTransform:
    """Median-impute, robust-scale and bound concept targets fold-locally."""

    def __init__(self):
        self.median_: np.ndarray | None = None
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "FoldLocalConceptTransform":
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(CONCEPT_COLUMNS):
            raise ValueError("Concept matrix has the wrong shape")
        finite = np.where(np.isfinite(values), values, np.nan)
        self.median_ = np.nanmedian(finite, axis=0)
        self.median_ = np.where(np.isfinite(self.median_), self.median_, 0.0)
        filled = np.where(np.isfinite(values), values, self.median_)
        q25, q75 = np.percentile(filled, [25.0, 75.0], axis=0)
        self.center_ = np.median(filled, axis=0)
        self.scale_ = np.where(q75 - q25 > 1e-8, q75 - q25, 1.0)
        return self

    def transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.median_ is None or self.center_ is None or self.scale_ is None:
            raise RuntimeError("Concept transform is not fitted")
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(CONCEPT_COLUMNS):
            raise ValueError("Concept matrix has the wrong shape")
        observed = np.isfinite(values)
        filled = np.where(observed, values, self.median_)
        scaled = np.tanh((filled - self.center_) / self.scale_)
        return scaled.astype(np.float32), observed.astype(np.float32)

    def fit_transform(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(values).transform(values)


def assistant_reliability_weights(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> np.ndarray:
    """Estimate calibration-aware teacher reliability from OOF predictions.

    Confidence alone would trust confidently wrong teachers.  Within each
    fixed probability bin, reliability therefore combines distance from 0.5
    with agreement between the mean prediction and observed prevalence.
    """

    probability = np.asarray(probabilities, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(probability) != len(labels) or not np.isfinite(probability).all():
        raise ValueError("Assistant probabilities must be finite and aligned")
    if bool(((probability < 0.0) | (probability > 1.0)).any()):
        raise ValueError("Assistant probabilities must lie in [0, 1]")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    weights = np.zeros(len(probability), dtype=float)
    for index in range(bins):
        selected = assignments == index
        if not selected.any():
            continue
        calibration = 1.0 - min(
            1.0, 2.0 * abs(probability[selected].mean() - labels[selected].mean())
        )
        confidence = 2.0 * np.abs(probability[selected] - 0.5)
        weights[selected] = calibration * confidence
    return np.clip(weights, 0.0, 1.0).astype(np.float32)


def bernoulli_js_per_record(student_logits, teacher_probability, temperature: float = 2.0):
    """Bounded answer-distillation loss without reducing over records."""

    import torch

    epsilon = 1e-6
    student_positive = torch.sigmoid(student_logits.reshape(-1) / temperature)
    teacher_probability = teacher_probability.reshape(-1).to(student_logits.dtype)
    teacher_logit = torch.logit(teacher_probability.clamp(epsilon, 1.0 - epsilon))
    teacher_positive = torch.sigmoid(teacher_logit / temperature)
    student = torch.stack((1.0 - student_positive, student_positive), dim=-1).clamp(
        epsilon, 1.0 - epsilon
    )
    teacher = torch.stack((1.0 - teacher_positive, teacher_positive), dim=-1).clamp(
        epsilon, 1.0 - epsilon
    )
    mixture = 0.5 * (student + teacher)
    js = 0.5 * (
        (student * (student.log() - mixture.log())).sum(dim=-1)
        + (teacher * (teacher.log() - mixture.log())).sum(dim=-1)
    )
    return js * float(temperature) ** 2


def masked_concept_loss(prediction, target, observed):
    """Smooth-L1 concept loss over actually measured targets only."""

    import torch
    from torch.nn import functional as functional

    if prediction.shape != target.shape or target.shape != observed.shape:
        raise ValueError("Concept tensors must have identical shapes")
    per_value = functional.smooth_l1_loss(prediction, target, reduction="none")
    denominator = observed.sum().clamp_min(1.0)
    return (per_value * observed).sum() / denominator
