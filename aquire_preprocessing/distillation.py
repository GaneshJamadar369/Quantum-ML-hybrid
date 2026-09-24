"""Losses for classical-to-quantum response distillation.

The teacher and student are binary classifiers.  Keeping the implementation
in probability space makes the direction of every divergence explicit:

* forward KL: ``KL(teacher || student)``;
* reverse KL: ``KL(student || teacher)``;
* Jensen-Shannon: the bounded symmetric alternative.

The hard-label term is always retained unless ``alpha`` is explicitly one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


DistillationKind = Literal["hard", "bidirectional_kl", "js"]


@dataclass(frozen=True)
class DistillationSpec:
    """Frozen definition of one student-training objective."""

    name: str
    kind: DistillationKind
    alpha: float = 0.5
    temperature: float = 2.0
    reverse_weight: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Distillation spec requires a name")
        if self.kind not in {"hard", "bidirectional_kl", "js"}:
            raise ValueError(f"Unsupported distillation kind: {self.kind}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.reverse_weight <= 1.0:
            raise ValueError("reverse_weight must be in [0, 1]")
        if self.kind == "hard" and self.alpha != 0.0:
            raise ValueError("hard-label control must have alpha=0")
        if self.kind != "bidirectional_kl" and self.reverse_weight != 0.0:
            raise ValueError("reverse_weight only applies to bidirectional KL")

    def to_dict(self) -> dict:
        return asdict(self)


FROZEN_DIVERGENCE_SCREEN = (
    DistillationSpec("hard", "hard", alpha=0.0, temperature=1.0),
    DistillationSpec("forward_t2", "bidirectional_kl", alpha=0.5, temperature=2.0),
    DistillationSpec(
        "mixed_rkl25_t2", "bidirectional_kl", alpha=0.5,
        temperature=2.0, reverse_weight=0.25,
    ),
    DistillationSpec(
        "symmetric_kl_t2", "bidirectional_kl", alpha=0.5,
        temperature=2.0, reverse_weight=0.5,
    ),
    DistillationSpec("js_t2", "js", alpha=0.5, temperature=2.0),
    DistillationSpec(
        "reverse_t2", "bidirectional_kl", alpha=0.5,
        temperature=2.0, reverse_weight=1.0,
    ),
)


def _bernoulli_distribution_from_logits(logits, temperature: float, epsilon: float):
    import torch

    positive = torch.sigmoid(logits / float(temperature))
    positive = positive.clamp(epsilon, 1.0 - epsilon)
    return torch.stack((1.0 - positive, positive), dim=-1)


def _bernoulli_distribution_from_probability(probability, temperature: float, epsilon: float):
    import torch

    probability = probability.clamp(epsilon, 1.0 - epsilon)
    teacher_logit = torch.logit(probability)
    return _bernoulli_distribution_from_logits(teacher_logit, temperature, epsilon)


def distillation_loss(
    student_logits,
    hard_targets,
    teacher_probability,
    spec: DistillationSpec,
    *,
    sample_weight=None,
    epsilon: float = 1e-6,
):
    """Return a finite scalar hard-label plus distillation loss.

    ``teacher_probability`` must come from a training-only teacher.  The
    temperature-squared factor preserves the conventional gradient scale.
    """
    import torch
    from torch.nn import functional as functional

    if student_logits.ndim != 1:
        student_logits = student_logits.reshape(-1)
    hard_targets = hard_targets.reshape(-1).to(student_logits.dtype)
    teacher_probability = teacher_probability.reshape(-1).to(student_logits.dtype)
    if not (
        len(student_logits) == len(hard_targets) == len(teacher_probability)
    ):
        raise ValueError("student, target and teacher tensors must have equal length")
    if not torch.isfinite(student_logits).all():
        raise ValueError("student logits must be finite")
    if not torch.isfinite(teacher_probability).all():
        raise ValueError("teacher probabilities must be finite")
    if bool(((teacher_probability < 0.0) | (teacher_probability > 1.0)).any()):
        raise ValueError("teacher probabilities must lie in [0, 1]")

    hard = functional.binary_cross_entropy_with_logits(
        student_logits, hard_targets, reduction="none"
    )
    if spec.kind == "hard":
        per_record = hard
    else:
        teacher = _bernoulli_distribution_from_probability(
            teacher_probability, spec.temperature, epsilon
        )
        student = _bernoulli_distribution_from_logits(
            student_logits, spec.temperature, epsilon
        )
        log_teacher = teacher.log()
        log_student = student.log()
        if spec.kind == "bidirectional_kl":
            forward = (teacher * (log_teacher - log_student)).sum(dim=-1)
            reverse = (student * (log_student - log_teacher)).sum(dim=-1)
            soft = (
                (1.0 - spec.reverse_weight) * forward
                + spec.reverse_weight * reverse
            )
        else:
            mixture = 0.5 * (teacher + student)
            log_mixture = mixture.log()
            soft = 0.5 * (
                (teacher * (log_teacher - log_mixture)).sum(dim=-1)
                + (student * (log_student - log_mixture)).sum(dim=-1)
            )
        soft = soft * float(spec.temperature) ** 2
        per_record = (1.0 - spec.alpha) * hard + spec.alpha * soft

    if sample_weight is not None:
        weight = sample_weight.reshape(-1).to(per_record.dtype)
        if len(weight) != len(per_record) or not torch.isfinite(weight).all():
            raise ValueError("sample weights must be finite and match the batch")
        per_record = per_record * weight
    result = per_record.mean()
    if not torch.isfinite(result):
        raise RuntimeError("distillation loss became non-finite")
    return result
