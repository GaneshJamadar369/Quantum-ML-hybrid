"""Difficulty-aware response distillation for binary ECG classification.

Unlike autoregressive DA-KD, this module operates on one ECG and one binary
target.  It never removes records or reweights the hard-label loss.  It only
changes the relative strength of the training-only JS teacher term.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


DifficultyMode = Literal[
    "hard", "uniform", "agreement", "dynamic", "curriculum", "hard_negative"
]


@dataclass(frozen=True)
class DifficultyDistillationSpec:
    name: str
    mode: DifficultyMode
    alpha: float = 0.5
    hard_negative_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in {
            "hard", "uniform", "agreement", "dynamic", "curriculum", "hard_negative"
        }:
            raise ValueError(f"Unsupported difficulty mode: {self.mode}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if self.mode == "hard" and self.alpha != 0.0:
            raise ValueError("Hard control must have alpha=0")
        if self.hard_negative_multiplier < 1.0:
            raise ValueError("Hard-negative multiplier cannot be below one")
        if self.mode != "hard_negative" and self.hard_negative_multiplier != 1.0:
            raise ValueError("Hard-negative multiplier only applies to its named arm")

    def to_dict(self) -> dict:
        return asdict(self)


FROZEN_DIFFICULTY_SCREEN = (
    DifficultyDistillationSpec("hard", "hard", alpha=0.0),
    DifficultyDistillationSpec("uniform_js", "uniform"),
    DifficultyDistillationSpec("agreement_js", "agreement"),
    DifficultyDistillationSpec("dynamic_difficulty_js", "dynamic"),
    DifficultyDistillationSpec("curriculum_difficulty_js", "curriculum"),
    DifficultyDistillationSpec(
        "hard_negative_difficulty_js",
        "hard_negative",
        hard_negative_multiplier=1.5,
    ),
)


def difficulty_weights(
    student_logits,
    teacher_probability,
    hard_targets,
    calibration_reliability,
    hard_negative,
    spec: DifficultyDistillationSpec,
    *,
    epoch: int,
    epochs: int,
):
    """Return detached, mean-one weights for the soft JS loss only.

    Teacher agreement prevents a confidently wrong teacher from dominating.
    Dynamic difficulty combines student hard-label error and student/teacher
    disagreement.  Curriculum mode transitions from uniform JS to the dynamic
    rule, preserving the stable early optimization observed in G6Q-KD3.
    """

    import torch

    logits = student_logits.reshape(-1)
    teacher = teacher_probability.reshape(-1).to(logits.dtype)
    targets = hard_targets.reshape(-1).to(logits.dtype)
    reliability = calibration_reliability.reshape(-1).to(logits.dtype)
    hard_negative = hard_negative.reshape(-1).to(logits.dtype)
    if not (
        len(logits)
        == len(teacher)
        == len(targets)
        == len(reliability)
        == len(hard_negative)
    ):
        raise ValueError("Difficulty tensors must have equal length")
    if spec.mode in {"hard", "uniform"}:
        return torch.ones_like(logits)

    with torch.no_grad():
        student = torch.sigmoid(logits)
        teacher_agreement = targets * teacher + (1.0 - targets) * (1.0 - teacher)
        student_correct = targets * student + (1.0 - targets) * (1.0 - student)
        student_error = 1.0 - student_correct
        disagreement = torch.abs(student - teacher)
        base = reliability.clamp(0.0, 1.0) * teacher_agreement.clamp(0.0, 1.0)
        if spec.mode == "agreement":
            raw = base
        else:
            dynamic = base * (0.5 * student_error + 0.5 * disagreement)
            if spec.mode == "curriculum":
                progress = float(epoch) / float(max(epochs - 1, 1))
                raw = (1.0 - progress) * torch.ones_like(dynamic) + progress * dynamic
            else:
                raw = dynamic
            if spec.mode == "hard_negative":
                raw = raw * (
                    1.0 + (spec.hard_negative_multiplier - 1.0) * hard_negative
                )
        # Keep every record represented in the distillation term, normalize the
        # effective soft-loss scale, and cap individual influence.
        raw = raw.clamp_min(1e-4)
        normalized = raw / raw.mean().clamp_min(1e-4)
        return normalized.clamp(0.25, 2.0)
