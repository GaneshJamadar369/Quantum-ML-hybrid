import pytest

from aquire_preprocessing.difficulty_distillation import (
    FROZEN_DIFFICULTY_SCREEN,
    difficulty_weights,
)


def test_frozen_difficulty_screen_is_unique_and_has_hard_control():
    assert FROZEN_DIFFICULTY_SCREEN[0].name == "hard"
    assert FROZEN_DIFFICULTY_SCREEN[0].alpha == 0.0
    assert len({spec.name for spec in FROZEN_DIFFICULTY_SCREEN}) == len(
        FROZEN_DIFFICULTY_SCREEN
    )


def test_dynamic_weight_prefers_teacher_correct_student_error():
    torch = pytest.importorskip("torch")
    spec = next(s for s in FROZEN_DIFFICULTY_SCREEN if s.mode == "dynamic")
    # First record: teacher correct and student wrong. Second: both correct.
    logits = torch.tensor([-2.0, 2.0], requires_grad=True)
    teacher = torch.tensor([0.95, 0.95])
    labels = torch.tensor([1.0, 1.0])
    reliability = torch.ones(2)
    hard_negative = torch.zeros(2)
    weight = difficulty_weights(
        logits,
        teacher,
        labels,
        reliability,
        hard_negative,
        spec,
        epoch=10,
        epochs=30,
    )
    assert weight[0] > weight[1]
    assert not weight.requires_grad
    assert 0.25 <= float(weight.min()) <= float(weight.max()) <= 2.0


def test_teacher_disagreement_with_label_is_downweighted():
    torch = pytest.importorskip("torch")
    spec = next(s for s in FROZEN_DIFFICULTY_SCREEN if s.mode == "agreement")
    logits = torch.zeros(2)
    teacher = torch.tensor([0.95, 0.05])
    labels = torch.ones(2)
    weight = difficulty_weights(
        logits,
        teacher,
        labels,
        torch.ones(2),
        torch.zeros(2),
        spec,
        epoch=0,
        epochs=30,
    )
    assert weight[0] > weight[1]


def test_hard_negative_arm_increases_hard_negative_relative_weight():
    torch = pytest.importorskip("torch")
    spec = next(s for s in FROZEN_DIFFICULTY_SCREEN if s.mode == "hard_negative")
    weight = difficulty_weights(
        torch.tensor([1.0, 1.0]),
        torch.tensor([0.1, 0.1]),
        torch.tensor([0.0, 0.0]),
        torch.ones(2),
        torch.tensor([1.0, 0.0]),
        spec,
        epoch=15,
        epochs=30,
    )
    assert weight[0] > weight[1]
