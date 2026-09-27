import inspect

import numpy as np
import torch

from aquire_preprocessing.models_diffusion import (
    ECGDiffusionEncoder,
    cosine_alpha_bar,
    min_snr_noise_loss,
    q_sample,
)
from run_diffusion_ecg_representation_export import _train_diffusion, _update_ema


def test_diffusion_encoder_shape_gradient_and_representation():
    torch.manual_seed(4)
    model = ECGDiffusionEncoder()
    clean = torch.randn(3, 12, 1000)
    timestep = torch.tensor([0, 200, 999])
    alpha_bar = cosine_alpha_bar(1000)
    noise = torch.randn_like(clean)
    noisy = q_sample(clean, timestep, alpha_bar, noise)
    prediction = model(noisy, timestep)
    representation = model.encode(clean)
    assert prediction.shape == clean.shape
    assert representation.shape == (3, 128)
    loss = min_snr_noise_loss(prediction, noise, timestep, alpha_bar)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_diffusion_schedule_and_sampling_are_finite():
    alpha_bar = cosine_alpha_bar(100)
    assert alpha_bar.shape == (100,)
    assert torch.all(alpha_bar[:-1] >= alpha_bar[1:])
    assert torch.all((alpha_bar > 0) & (alpha_bar < 1))
    clean = torch.zeros(2, 12, 1000)
    noise = torch.ones_like(clean)
    sampled = q_sample(clean, torch.tensor([0, 99]), alpha_bar, noise)
    assert torch.isfinite(sampled).all()
    assert sampled[1].abs().mean() > sampled[0].abs().mean()


def test_ema_update_and_pretraining_api_are_label_free():
    torch.manual_seed(8)
    model = ECGDiffusionEncoder()
    ema = ECGDiffusionEncoder()
    before = next(ema.parameters()).detach().clone()
    _update_ema(ema, model, 0.5)
    after = next(ema.parameters()).detach()
    expected = 0.5 * before + 0.5 * next(model.parameters()).detach()
    assert torch.allclose(after, expected)
    assert "labels" not in inspect.signature(_train_diffusion).parameters


def test_encoder_rejects_wrong_shape_and_nonfinite():
    model = ECGDiffusionEncoder()
    for values in (torch.zeros(2, 12, 999), torch.full((2, 12, 1000), float("nan"))):
        try:
            model.encode(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid ECG tensor was accepted")

