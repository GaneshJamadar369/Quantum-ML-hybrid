"""Compact denoising-diffusion encoder for 10-second 12-lead ECGs."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int = 64) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be even and >= 4")
        self.dimension = dimension

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        if timestep.ndim != 1:
            raise ValueError("timestep must have shape (batch,)")
        half = self.dimension // 2
        frequency = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        phase = timestep.float().unsqueeze(1) * frequency.unsqueeze(0)
        return torch.cat((phase.sin(), phase.cos()), dim=1)


def _groups(channels: int) -> int:
    for value in (8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class TimeResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, stride: int = 1):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=2)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.activation = nn.SiLU()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv1d(in_channels, out_channels, 1, stride=stride)
        )

    def forward(self, values: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(self.activation(self.norm1(values)))
        hidden = hidden + self.time_projection(time_embedding).unsqueeze(-1)
        hidden = self.conv2(self.activation(self.norm2(hidden)))
        return hidden + self.skip(values)


class ECGDiffusionEncoder(nn.Module):
    """Predict DDPM noise and expose the forced bottleneck as h128.

    The decoder has no waveform skip connections, so the 96-channel temporal
    bottleneck cannot be bypassed during label-free denoising pretraining.
    """

    def __init__(self, time_dim: int = 128, embedding_dim: int = 128) -> None:
        super().__init__()
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(64),
            nn.Linear(64, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.stem = nn.Conv1d(12, 32, 7, padding=3)
        self.down1 = TimeResidualBlock(32, 32, time_dim)
        self.down2 = TimeResidualBlock(32, 48, time_dim, stride=2)
        self.down3 = TimeResidualBlock(48, 64, time_dim, stride=2)
        self.down4 = TimeResidualBlock(64, 96, time_dim, stride=2)
        self.middle = TimeResidualBlock(96, 96, time_dim)
        self.up3 = TimeResidualBlock(96, 64, time_dim)
        self.up2 = TimeResidualBlock(64, 48, time_dim)
        self.up1 = TimeResidualBlock(48, 32, time_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv1d(32, 12, 3, padding=1),
        )
        self.representation = nn.Sequential(
            nn.Linear(192, embedding_dim),
            nn.SiLU(),
            nn.LayerNorm(embedding_dim),
        )

    @staticmethod
    def _validate(signal: torch.Tensor) -> None:
        if signal.ndim != 3 or tuple(signal.shape[1:]) != (12, 1000):
            raise ValueError(f"Expected (batch, 12, 1000), received {tuple(signal.shape)}")
        if not torch.isfinite(signal).all():
            raise ValueError("ECG input contains NaN or infinity")

    def _encode_features(
        self, signal: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.stem(signal)
        hidden = self.down1(hidden, time_embedding)
        hidden = self.down2(hidden, time_embedding)
        hidden = self.down3(hidden, time_embedding)
        hidden = self.down4(hidden, time_embedding)
        return self.middle(hidden, time_embedding)

    def forward(self, noisy_signal: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        self._validate(noisy_signal)
        if timestep.shape != (len(noisy_signal),):
            raise ValueError("timestep must contain one value per ECG")
        time_embedding = self.time_embedding(timestep)
        hidden = self._encode_features(noisy_signal, time_embedding)
        hidden = nn.functional.interpolate(hidden, size=250, mode="linear", align_corners=False)
        hidden = self.up3(hidden, time_embedding)
        hidden = nn.functional.interpolate(hidden, size=500, mode="linear", align_corners=False)
        hidden = self.up2(hidden, time_embedding)
        hidden = nn.functional.interpolate(hidden, size=1000, mode="linear", align_corners=False)
        hidden = self.up1(hidden, time_embedding)
        return self.output(hidden)

    def encode(self, clean_signal: torch.Tensor) -> torch.Tensor:
        """Return a 128-dimensional clean-ECG representation."""
        self._validate(clean_signal)
        timestep = torch.zeros(len(clean_signal), device=clean_signal.device)
        hidden = self._encode_features(clean_signal, self.time_embedding(timestep))
        pooled = torch.cat((hidden.mean(dim=-1), hidden.amax(dim=-1)), dim=1)
        return self.representation(pooled)


def cosine_alpha_bar(steps: int = 1000, offset: float = 0.008) -> torch.Tensor:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    grid = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    curve = torch.cos(((grid / steps + offset) / (1 + offset)) * math.pi / 2).square()
    curve = curve / curve[0]
    return curve[1:].clamp(1e-6, 1.0 - 1e-6).float()


def q_sample(
    clean: torch.Tensor, timestep: torch.Tensor, alpha_bar: torch.Tensor, noise: torch.Tensor
) -> torch.Tensor:
    if clean.shape != noise.shape or timestep.shape != (len(clean),):
        raise ValueError("incompatible diffusion tensors")
    alpha = alpha_bar.to(clean.device)[timestep].view(-1, 1, 1)
    return alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise


def min_snr_noise_loss(
    prediction: torch.Tensor,
    target_noise: torch.Tensor,
    timestep: torch.Tensor,
    alpha_bar: torch.Tensor,
    gamma: float = 5.0,
) -> torch.Tensor:
    if prediction.shape != target_noise.shape:
        raise ValueError("prediction and noise shapes differ")
    alpha = alpha_bar.to(prediction.device)[timestep]
    snr = alpha / (1.0 - alpha).clamp_min(1e-8)
    weight = torch.minimum(snr, torch.full_like(snr, gamma)) / snr.clamp_min(1e-8)
    per_record = (prediction - target_noise).square().mean(dim=(1, 2))
    return (weight * per_record).mean()

