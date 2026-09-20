"""1D Deep Learning architectures for 12-lead raw ECG waveforms."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation1D(nn.Module):
    """Channel/Lead-wise Squeeze-and-Excitation block for 1D signals.
    
    Captures inter-lead electrophysiological relationships (e.g. reciprocal ST changes).
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        b, c, _ = x.size()
        w = x.mean(dim=2)  # Global average pooling -> (B, C)
        w = self.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w)).view(b, c, 1)
        return x * w


class ResidualBlock1D(nn.Module):
    """1D Residual block with batch normalization and SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        kernel_size: int = 7,
        use_se: bool = True,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=1, padding=padding, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SqueezeExcitation1D(out_channels) if use_se else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return self.relu(out)


class ECGResNet1D(nn.Module):
    """1D Deep Residual Network for 12-lead ECG Myocardial Infarction detection.
    
    Accepts raw 12-lead waveforms of shape (Batch, 12, Length) at 100 Hz.
    Outputs calibrated logits and extracted temporal embedding representations.
    """

    def __init__(
        self,
        in_channels: int = 12,
        base_filters: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # 4 Residual Stages with Squeeze-and-Excitation
        self.stage1 = nn.Sequential(
            ResidualBlock1D(base_filters, base_filters, stride=1),
            ResidualBlock1D(base_filters, base_filters, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock1D(base_filters, base_filters * 2, stride=2),
            ResidualBlock1D(base_filters * 2, base_filters * 2, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock1D(base_filters * 2, base_filters * 4, stride=2),
            ResidualBlock1D(base_filters * 4, base_filters * 4, stride=1),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock1D(base_filters * 4, base_filters * 8, stride=2),
            ResidualBlock1D(base_filters * 8, base_filters * 8, stride=1),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.embedding_layer = nn.Sequential(
            nn.Linear(base_filters * 8, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract continuous temporal embedding vector from raw ECG."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)
        return self.embedding_layer(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning (logit, embedding)."""
        embedding = self.extract_embedding(x)
        logit = self.classifier(embedding).squeeze(-1)
        return logit, embedding
