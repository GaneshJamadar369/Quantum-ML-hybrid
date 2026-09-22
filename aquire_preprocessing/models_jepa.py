"""Label-free masked latent prediction for compact 12-lead ECG representations."""

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from .models_transformer import ECGPatchTransformer


class ECGMaskedLatentPredictor(nn.Module):
    """JEPA-style online/EMA-target model with no diagnostic-label input.

    The online encoder sees masked patch tokens and predicts target-encoder
    latents at those positions. The target encoder sees the complete waveform,
    receives no gradients and is updated only by exponential moving average.
    """

    def __init__(self, encoder: ECGPatchTransformer | None = None) -> None:
        super().__init__()
        self.online_encoder = encoder or ECGPatchTransformer()
        self.target_encoder = deepcopy(self.online_encoder)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.online_encoder.width))
        nn.init.normal_(self.mask_token, std=0.02)
        width = self.online_encoder.width
        self.predictor = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        # The diagnostic classifier is outside the self-supervised path. The
        # pooled embedding head remains trainable through the global latent
        # objective below.
        for parameter in self.online_encoder.classifier.parameters():
            parameter.requires_grad_(False)
        self._freeze_target()

    def _freeze_target(self) -> None:
        self.target_encoder.eval()
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("EMA momentum must be in [0, 1)")
        for target, online in zip(
            self.target_encoder.parameters(), self.online_encoder.parameters(), strict=True
        ):
            target.lerp_(online.detach(), 1.0 - momentum)
        for target, online in zip(
            self.target_encoder.buffers(), self.online_encoder.buffers(), strict=True
        ):
            target.copy_(online)

    def forward(
        self, context_signal: torch.Tensor, target_signal: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if token_mask.ndim != 2 or token_mask.shape != (
            len(context_signal), self.online_encoder.tokens
        ):
            raise ValueError("Invalid JEPA token mask shape")
        if not token_mask.any(dim=1).all():
            raise ValueError("Every ECG must contain at least one masked token")
        online_tokens = self.online_encoder.encode_tokens(
            context_signal, token_mask=token_mask, mask_token=self.mask_token
        )
        predicted = self.predictor(online_tokens)
        self.target_encoder.eval()
        with torch.no_grad():
            target_tokens = self.target_encoder.encode_tokens(target_signal)
        masked_prediction = predicted[token_mask]
        masked_target = target_tokens[token_mask]
        prediction_loss = 2.0 - 2.0 * (
            F.normalize(masked_prediction, dim=-1)
            * F.normalize(masked_target, dim=-1)
        ).sum(dim=-1).mean()
        online_representation = self.online_encoder.pool_encoded(online_tokens)
        with torch.no_grad():
            target_representation = self.target_encoder.pool_encoded(target_tokens)
        global_loss = 2.0 - 2.0 * (
            F.normalize(online_representation, dim=-1)
            * F.normalize(target_representation, dim=-1)
        ).sum(dim=-1).mean()
        # A small VICReg-style variance term makes complete representation
        # collapse observable and directly penalized without diagnostic labels.
        feature_std = torch.sqrt(
            online_representation.var(dim=0, unbiased=False) + 1e-4
        )
        variance_loss = F.relu(1.0 - feature_std).mean()
        loss = prediction_loss + 0.25 * global_loss + 0.05 * variance_loss
        return loss, {
            "prediction_loss": prediction_loss.detach(),
            "global_loss": global_loss.detach(),
            "variance_loss": variance_loss.detach(),
            "mean_feature_std": feature_std.mean().detach(),
        }

    @torch.no_grad()
    def encode(self, signal: torch.Tensor) -> torch.Tensor:
        """Return the frozen EMA-target 128-dimensional representation."""
        self.target_encoder.eval()
        _, representation = self.target_encoder(signal)
        return representation
