"""Compact patch Transformer for 10-second 12-lead ECG representation studies."""

from __future__ import annotations

import torch
from torch import nn


class ECGPatchTransformer(nn.Module):
    """Return an MI logit and a 128-dimensional fold-coherent ECG embedding.

    The model is intentionally small.  A 100 ms patch preserves all raw
    voltage samples in the patch before the learned projection.  Attention
    integrates patterns across the ten-second recording; mean/max pooling
    exposes both repeated and local responses to the classifier.
    """

    def __init__(
        self,
        in_channels: int = 12,
        signal_length: int = 1000,
        patch_size: int = 10,
        width: int = 96,
        heads: int = 4,
        layers: int = 3,
        feedforward_width: int = 192,
        embedding_dim: int = 128,
        encoder_dropout: float = 0.15,
        token_dropout: float = 0.10,
        head_dropout: float = 0.25,
    ) -> None:
        super().__init__()
        if in_channels != 12 or signal_length != 1000:
            raise ValueError("The frozen first screen requires 12 leads and 1000 samples")
        if patch_size < 1 or signal_length % patch_size:
            raise ValueError("patch_size must divide signal_length")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if min(encoder_dropout, token_dropout, head_dropout) < 0 or max(encoder_dropout, token_dropout, head_dropout) >= 1:
            raise ValueError("Dropout must be in [0, 1)")
        self.in_channels = in_channels
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.tokens = signal_length // patch_size
        self.width = width
        self.patch_projection = nn.Linear(in_channels * patch_size, width)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.tokens, width))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.token_norm = nn.LayerNorm(width)
        self.token_dropout = nn.Dropout(token_dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=feedforward_width,
            dropout=encoder_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        # TransformerEncoder clones the supplied layer, including its initial
        # values.  Reinitialize each clone independently to avoid identical
        # starting attention maps at every depth.
        for block in self.encoder.layers:
            for projection in (
                block.self_attn.out_proj,
                block.linear1,
                block.linear2,
            ):
                nn.init.xavier_uniform_(projection.weight)
                if projection.bias is not None:
                    nn.init.zeros_(projection.bias)
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)
        self.final_norm = nn.LayerNorm(width)
        self.embedding = nn.Sequential(
            nn.Linear(2 * width, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(head_dropout),
        )
        self.classifier = nn.Linear(embedding_dim, 1)

    def tokenize(self, signal: torch.Tensor) -> torch.Tensor:
        """Project a waveform to position-aware patch tokens."""
        if signal.ndim != 3 or tuple(signal.shape[1:]) != (self.in_channels, self.signal_length):
            raise ValueError(f"Expected (batch, 12, 1000), received {tuple(signal.shape)}")
        if not torch.isfinite(signal).all():
            raise ValueError("ECG input contains NaN or infinity")
        patches = signal.unfold(-1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(signal.shape[0], self.tokens, -1)
        return self.token_dropout(
            self.token_norm(self.patch_projection(patches) + self.position_embedding)
        )

    def encode_tokens(
        self,
        signal: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        mask_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode patches, optionally replacing selected tokens before attention."""
        tokens = self.tokenize(signal)
        if token_mask is not None:
            if token_mask.shape != tokens.shape[:2] or token_mask.dtype != torch.bool:
                raise ValueError("token_mask must be boolean with shape (batch, tokens)")
            if mask_token is None or mask_token.shape != (1, 1, self.width):
                raise ValueError(f"mask_token must have shape (1, 1, {self.width})")
            tokens = torch.where(token_mask.unsqueeze(-1), mask_token.expand_as(tokens), tokens)
        return self.final_norm(self.encoder(tokens))

    def pool_encoded(self, encoded: torch.Tensor) -> torch.Tensor:
        if encoded.ndim != 3 or tuple(encoded.shape[1:]) != (self.tokens, self.width):
            raise ValueError(
                f"Expected encoded tokens (batch, {self.tokens}, {self.width}), "
                f"received {tuple(encoded.shape)}"
            )
        pooled = torch.cat((encoded.mean(dim=1), encoded.amax(dim=1)), dim=1)
        return self.embedding(pooled)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode_tokens(signal)
        representation = self.pool_encoded(encoded)
        logits = self.classifier(representation).squeeze(-1)
        return logits, representation
