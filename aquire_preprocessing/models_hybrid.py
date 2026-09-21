"""Multimodal Hybrid Neural Fusion combining Clinical Tabular Features with Raw Waveforms."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .models_1d import ECGResNet1D


class GatedCrossAttentionFusion(nn.Module):
    """Two-token self-attention followed by gated multimodal fusion.

    The previous implementation used one tabular query and exactly one waveform
    key/value.  Attention over a single key always has weight one, so its output
    was independent of the tabular query.  That made the alleged multimodal
    model a waveform-only model.  Here both projected modalities are tokens in
    the attention context, and the final gate directly receives both tokens.
    """

    def __init__(self, tabular_dim: int = 64, waveform_dim: int = 128, fused_dim: int = 128):
        super().__init__()
        self.proj_tabular = nn.Linear(tabular_dim, fused_dim)
        self.proj_waveform = nn.Linear(waveform_dim, fused_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=4, batch_first=True
        )
        
        # Gating network
        self.gate = nn.Sequential(nn.Linear(fused_dim * 2, fused_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, tab_emb: torch.Tensor, wave_emb: torch.Tensor) -> torch.Tensor:
        t_proj = self.proj_tabular(tab_emb)
        w_proj = self.proj_waveform(wave_emb)

        # (B, 2, D): each modality can now attend to the other and itself.
        tokens = torch.stack([t_proj, w_proj], dim=1)
        attended, _ = self.cross_attn(tokens, tokens, tokens, need_weights=False)
        t_context = attended[:, 0]
        w_context = attended[:, 1]

        # The direct projected tokens in the gate provide an explicit gradient
        # path from each modality even if attention initially becomes diffuse.
        gate_coeff = self.gate(torch.cat([t_proj, w_proj], dim=-1))
        fused = gate_coeff * t_context + (1.0 - gate_coeff) * w_context
        return self.norm(fused)


class ECGMultimodalHybrid(nn.Module):
    """Deep Multimodal Hybrid Network for 12-lead ECG Infarction Detection.
    
    Processes:
      1. Tabular clinical features (106 approved metrics) via MLP Encoder.
      2. 12-lead raw ECG waveforms via ECGResNet1D backbone.
      3. Combines representations via Gated Cross-Attention Fusion.
    """

    def __init__(
        self,
        num_tabular_features: int = 106,
        tabular_hidden: int = 64,
        waveform_channels: int = 12,
        waveform_embedding_dim: int = 128,
        fused_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        # Tabular Encoder
        self.tabular_encoder = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, tabular_hidden),
            nn.BatchNorm1d(tabular_hidden),
            nn.ReLU(inplace=True),
        )

        # Waveform Encoder
        self.waveform_encoder = ECGResNet1D(
            in_channels=waveform_channels,
            base_filters=32,
            embedding_dim=waveform_embedding_dim,
            dropout=dropout,
        )

        # Multimodal Fusion
        self.fusion = GatedCrossAttentionFusion(
            tabular_dim=tabular_hidden,
            waveform_dim=waveform_embedding_dim,
            fused_dim=fused_dim,
        )

        # Joint Classification Head
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        raw_signals: torch.Tensor,
        tabular_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass taking raw signal (B, 12, L) and tabular features (B, D)."""
        tab_emb = self.tabular_encoder(tabular_features)
        _, wave_emb = self.waveform_encoder(raw_signals)
        fused_emb = self.fusion(tab_emb, wave_emb)
        logit = self.head(fused_emb).squeeze(-1)
        return logit, fused_emb
