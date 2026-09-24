"""Small multimodal fusion front ends for a four-qubit predictive core.

The modules deliberately end in four bounded angles.  They are representation
learners, not classifiers: a downstream VQC (or a matched classical ablation)
must turn the angles into a decision score.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def clinical_feature_groups(feature_names: Sequence[str]) -> list[list[int]]:
    """Create eight regional/rhythm groups that partition the manifest.

    The split follows ECG reading practice: rhythm, limb/precordial QRS, four
    regional ST/T groups, and a global/spatial group.  It is derived only from
    feature names and is therefore fixed before outcomes are inspected.
    """
    groups: list[list[int]] = [[] for _ in range(8)]
    limb_leads = {"i", "ii", "iii", "avr", "avl", "avf"}
    precordial_leads = {"v1", "v2", "v3", "v4", "v5", "v6"}
    qrs_suffixes = {"range_mv", "r_amp_mv", "s_amp_mv", "rs_ratio"}
    for index, name in enumerate(feature_names):
        lower = name.lower()
        if lower in {"heart_rate_bpm", "rr_median_ms", "rr_iqr_ms", "rr_cv"}:
            group = 0
        else:
            pieces = lower.split("__")
            lead = pieces[0]
            suffix = pieces[-1]
            if lead in limb_leads and suffix in qrs_suffixes:
                group = 1
            elif lead in precordial_leads and suffix in qrs_suffixes:
                group = 2
            elif lead in {"ii", "iii", "avf"} and suffix in {"st60_mv", "t_polarity"}:
                group = 3
            elif lead in {"i", "avl"} and suffix in {"st60_mv", "t_polarity"}:
                group = 4
            elif lead in {"v5", "v6"} and suffix in {"st60_mv", "t_polarity"}:
                group = 5
            elif lead in {"v1", "v2", "v3", "v4"} and suffix in {"st60_mv", "t_polarity"}:
                group = 6
            elif lower.startswith("clinical__inferior__"):
                group = 1 if suffix in {"r_mean_mv", "rs_median"} else 3
            elif lower.startswith("clinical__high_lateral__"):
                group = 1 if suffix in {"r_mean_mv", "rs_median"} else 4
            elif lower.startswith("clinical__lateral__"):
                group = 2 if suffix in {"r_mean_mv", "rs_median"} else 5
            elif lower.startswith("clinical__anterior__"):
                group = 2 if suffix in {"r_mean_mv", "rs_median"} else 6
            elif lower == "clinical__frontal_axis_proxy_deg":
                group = 1
            elif lower == "clinical__precordial_transition_lead":
                group = 2
            else:
                # aVR ST/T, reciprocity/contrast and global burden.
                group = 7
        groups[group].append(index)
    if any(not group for group in groups):
        raise ValueError("Every frozen clinical feature group must be non-empty")
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(len(feature_names))) or len(flattened) != len(set(flattened)):
        raise ValueError("Clinical groups must partition the feature manifest exactly")
    return groups


class ClinicalGroupTokenizer(nn.Module):
    """Encode eight predeclared clinical feature families as separate tokens."""

    def __init__(self, groups: Sequence[Sequence[int]], width: int = 32) -> None:
        super().__init__()
        self.groups = [tuple(int(index) for index in group) for group in groups]
        if not self.groups or any(not group for group in self.groups):
            raise ValueError("Clinical feature groups must be non-empty")
        self.feature_count = 1 + max(index for group in self.groups for index in group)
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * len(group), width),
                    nn.GELU(),
                    nn.LayerNorm(width),
                )
                for group in self.groups
            ]
        )
        self.group_embedding = nn.Parameter(torch.zeros(1, len(self.groups), width))
        nn.init.trunc_normal_(self.group_embedding, std=0.02)

    def forward(
        self, clinical: torch.Tensor, observed_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if clinical.ndim != 2 or clinical.shape[1] != self.feature_count:
            raise ValueError(
                f"Expected clinical shape (batch, {self.feature_count}), got {tuple(clinical.shape)}"
            )
        if not torch.isfinite(clinical).all():
            raise ValueError("Clinical input contains NaN or infinity")
        if observed_mask is None:
            observed_mask = torch.ones_like(clinical)
        if observed_mask.shape != clinical.shape or not torch.isfinite(observed_mask).all():
            raise ValueError("observed_mask must be finite and match clinical shape")
        tokens = [
            encoder(
                torch.cat(
                    [clinical[:, list(group)], observed_mask[:, list(group)].to(clinical.dtype)],
                    dim=-1,
                )
            )
            for encoder, group in zip(self.encoders, self.groups)
        ]
        return torch.stack(tokens, dim=1) + self.group_embedding


class _FusionBase(nn.Module):
    def __init__(self, groups: Sequence[Sequence[int]], width: int, dropout: float) -> None:
        super().__init__()
        self.clinical_tokens = ClinicalGroupTokenizer(groups, width=width)
        self.wave_projection = nn.Sequential(
            nn.Linear(96, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.representation = nn.Sequential(
            nn.Linear(2 * width, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Dropout(dropout),
        )
        self.to_angles = nn.Linear(width, 4)

    @staticmethod
    def _check_wave(wave_tokens: torch.Tensor) -> None:
        if wave_tokens.ndim != 3 or wave_tokens.shape[2] != 96:
            raise ValueError(f"Expected waveform tokens (batch, patches, 96), got {tuple(wave_tokens.shape)}")
        if not torch.isfinite(wave_tokens).all():
            raise ValueError("Waveform tokens contain NaN or infinity")

    def _finish(self, fused: torch.Tensor, *, return_embedding: bool) -> torch.Tensor:
        embedding = self.representation(fused)
        if return_embedding:
            return embedding
        # Narrow angles avoid saturating all rotations while preserving sign.
        return (torch.pi / 2.0) * torch.tanh(self.to_angles(embedding))


class FiLMQuantumInput(_FusionBase):
    """Clinical conditioning of every ECG patch through feature-wise affine modulation."""

    def __init__(self, groups: Sequence[Sequence[int]], width: int = 32, dropout: float = 0.15) -> None:
        super().__init__(groups, width, dropout)
        self.conditioner = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * width, 2 * width)
        )
        self.output_norm = nn.LayerNorm(2 * width)

    def forward(
        self, wave_tokens: torch.Tensor, clinical: torch.Tensor, *,
        observed_mask: torch.Tensor | None = None, return_embedding: bool = False
    ) -> torch.Tensor:
        self._check_wave(wave_tokens)
        wave = self.wave_projection(wave_tokens)
        clinical_pool = self.clinical_tokens(clinical, observed_mask).mean(dim=1)
        gamma, beta = self.conditioner(clinical_pool).chunk(2, dim=-1)
        conditioned = wave * (1.0 + 0.25 * torch.tanh(gamma).unsqueeze(1)) + beta.unsqueeze(1)
        fused = self.output_norm(torch.cat([conditioned.mean(dim=1), conditioned.amax(dim=1)], dim=-1))
        return self._finish(fused, return_embedding=return_embedding)


class LowRankBilinearQuantumInput(_FusionBase):
    """Low-rank multiplicative fusion without a full outer-product parameter tensor."""

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        width: int = 32,
        rank: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__(groups, width, dropout)
        if rank < 1:
            raise ValueError("rank must be positive")
        self.rank = int(rank)
        self.wave_factors = nn.Linear(2 * width, rank * width)
        self.clinical_factors = nn.Linear(2 * width, rank * width)
        self.rank_weights = nn.Parameter(torch.full((rank,), 1.0 / rank))
        self.fusion_norm = nn.LayerNorm(2 * width)

    def forward(
        self, wave_tokens: torch.Tensor, clinical: torch.Tensor, *,
        observed_mask: torch.Tensor | None = None, return_embedding: bool = False
    ) -> torch.Tensor:
        self._check_wave(wave_tokens)
        wave = self.wave_projection(wave_tokens)
        clinical_tokens = self.clinical_tokens(clinical, observed_mask)
        wave_pool = torch.cat([wave.mean(dim=1), wave.amax(dim=1)], dim=-1)
        clinical_pool = torch.cat([clinical_tokens.mean(dim=1), clinical_tokens.amax(dim=1)], dim=-1)
        wave_factor = self.wave_factors(wave_pool).view(-1, self.rank, wave.shape[-1])
        clinical_factor = self.clinical_factors(clinical_pool).view(-1, self.rank, wave.shape[-1])
        interaction = (wave_factor * clinical_factor * self.rank_weights.view(1, -1, 1)).sum(dim=1)
        fused = self.fusion_norm(torch.cat([interaction, wave.mean(dim=1) + clinical_tokens.mean(dim=1)], dim=-1))
        return self._finish(fused, return_embedding=return_embedding)


class ClinicalQueryCrossAttentionQuantumInput(_FusionBase):
    """Let clinical feature-family tokens retrieve relevant ECG time patches."""

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        width: int = 32,
        heads: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__(groups, width, dropout)
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2 * width, width), nn.Sigmoid())
        self.output_norm = nn.LayerNorm(2 * width)
        self.last_attention: torch.Tensor | None = None

    def forward(
        self, wave_tokens: torch.Tensor, clinical: torch.Tensor, *,
        observed_mask: torch.Tensor | None = None, return_embedding: bool = False
    ) -> torch.Tensor:
        self._check_wave(wave_tokens)
        wave = self.wave_projection(wave_tokens)
        queries = self.clinical_tokens(clinical, observed_mask)
        context, weights = self.attention(
            queries, wave, wave, need_weights=True, average_attn_weights=False
        )
        gate = self.gate(torch.cat([queries, context], dim=-1))
        attended = gate * context + (1.0 - gate) * queries
        fused = self.output_norm(torch.cat([attended.mean(dim=1), wave.amax(dim=1)], dim=-1))
        # Detached weights support explanation export without retaining a graph.
        self.last_attention = weights.detach()
        return self._finish(fused, return_embedding=return_embedding)


class CrossAttentionBilinearQuantumInput(_FusionBase):
    """Clinical-query patch attention followed by rank-four bilinear pooling."""

    def __init__(
        self,
        groups: Sequence[Sequence[int]],
        width: int = 32,
        heads: int = 4,
        rank: int = 4,
        dropout: float = 0.15,
    ) -> None:
        super().__init__(groups, width, dropout)
        if width % heads or rank < 1:
            raise ValueError("width must divide heads and rank must be positive")
        self.rank = int(rank)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(2 * width, width), nn.Sigmoid())
        self.attended_factors = nn.Linear(width, rank * width)
        self.wave_factors = nn.Linear(width, rank * width)
        self.rank_weights = nn.Parameter(torch.full((rank,), 1.0 / rank))
        self.output_norm = nn.LayerNorm(2 * width)
        self.last_attention: torch.Tensor | None = None

    def forward(
        self, wave_tokens: torch.Tensor, clinical: torch.Tensor, *,
        observed_mask: torch.Tensor | None = None, return_embedding: bool = False
    ) -> torch.Tensor:
        self._check_wave(wave_tokens)
        wave = self.wave_projection(wave_tokens)
        queries = self.clinical_tokens(clinical, observed_mask)
        context, weights = self.attention(
            queries, wave, wave, need_weights=True, average_attn_weights=False
        )
        gate = self.gate(torch.cat([queries, context], dim=-1))
        attended_pool = (queries + gate * context).mean(dim=1)
        wave_pool = wave.mean(dim=1)
        left = self.attended_factors(attended_pool).view(-1, self.rank, wave.shape[-1])
        right = self.wave_factors(wave_pool).view(-1, self.rank, wave.shape[-1])
        interaction = (left * right * self.rank_weights.view(1, -1, 1)).sum(dim=1)
        fused = self.output_norm(torch.cat([interaction, attended_pool + wave_pool], dim=-1))
        self.last_attention = weights.detach()
        return self._finish(fused, return_embedding=return_embedding)


def build_quantum_input_fusion(
    name: str,
    groups: Sequence[Sequence[int]],
    *,
    width: int = 32,
    dropout: float = 0.15,
) -> nn.Module:
    normalized = name.lower().replace("-", "_")
    if normalized == "film":
        return FiLMQuantumInput(groups, width=width, dropout=dropout)
    if normalized in {"lmf", "low_rank_bilinear"}:
        return LowRankBilinearQuantumInput(groups, width=width, rank=4, dropout=dropout)
    if normalized in {"cross_attention", "crossattn"}:
        return ClinicalQueryCrossAttentionQuantumInput(groups, width=width, heads=4, dropout=dropout)
    if normalized in {"cross_attention_lmf", "crossattn_lmf"}:
        return CrossAttentionBilinearQuantumInput(
            groups, width=width, heads=4, rank=4, dropout=dropout
        )
    raise ValueError(f"Unknown advanced fusion: {name}")


class ResidualAngleAdapter(nn.Module):
    """Learn a small correction while preserving a proven q4 representation.

    The final projection and gate start at zero, so the initial output is the
    supplied base angle vector.  This makes degradation an explicit learned
    choice rather than an unavoidable consequence of replacing h128.
    """

    def __init__(
        self,
        waveform_dim: int = 128,
        clinical_dim: int | None = None,
        hidden: int = 16,
        max_residual: float = 0.25,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden < 4 or not 0 < max_residual <= 1:
            raise ValueError("Invalid residual adapter capacity")
        self.clinical_dim = clinical_dim
        self.max_residual = float(max_residual)
        self.waveform = nn.Sequential(
            nn.LayerNorm(waveform_dim),
            nn.Linear(waveform_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if clinical_dim is not None:
            self.clinical = nn.Sequential(
                nn.LayerNorm(2 * clinical_dim),
                nn.Linear(2 * clinical_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            joined = 2 * hidden
        else:
            self.clinical = None
            joined = hidden
        self.delta = nn.Linear(joined, 4)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        base_angles: torch.Tensor,
        waveform: torch.Tensor,
        clinical: torch.Tensor | None = None,
        observed_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_angles.ndim != 2 or base_angles.shape[1] != 4:
            raise ValueError("base_angles must have shape (batch, 4)")
        encoded = [self.waveform(waveform)]
        if self.clinical is not None:
            if clinical is None or clinical.shape[1] != self.clinical_dim:
                raise ValueError("Clinical input is required and has the wrong width")
            if observed_mask is None:
                observed_mask = torch.ones_like(clinical)
            if observed_mask.shape != clinical.shape:
                raise ValueError("Clinical mask shape mismatch")
            encoded.append(self.clinical(torch.cat([clinical, observed_mask], dim=-1)))
        raw_delta = torch.tanh(self.delta(torch.cat(encoded, dim=-1)))
        gate = torch.sigmoid(self.gate_logit)
        # Work in the unconstrained latent of the tanh angle map so a small
        # correction behaves consistently near and away from the boundaries.
        scale = torch.pi / 2.0
        base_latent = torch.atanh(torch.clamp(base_angles / scale, -0.999, 0.999))
        corrected = scale * torch.tanh(base_latent + gate * self.max_residual * raw_delta)
        return corrected, corrected - base_angles


class OrthogonalQ4Mixer(nn.Module):
    """Six-parameter, information-preserving rotation of q4 latent angles."""

    def __init__(self) -> None:
        super().__init__()
        self.skew_parameters = nn.Parameter(torch.zeros(6))
        self.register_buffer(
            "pairs",
            torch.tensor([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]),
        )

    def orthogonal_matrix(self) -> torch.Tensor:
        skew = torch.zeros(4, 4, device=self.skew_parameters.device, dtype=self.skew_parameters.dtype)
        for value, pair in zip(self.skew_parameters, self.pairs):
            left, right = int(pair[0]), int(pair[1])
            skew[left, right] = value
            skew[right, left] = -value
        return torch.matrix_exp(skew)

    def forward(self, base_angles: torch.Tensor) -> torch.Tensor:
        if base_angles.ndim != 2 or base_angles.shape[1] != 4:
            raise ValueError("OrthogonalQ4Mixer expects (batch, 4)")
        scale = torch.pi / 2.0
        latent = torch.atanh(torch.clamp(base_angles / scale, -0.999, 0.999))
        rotated = latent @ self.orthogonal_matrix()
        return scale * torch.tanh(rotated)
