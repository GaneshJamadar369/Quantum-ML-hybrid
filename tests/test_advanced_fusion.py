import numpy as np
import pytest
import torch

from aquire_preprocessing.models_advanced_fusion import (
    ClinicalQueryCrossAttentionQuantumInput,
    build_quantum_input_fusion,
    clinical_feature_groups,
)


FEATURES = [
    "heart_rate_bpm", "rr_median_ms", "i__range_mv", "v2__range_mv",
    "ii__st60_mv", "i__t_polarity", "v5__st60_mv", "v2__t_polarity",
    "clinical__global_st_rms_mv",
]


def test_groups_are_an_exact_partition():
    groups = clinical_feature_groups(FEATURES)
    assert sorted(index for group in groups for index in group) == list(range(len(FEATURES)))
    assert all(groups)


@pytest.mark.parametrize("name", ["film", "lmf", "cross_attention", "cross_attention_lmf"])
def test_fusions_produce_bounded_finite_q4_with_gradients(name):
    torch.manual_seed(4)
    groups = clinical_feature_groups(FEATURES)
    model = build_quantum_input_fusion(name, groups, width=16, dropout=0.0)
    wave = torch.randn(5, 20, 96, requires_grad=True)
    clinical = torch.randn(5, len(FEATURES), requires_grad=True)
    angles = model(wave, clinical)
    embedding = model(wave, clinical, return_embedding=True)
    assert angles.shape == (5, 4)
    assert embedding.shape == (5, 16)
    assert torch.isfinite(angles).all()
    assert float(angles.detach().abs().max()) <= np.pi / 2 + 1e-6
    angles.square().mean().backward()
    assert wave.grad is not None and torch.isfinite(wave.grad).all()
    assert clinical.grad is not None and torch.isfinite(clinical.grad).all()


def test_cross_attention_depends_on_patch_content_and_has_nontrivial_weights():
    torch.manual_seed(8)
    groups = clinical_feature_groups(FEATURES)
    model = ClinicalQueryCrossAttentionQuantumInput(groups, width=16, heads=4, dropout=0.0).eval()
    clinical = torch.randn(3, len(FEATURES))
    wave = torch.randn(3, 12, 96)
    first = model(wave, clinical)
    weights = model.last_attention
    second = model(torch.roll(wave, shifts=1, dims=1) + 0.2 * torch.randn_like(wave), clinical)
    assert weights is not None and weights.shape == (3, 4, 8, 12)
    assert weights.std(dim=-1).mean() > 0
    assert not torch.allclose(first, second)


def test_invalid_inputs_are_rejected():
    groups = clinical_feature_groups(FEATURES)
    model = build_quantum_input_fusion("film", groups, width=16)
    with pytest.raises(ValueError):
        model(torch.randn(2, 10, 95), torch.randn(2, len(FEATURES)))
    bad = torch.randn(2, len(FEATURES))
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError):
        model(torch.randn(2, 10, 96), bad)
