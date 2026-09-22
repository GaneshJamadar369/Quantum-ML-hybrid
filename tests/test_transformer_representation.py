import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from aquire_preprocessing.models_transformer import ECGPatchTransformer
from run_transformer_representation_export import stratified_patient_inner_split


def test_patch_transformer_contract_and_gradient():
    torch.manual_seed(17)
    model = ECGPatchTransformer()
    assert model.tokens == 100
    assert model.width == 96
    assert len(model.encoder.layers) == 3
    assert model.encoder.layers[0].self_attn.num_heads == 4
    assert sum(parameter.numel() for parameter in model.parameters()) < 300_000
    assert not torch.equal(
        model.encoder.layers[0].self_attn.in_proj_weight,
        model.encoder.layers[1].self_attn.in_proj_weight,
    )
    logits, embedding = model(torch.randn(3, 12, 1000))
    assert logits.shape == (3,)
    assert embedding.shape == (3, 128)
    assert torch.isfinite(logits).all() and torch.isfinite(embedding).all()
    logits.square().mean().backward()
    assert model.patch_projection.weight.grad is not None
    assert model.encoder.layers[-1].self_attn.in_proj_weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_transformer_rejects_bad_ecg_shape_and_nonfinite_input():
    model = ECGPatchTransformer()
    with pytest.raises(ValueError, match="Expected"):
        model(torch.randn(2, 12, 999))
    bad = torch.zeros(2, 12, 1000)
    bad[0, 1, 5] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        model(bad)


def test_inner_split_is_patient_stratified_and_reproducible():
    patients = np.repeat(np.arange(100), 2)
    labels = np.repeat(np.arange(100) % 2, 2)
    frame = pd.DataFrame({"patient_id": patients, "mi_label": labels})
    outer = np.arange(len(frame))
    fit_a, val_a = stratified_patient_inner_split(frame, outer, seed=32)
    fit_b, val_b = stratified_patient_inner_split(frame, outer, seed=32)
    assert np.array_equal(fit_a, fit_b) and np.array_equal(val_a, val_b)
    assert not set(frame.iloc[fit_a].patient_id) & set(frame.iloc[val_a].patient_id)
    assert len(set(frame.iloc[val_a].patient_id)) == 10
    assert frame.iloc[val_a].mi_label.nunique() == 2
