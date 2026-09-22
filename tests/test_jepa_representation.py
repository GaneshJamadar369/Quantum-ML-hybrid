import pytest

torch = pytest.importorskip("torch")

from aquire_preprocessing.models_jepa import ECGMaskedLatentPredictor
from aquire_preprocessing.models_transformer import ECGPatchTransformer
from run_jepa_transformer_representation_export import random_token_mask


def _small_encoder():
    return ECGPatchTransformer(
        patch_size=100, width=16, heads=2, layers=1,
        feedforward_width=24, embedding_dim=12,
        encoder_dropout=0.0, token_dropout=0.0, head_dropout=0.0,
    )


def test_jepa_loss_updates_online_but_not_target_by_gradient():
    torch.manual_seed(9)
    model = ECGMaskedLatentPredictor(_small_encoder())
    signal = torch.randn(4, 12, 1000)
    mask = random_token_mask(4, model.online_encoder.tokens, 0.6)
    before = next(model.target_encoder.parameters()).detach().clone()
    loss, components = model(signal, signal, mask)
    assert torch.isfinite(loss)
    assert set(components) == {
        "prediction_loss", "global_loss", "variance_loss", "mean_feature_std"
    }
    loss.backward()
    assert model.online_encoder.patch_projection.weight.grad is not None
    assert model.predictor[-1].weight.grad is not None
    assert model.online_encoder.embedding[0].weight.grad is not None
    assert model.online_encoder.classifier.weight.grad is None
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    optimizer = torch.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.01
    )
    optimizer.step()
    model.update_target(0.9)
    after = next(model.target_encoder.parameters()).detach()
    assert not torch.equal(before, after)


def test_jepa_encode_and_mask_contracts():
    model = ECGMaskedLatentPredictor(_small_encoder())
    signal = torch.randn(3, 12, 1000)
    mask = random_token_mask(3, model.online_encoder.tokens, 0.5)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [5, 5, 5]
    representation = model.encode(signal)
    assert representation.shape == (3, 12)
    assert torch.isfinite(representation).all()
    with pytest.raises(ValueError, match="mask shape"):
        model(signal, signal, torch.ones(3, 9, dtype=torch.bool))
