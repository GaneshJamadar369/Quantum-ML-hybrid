import pytest

torch = pytest.importorskip("torch")
from aquire_preprocessing.models_1d import ECGResNet1D
from aquire_preprocessing.models_hybrid import ECGMultimodalHybrid


def test_ecg_resnet_1d_forward_pass():
    batch_size = 4
    channels = 12
    length = 1000  # 10s @ 100Hz

    model = ECGResNet1D(in_channels=channels, base_filters=16, embedding_dim=64)
    model.eval()

    x = torch.randn(batch_size, channels, length)
    with torch.no_grad():
        logits, emb = model(x)

    assert logits.shape == (batch_size,)
    assert emb.shape == (batch_size, 64)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(emb).all()


def test_ecg_resnet_1d_gradient_flow():
    model = ECGResNet1D(in_channels=12, base_filters=16, embedding_dim=64)
    model.train()

    x = torch.randn(4, 12, 1000)
    target = torch.tensor([1.0, 0.0, 1.0, 0.0])

    logits, _ = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    # Verify gradients exist in stem and stages
    assert model.stem[0].weight.grad is not None
    assert model.classifier.weight.grad is not None


def test_ecg_multimodal_hybrid_forward_pass():
    batch_size = 4
    channels = 12
    length = 1000
    num_features = 106

    model = ECGMultimodalHybrid(
        num_tabular_features=num_features,
        tabular_hidden=32,
        waveform_channels=channels,
        waveform_embedding_dim=64,
        fused_dim=64,
    )
    model.eval()

    raw_signal = torch.randn(batch_size, channels, length)
    tabular = torch.randn(batch_size, num_features)

    with torch.no_grad():
        logits, fused_emb = model(raw_signal, tabular)

    assert logits.shape == (batch_size,)
    assert fused_emb.shape == (batch_size, 64)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(fused_emb).all()


def test_ecg_multimodal_hybrid_backprop():
    model = ECGMultimodalHybrid(
        num_tabular_features=106,
        tabular_hidden=32,
        waveform_channels=12,
        waveform_embedding_dim=64,
        fused_dim=64,
    )
    model.train()

    raw_signal = torch.randn(4, 12, 1000)
    tabular = torch.randn(4, 106)
    target = torch.tensor([0.0, 1.0, 1.0, 0.0])

    logits, _ = model(raw_signal, tabular)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    assert model.tabular_encoder[0].weight.grad is not None
    assert model.tabular_encoder[0].weight.grad.abs().sum() > 0
    assert model.waveform_encoder.stem[0].weight.grad is not None
    assert model.waveform_encoder.stem[0].weight.grad.abs().sum() > 0
    assert model.fusion.gate[0].weight.grad is not None
    assert model.fusion.proj_tabular.weight.grad is not None
    assert model.fusion.proj_tabular.weight.grad.abs().sum() > 0
    assert model.head[-1].weight.grad is not None


def test_multimodal_output_depends_on_each_modality():
    """Regression test for the former single-key attention defect."""
    torch.manual_seed(3)
    model = ECGMultimodalHybrid(
        num_tabular_features=8,
        tabular_hidden=16,
        waveform_channels=12,
        waveform_embedding_dim=32,
        fused_dim=32,
        dropout=0.0,
    ).eval()
    signal = torch.randn(4, 12, 1000)
    tabular = torch.randn(4, 8)
    with torch.no_grad():
        baseline, _ = model(signal, tabular)
        changed_tabular, _ = model(signal, tabular + 2.0)
        changed_signal, _ = model(signal * 0.0, tabular)
    assert not torch.allclose(baseline, changed_tabular)
    assert not torch.allclose(baseline, changed_signal)
