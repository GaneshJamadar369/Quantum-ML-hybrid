import json
from pathlib import Path

import numpy as np
import torch

from aquire_preprocessing.models_quantum import (
    TorchStructuredReuploadingQuantumClassifier,
)
from aquire_preprocessing.structured_reuploading import (
    FoldLocalConceptEncoder,
    load_structured_reuploading_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _groups():
    approved = json.loads(
        (ROOT / "configs/approved_feature_manifest_v0_4.json").read_text()
    )["approved_features"]
    _, groups = load_structured_reuploading_config(
        ROOT / "configs/structured_reuploading_v1.json", approved
    )
    return approved, groups


def test_concept_groups_are_approved_and_disjoint():
    _, groups = _groups()
    flattened = [feature for group in groups for feature in group.features]
    assert len(groups) == 4
    assert len(flattened) == len(set(flattened))
    assert all(group.anchor_feature in group.features for group in groups)


def test_fold_local_concept_encoder_is_finite_and_label_free():
    approved, groups = _groups()
    rng = np.random.default_rng(8)
    values = rng.normal(size=(80, len(approved)))
    values[::9, approved.index("v1__st60_mv")] = np.nan
    encoder = FoldLocalConceptEncoder(groups, seed=12)
    train = encoder.fit_transform(values[:60], approved)
    validation = encoder.transform(values[60:])
    assert train.shape == (60, 4)
    assert validation.shape == (20, 4)
    assert np.isfinite(train).all() and np.isfinite(validation).all()
    assert np.max(np.abs(train)) <= np.pi / 2 + 1e-6
    assert len(encoder.audit()) == 4


def test_structured_statevector_is_normalized_differentiable_and_block_sensitive():
    torch.manual_seed(4)
    model = TorchStructuredReuploadingQuantumClassifier()
    x = torch.randn(6, 8, requires_grad=True)
    observables = model.quantum_observables(x)
    logits = model(x)
    assert observables.shape == (6, 8)
    assert logits.shape == (6,)
    assert torch.isfinite(logits).all()
    assert torch.max(torch.abs(observables)) <= 1.0 + 1e-5
    changed = x.detach().clone()
    changed[:, 4:] += 0.7
    assert not torch.allclose(logits.detach(), model(changed).detach())
    logits.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert model.feature_scales.grad is not None
