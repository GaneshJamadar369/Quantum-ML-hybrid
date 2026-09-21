import numpy as np
import pytest

pytest.importorskip("pennylane")
pytest.importorskip("torch")

from run_qsvc_hqnn_screen import _kernel_scores, run_screen


def test_iqp_qsvc_and_matched_kernel_controls_return_finite_scores():
    rng = np.random.default_rng(13)
    train = rng.uniform(-1.2, 1.2, (24, 4)).astype(np.float32)
    validation = rng.uniform(-1.2, 1.2, (5, 4)).astype(np.float32)
    labels = np.tile([0, 1], 12)
    scores, audit = _kernel_scores(train, validation, labels)
    assert set(scores) == {"iqp_qsvc", "rbf_svc", "laplacian_svc"}
    assert all(score.shape == (5,) and np.isfinite(score).all() for score in scores.values())
    assert audit["kernel_diagnostics_128"]["negative_eigenvalue_count"] == 0


def test_screen_rejects_sealed_folds_before_reading_data(tmp_path):
    with pytest.raises(ValueError, match="development folds"):
        run_screen("qsvc", tmp_path, tmp_path / "missing.csv", tmp_path / "out", folds=(9,))
