from __future__ import annotations

import numpy as np
import pytest

from apps.api.aquire_api.inference import FixedParallelHybrid


def test_both_routes_run_before_fusion():
    calls = []

    def quantum(signal):
        calls.append("quantum")
        return 0.8

    def classical(signal):
        calls.append("classical")
        return -0.1

    def fusion(s_q, s_c):
        calls.append(("fusion", s_q, s_c))
        return 0.7, 0.668187772

    model = FixedParallelHybrid(
        quantum_route=quantum,
        classical_route=classical,
        fusion_route=fusion,
        calibrator=lambda value: value,
        threshold=0.6,
    )
    result = model.predict(np.zeros((12, 1000), dtype=np.float32))
    assert calls == ["quantum", "classical", ("fusion", 0.8, -0.1)]
    assert result.label == "MI_PATTERN"


def test_route_failure_never_uses_single_route_fallback():
    classical_called = False

    def broken_quantum(signal):
        raise RuntimeError("quantum route failed")

    def classical(signal):
        nonlocal classical_called
        classical_called = True
        return 0.2

    model = FixedParallelHybrid(
        quantum_route=broken_quantum,
        classical_route=classical,
        fusion_route=lambda s_q, s_c: (0.0, 0.5),
        calibrator=lambda value: value,
        threshold=0.5,
    )
    with pytest.raises(RuntimeError, match="quantum route failed"):
        model.predict(np.zeros((12, 1000), dtype=np.float32))
    assert classical_called is False
