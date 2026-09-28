from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from apps.api.aquire_api.ecg_parser import CANONICAL_LEADS
from apps.api.aquire_api.main import create_app
from apps.api.aquire_api.settings import Settings


def _csv_payload() -> bytes:
    stream = io.StringIO()
    stream.write(",".join(CANONICAL_LEADS) + "\n")
    signal = np.zeros((1000, 12), dtype=float)
    signal[:, 1] = np.sin(np.linspace(0, 20 * np.pi, 1000))
    for row in signal:
        stream.write(",".join(str(value) for value in row) + "\n")
    return stream.getvalue().encode()


def _client(tmp_path: Path) -> TestClient:
    app = create_app(Settings(bundle_root=tmp_path / "missing"))
    return TestClient(app)


def test_catalog_exposes_fixed_parallel_routes(tmp_path):
    with _client(tmp_path) as client:
        response = client.get("/api/v1/architecture")
        assert response.status_code == 200
        payload = response.json()
        assert payload["routing"] == "fixed_parallel_all_eligible_inputs"
        assert payload["routes"]["quantum"]["active"] is True
        assert payload["routes"]["classical"]["active"] is True
        assert payload["routes"]["fusion"]["active"] is True


def test_benchmark_catalog_explains_threshold_and_confusion_rates(tmp_path):
    with _client(tmp_path) as client:
        payload = client.get("/api/v1/benchmarks").json()
    assert payload["threshold_selection"]["frozen_threshold"] == 0.43531340285804376
    assert payload["threshold_selection"]["target_specificity"] == 0.9
    matrix = payload["confusion_matrix"]
    assert matrix["true_positives"]["count"] == 3398
    assert matrix["false_negatives"]["count"] == 970
    assert matrix["false_positives"]["rate"] == 0.1
    assert matrix["true_negatives"]["rate"] == 0.9


def test_signal_characteristics_are_conservative_and_finite():
    from apps.api.aquire_api.main import _signal_characteristics

    signal = np.linspace(-1.0, 1.0, 12_000, dtype=float).reshape(12, 1000)
    values = {
        "heart_rate_bpm": 75.0,
        "rr_median_ms": 800.0,
        "rr_iqr_ms": 30.0,
        "rr_cv": 0.04,
        "clinical__global_st_positive_count": 2.0,
        "clinical__global_st_negative_count": 1.0,
        "clinical__global_t_inversion_count": 3.0,
        "clinical__precordial_transition_lead": 3.0,
        "clinical__frontal_axis_proxy_deg": 42.0,
    }
    summary = _signal_characteristics(
        signal, values, [], list(values), quality_state="PASS", failed_leads=0, quality_issues=[]
    )
    assert summary["rhythm"]["heart_rate_context"] == "Within typical resting adult range"
    assert summary["st_t"]["st_positive_leads"] == 2.0
    assert summary["spatial"]["precordial_transition"] == "V3"
    assert summary["availability"]["intervals_omitted"] == ["PR", "QRS", "QT", "QTc"]


def test_csv_inspection_returns_canonical_preview(tmp_path):
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/ecg/inspect",
            files={"file": ("sample.csv", _csv_payload(), "text/csv")},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["valid"] is True
        assert payload["input"]["shape"] == [12, 1000]
        assert payload["input"]["lead_order"] == list(CANONICAL_LEADS)
        assert len(payload["input"]["preview"]["II"]) == 200


def test_prediction_never_falls_back_to_one_route(tmp_path):
    with _client(tmp_path) as client:
        response = client.post(
            "/api/v1/predictions",
            files={"file": ("sample.csv", _csv_payload(), "text/csv")},
        )
        assert response.status_code == 503
        detail = response.json()["detail"]
        assert detail["code"] == "MODEL_BUNDLE_UNAVAILABLE"
        assert detail["context"]["both_routes_required"] is True


def test_ready_api_abstains_on_failed_signal_quality(tmp_path, monkeypatch):
    from apps.api.aquire_api import main as api_main

    class FakeBundle:
        model_version = "test"

    class FakeRuntime:
        threshold = 0.5

        def __init__(self, bundle):
            self.bundle = bundle

        def golden_self_test(self):
            return {"cases": 1, "max_abs_error": 0.0, "tolerance": 0.0002}

        def quantum_score(self, signal):
            raise AssertionError("Quantum route must not run after QC failure")

        classical_score = quantum_score

        def fusion_score(self, quantum, classical):
            raise AssertionError("Fusion must not run after QC failure")

        def calibrate(self, probability):
            raise AssertionError("Calibration must not run after QC failure")

    monkeypatch.setattr(api_main, "verify_bundle", lambda *args, **kwargs: FakeBundle())
    monkeypatch.setattr(api_main, "FrozenHybridBundle", FakeRuntime)
    app = api_main.create_app(Settings(bundle_root=tmp_path))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            files={"file": ("flat.csv", _csv_payload(), "text/csv")},
        )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "QC_ABSTENTION"
    assert detail["context"]["both_routes_executed"] is False
