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
