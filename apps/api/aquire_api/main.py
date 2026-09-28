"""FastAPI entrypoint for the AQUIRE-Med hackathon prototype."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from aquire_preprocessing.prototype_bundle import BundleError, VerifiedBundle, verify_bundle
from aquire_preprocessing.production_bundle import FrozenHybridBundle
from aquire_preprocessing.quality import assess_quality

from .catalog import ARCHITECTURE, BENCHMARKS, MODEL_CARD
from .ecg_parser import CANONICAL_LEADS, ECGParseError, infer_format, parse_upload
from .inference import FixedParallelHybrid, HybridInferenceError
from .schemas import HealthResponse, InputSummary, InspectionResponse
from .settings import Settings


def _preview(signal: np.ndarray, points: int = 200) -> dict[str, list[float]]:
    indices = np.linspace(0, signal.shape[1] - 1, points, dtype=int)
    return {lead: signal[row, indices].astype(float).tolist() for row, lead in enumerate(CANONICAL_LEADS)}


async def _read_upload(upload: UploadFile, limit: int) -> bytes:
    payload = await upload.read(limit + 1)
    if len(payload) > limit:
        raise HTTPException(
            status_code=413,
            detail={"code": "UPLOAD_TOO_LARGE", "message": f"Maximum upload size is {limit} bytes"},
        )
    if not payload:
        raise HTTPException(status_code=422, detail={"code": "EMPTY_UPLOAD", "message": "The upload is empty"})
    return payload


def _inspect(payload: bytes, upload: UploadFile) -> InspectionResponse:
    try:
        source_format = infer_format(upload.filename, upload.content_type)
        parsed = parse_upload(payload, source_format)
    except ECGParseError as error:
        return InspectionResponse(valid=False, errors=[str(error)])
    signal = parsed.signal_mv
    summary = InputSummary(
        format=parsed.source_format,
        sampling_rate_hz=parsed.sampling_rate_hz,
        shape=signal.shape,
        lead_order=list(parsed.lead_order),
        signal_sha256=parsed.checksum,
        finite_fraction=float(np.isfinite(signal).mean()),
        amplitude_min_mv=float(signal.min()),
        amplitude_max_mv=float(signal.max()),
        preview=_preview(signal),
    )
    warnings = []
    if float(np.max(np.abs(signal))) > 20.0:
        warnings.append("Amplitude exceeds the prototype inspection range; verify physical units")
    return InspectionResponse(valid=True, input=summary, warnings=warnings)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.bundle = None
        app.state.predictor = None
        app.state.bundle_error = None
        try:
            app.state.bundle = verify_bundle(
                settings.bundle_root,
                allow_uncalibrated=settings.allow_uncalibrated,
            )
            runtime = FrozenHybridBundle(app.state.bundle)
            app.state.golden_self_test = runtime.golden_self_test()
            app.state.predictor = FixedParallelHybrid(
                quantum_route=runtime.quantum_score,
                classical_route=runtime.classical_score,
                fusion_route=runtime.fusion_score,
                calibrator=runtime.calibrate,
                threshold=runtime.threshold,
            )
        except BundleError as error:
            app.state.bundle_error = str(error)
        except Exception as error:
            app.state.bundle_error = f"Bundle runtime initialization failed: {error}"
        yield

    app = FastAPI(
        title="AQUIRE-Med Hybrid QML API",
        version="0.1.0",
        description="Fixed parallel q4-VQC and morphology-HGB MI-pattern research prototype",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/api/v1/health/live", response_model=HealthResponse)
    async def live() -> HealthResponse:
        return HealthResponse(status="ok", model_ready=False)

    @app.get("/api/v1/health/ready", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
    async def ready(request: Request) -> HealthResponse:
        bundle: VerifiedBundle | None = request.app.state.bundle
        if bundle is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "model_ready": False,
                    "detail": request.app.state.bundle_error,
                },
            )
        return HealthResponse(status="ok", model_ready=True, model_version=bundle.model_version)

    @app.get("/api/v1/model-card")
    async def model_card():
        return MODEL_CARD

    @app.get("/api/v1/architecture")
    async def architecture():
        return ARCHITECTURE

    @app.get("/api/v1/benchmarks")
    async def benchmarks():
        return BENCHMARKS

    @app.post("/api/v1/ecg/inspect", response_model=InspectionResponse)
    async def inspect_ecg(file: UploadFile = File(...)) -> InspectionResponse:
        payload = await _read_upload(file, settings.max_upload_bytes)
        return _inspect(payload, file)

    @app.post("/api/v1/predictions")
    async def predict(request: Request, file: UploadFile = File(...)):
        payload = await _read_upload(file, settings.max_upload_bytes)
        inspection = _inspect(payload, file)
        if not inspection.valid:
            raise HTTPException(
                status_code=422,
                detail={"code": "INVALID_ECG", "message": "; ".join(inspection.errors)},
            )
        if request.app.state.bundle is None or request.app.state.predictor is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "MODEL_BUNDLE_UNAVAILABLE",
                    "message": "A verified hybrid bundle is required; no fallback single-route prediction is allowed",
                    "context": {"both_routes_required": True},
                },
            )
        source_format = infer_format(file.filename, file.content_type)
        parsed = parse_upload(payload, source_format)
        quality = assess_quality(parsed.signal_mv, ecg_id=-1, fs=100)
        if quality.qc_status == "FAIL":
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "QC_ABSTENTION",
                    "message": "The ECG failed the frozen quality policy; no prediction was produced",
                    "context": {
                        "quality_state": quality.qc_status,
                        "failed_leads": quality.n_failed_leads,
                        "issues": quality.summary_issues,
                        "both_routes_executed": False,
                    },
                },
            )
        try:
            result = request.app.state.predictor.predict(parsed.signal_mv)
        except HybridInferenceError as error:
            raise HTTPException(
                status_code=422,
                detail={"code": "HYBRID_INFERENCE_FAILED", "message": str(error)},
            ) from error
        return {
            "prediction": result.label,
            "mi_pattern_probability": result.calibrated_probability,
            "decision_threshold": result.threshold,
            "routes": {
                "quantum": {"active": True, "score": result.quantum_score},
                "classical": {"active": True, "score": result.classical_score},
            },
            "fusion": {
                "active": True,
                "raw_logit": result.fusion_logit,
                "uncalibrated_probability": result.fused_probability,
            },
            "model_version": request.app.state.bundle.model_version,
            "signal_sha256": parsed.checksum,
            "interpretation": "Research MI-pattern screening output; not a diagnosis or future-event risk estimate",
        }

    return app


app = create_app()
