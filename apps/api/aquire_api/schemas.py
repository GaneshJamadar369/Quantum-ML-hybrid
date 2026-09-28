"""Public response schemas for the prototype API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    model_ready: bool
    model_version: str | None = None
    detail: str | None = None


class InputSummary(BaseModel):
    format: Literal["csv", "json", "wfdb_zip"]
    sampling_rate_hz: int
    shape: tuple[int, int]
    lead_order: list[str]
    signal_sha256: str
    finite_fraction: float
    amplitude_min_mv: float
    amplitude_max_mv: float
    preview: dict[str, list[float]] = Field(default_factory=dict)


class InspectionResponse(BaseModel):
    valid: bool
    input: InputSummary | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ErrorDetail(BaseModel):
    code: str
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
