"""Safe parsing of the three prototype ECG upload formats."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np


CANONICAL_LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


class ECGParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedECG:
    signal_mv: np.ndarray
    sampling_rate_hz: int
    lead_order: tuple[str, ...]
    source_format: str

    def __post_init__(self) -> None:
        if self.signal_mv.shape != (12, 1000):
            raise ECGParseError(f"Expected signal shape (12, 1000), received {self.signal_mv.shape}")
        if self.sampling_rate_hz != 100:
            raise ECGParseError(f"Expected 100 Hz, received {self.sampling_rate_hz} Hz")
        if self.lead_order != CANONICAL_LEADS:
            raise ECGParseError("Leads are not in canonical order")
        if not np.isfinite(self.signal_mv).all():
            raise ECGParseError("Signal contains NaN or infinity")

    @property
    def checksum(self) -> str:
        canonical = np.ascontiguousarray(self.signal_mv, dtype="<f4")
        return hashlib.sha256(canonical.tobytes()).hexdigest()


def infer_format(filename: str | None, content_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".zip" or content_type in {"application/zip", "application/x-zip-compressed"}:
        return "wfdb_zip"
    if suffix == ".csv" or content_type in {"text/csv", "application/csv"}:
        return "csv"
    if suffix == ".json" or content_type == "application/json":
        return "json"
    raise ECGParseError("Supported uploads are .csv, .json, or a WFDB .zip")


def parse_csv(payload: bytes) -> ParsedECG:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ECGParseError("CSV must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != list(CANONICAL_LEADS):
        raise ECGParseError(f"CSV columns must exactly equal {list(CANONICAL_LEADS)}")
    rows: list[list[float]] = []
    try:
        for row in reader:
            rows.append([float(row[lead]) for lead in CANONICAL_LEADS])
    except (TypeError, ValueError) as error:
        raise ECGParseError("CSV contains a non-numeric ECG value") from error
    if len(rows) != 1000:
        raise ECGParseError(f"CSV must contain exactly 1000 samples, found {len(rows)}")
    return ParsedECG(np.asarray(rows, dtype=np.float32).T, 100, CANONICAL_LEADS, "csv")


def parse_json(payload: bytes) -> ParsedECG:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ECGParseError("Invalid JSON ECG payload") from error
    if isinstance(value, dict):
        signal = value.get("signal_mv")
        sampling_rate = int(value.get("sampling_rate_hz", 100))
        leads = tuple(value.get("lead_order", CANONICAL_LEADS))
    else:
        signal, sampling_rate, leads = value, 100, CANONICAL_LEADS
    try:
        array = np.asarray(signal, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ECGParseError("JSON signal must be a numeric [12][1000] array") from error
    return ParsedECG(array, sampling_rate, leads, "json")


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = [member for member in archive.infolist() if not member.is_dir()]
    if not 1 <= len(members) <= 4:
        raise ECGParseError("WFDB ZIP must contain one .hea/.dat pair")
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ECGParseError("WFDB ZIP contains an unsafe path")
        if member.file_size > 5 * 1024 * 1024:
            raise ECGParseError("WFDB ZIP member exceeds the upload limit")
    return members


def parse_wfdb_zip(payload: bytes) -> ParsedECG:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ECGParseError("Invalid WFDB ZIP") from error
    with archive:
        members = _safe_members(archive)
        names = [Path(member.filename) for member in members]
        headers = [name for name in names if name.suffix.lower() == ".hea"]
        data_files = [name for name in names if name.suffix.lower() == ".dat"]
        if len(headers) != 1 or len(data_files) != 1 or headers[0].stem != data_files[0].stem:
            raise ECGParseError("WFDB ZIP must contain one matching .hea and .dat pair")
        with tempfile.TemporaryDirectory(prefix="aquire-wfdb-") as temp:
            archive.extractall(temp, members=members)
            try:
                import wfdb

                record = wfdb.rdrecord(str(Path(temp) / headers[0].stem), physical=True)
            except Exception as error:
                raise ECGParseError(f"Cannot read WFDB record: {error}") from error
    signal = np.asarray(record.p_signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ECGParseError("WFDB record has an invalid signal matrix")
    names = tuple(record.sig_name)
    if set(names) != set(CANONICAL_LEADS):
        raise ECGParseError(f"WFDB leads must equal {list(CANONICAL_LEADS)}")
    order = [names.index(lead) for lead in CANONICAL_LEADS]
    return ParsedECG(signal[:, order].T, int(record.fs), CANONICAL_LEADS, "wfdb_zip")


def parse_upload(payload: bytes, source_format: str) -> ParsedECG:
    if source_format == "csv":
        return parse_csv(payload)
    if source_format == "json":
        return parse_json(payload)
    if source_format == "wfdb_zip":
        return parse_wfdb_zip(payload)
    raise ECGParseError(f"Unsupported source format: {source_format}")
