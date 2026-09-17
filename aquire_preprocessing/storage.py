"""Chunked, atomic HDF5 storage for processed ECG records."""

from __future__ import annotations

import os
import hashlib
from pathlib import Path

import numpy as np

from .contracts import ProcessedECG


class HDF5RecordWriter:
    def __init__(self, output: Path, n_samples: int, compression: str = "lzf"):
        try:
            import h5py  # type: ignore
        except ImportError as exc:
            raise RuntimeError("h5py is required for HDF5 output; install requirements.lock") from exc
        self.h5py = h5py
        self.output = Path(output)
        self.temp = self.output.with_suffix(self.output.suffix + ".tmp")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.output.exists():
            raise FileExistsError(
                f"Refusing to replace completed artifact {self.output}; move it before starting a new run"
            )
        self.handle = h5py.File(self.temp, "a")
        self.n_samples = int(n_samples)
        self.compression = compression
        self._ensure_datasets()

    def _ensure_datasets(self) -> None:
        h5 = self.handle
        specs = {
            "minimal_signal": ((12, self.n_samples), np.float32),
            "accepted_signal": ((12, self.n_samples), np.float32),
            "sample_mask": ((12, self.n_samples), np.bool_),
            "lead_mask": ((12,), np.bool_),
            "ecg_id": ((), np.int64),
            "patient_id": ((), np.int64),
            "mi_label": ((), np.int8),
            "strat_fold": ((), np.int8),
            "hard_negative": ((), np.bool_),
            "eligibility": ((), "S16"),
            "source_checksum": ((), "S64"),
            "accepted_checksum": ((), "S64"),
            "pipeline_version": ((), "S32"),
        }
        for name, (tail, dtype) in specs.items():
            if name in h5:
                if h5[name].shape[1:] != tail:
                    raise ValueError(
                        f"Cannot resume {self.temp}: dataset {name} has shape "
                        f"{h5[name].shape[1:]}, expected {tail}"
                    )
                continue
            shape = (0,) + tail
            maxshape = (None,) + tail
            chunks = (1,) + tail if tail else (1024,)
            kwargs = {"compression": self.compression} if tail else {}
            h5.create_dataset(name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=dtype, **kwargs)
        committed = int(h5.attrs.get("committed_count", min(ds.shape[0] for ds in h5.values())))
        for dataset in h5.values():
            if dataset.shape[0] != committed:
                dataset.resize(committed, axis=0)
        h5.attrs["committed_count"] = committed
        h5.flush()

    @property
    def count(self) -> int:
        return int(self.handle.attrs["committed_count"])

    def append(self, record: ProcessedECG) -> int:
        index = self.count
        payload = {
            "minimal_signal": record.minimal_signal,
            "accepted_signal": record.accepted_signal,
            "sample_mask": record.sample_mask,
            "lead_mask": record.lead_mask,
            "ecg_id": record.ecg_id,
            "patient_id": record.patient_id,
            "mi_label": record.mi_label,
            "strat_fold": record.strat_fold,
            "hard_negative": record.hard_negative,
            "eligibility": record.eligibility.value.encode(),
            "source_checksum": record.provenance.get("source_checksum", "").encode(),
            "accepted_checksum": hashlib.sha256(
                np.ascontiguousarray(record.accepted_signal).tobytes()
            ).hexdigest().encode(),
            "pipeline_version": str(record.provenance.get("pipeline_version", "")).encode(),
        }
        for name, value in payload.items():
            dataset = self.handle[name]
            dataset.resize(index + 1, axis=0)
            dataset[index] = value
        self.handle.flush()
        self.handle.attrs["committed_count"] = index + 1
        self.handle.flush()
        return index

    def close(self, commit: bool = True) -> None:
        self.handle.close()
        if commit:
            os.replace(self.temp, self.output)

    def __enter__(self) -> "HDF5RecordWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(commit=exc_type is None)
