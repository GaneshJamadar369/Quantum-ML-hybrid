"""Lazy HDF5 datasets; augmentation is applied only to training samples."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np


class HDF5ECGDataset:
    """Process-safe lazy reader for primary ECG arrays.

    The file handle is opened on first access in each worker. An augmentation
    callback is accepted only when ``training=True``.
    """

    def __init__(
        self,
        path: Path,
        indices: Optional[np.ndarray] = None,
        normalizer=None,
        augmentation: Optional[Callable[[np.ndarray, int, int], np.ndarray]] = None,
        training: bool = False,
    ):
        self.path = Path(path)
        self.indices = None if indices is None else np.asarray(indices, dtype=int)
        self.normalizer = normalizer
        self.augmentation = augmentation
        self.training = bool(training)
        if augmentation is not None and not self.training:
            raise ValueError("Augmentation is permitted only for a training dataset")
        self._handle = None

    def _h5(self):
        if self._handle is None:
            try:
                import h5py
            except ImportError as exc:
                raise RuntimeError("h5py is required for HDF5ECGDataset") from exc
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def __len__(self) -> int:
        if self.indices is not None:
            return int(len(self.indices))
        return int(self._h5()["ecg_id"].shape[0])

    def __getitem__(self, item: int) -> dict:
        index = int(self.indices[item]) if self.indices is not None else int(item)
        h5 = self._h5()
        signal = h5["accepted_signal"][index].astype(np.float32)
        ecg_id = int(h5["ecg_id"][index])
        fold = int(h5["strat_fold"][index])
        if self.normalizer is not None:
            signal = self.normalizer.transform(signal)
        if self.augmentation is not None:
            signal = self.augmentation(signal, ecg_id, fold)
        return {
            "signal": signal,
            "sample_mask": h5["sample_mask"][index].astype(bool),
            "lead_mask": h5["lead_mask"][index].astype(bool),
            "label": int(h5["mi_label"][index]),
            "ecg_id": ecg_id,
            "patient_id": int(h5["patient_id"][index]),
            "fold": fold,
        }

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self):
        self.close()
