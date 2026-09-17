"""Deployable ECG features and fold-local PTB-XL+ transformation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .config import CANONICAL_LEAD_ORDER, DEV_FOLDS
from .manifest import guard_fold_access


@dataclass
class FeatureBundle:
    ecg_id: int
    values: Dict[str, float]
    extractor: str = "aquire-local-v0.2.0"
    failures: List[str] = field(default_factory=list)


def _r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    centered = signal - np.median(signal)
    envelope = np.abs(centered)
    prominence = max(float(np.median(np.abs(centered - np.median(centered)))) * 2.0, 0.02)
    peaks, _ = find_peaks(envelope, distance=max(1, int(0.25 * fs)), prominence=prominence)
    return peaks


def extract_deployable_features(
    signal_mv: np.ndarray,
    fs: int,
    ecg_id: int,
    sample_mask: Optional[np.ndarray] = None,
    lead_mask: Optional[np.ndarray] = None,
) -> FeatureBundle:
    """Extract deterministic features available from a new waveform.

    Fiducial intervals are attempted with NeuroKit2 when installed. The core
    amplitude, RR, ST and quality features remain available without it.
    """
    signal_mv = np.asarray(signal_mv, dtype=float)
    sample_mask = np.ones_like(signal_mv, dtype=bool) if sample_mask is None else np.asarray(sample_mask, bool)
    lead_mask = sample_mask.any(axis=1) if lead_mask is None else np.asarray(lead_mask, bool)
    values: Dict[str, float] = {}
    failures: List[str] = []

    preferred = CANONICAL_LEAD_ORDER.index("II")
    candidates = [preferred] + [i for i in range(len(CANONICAL_LEAD_ORDER)) if i != preferred]
    peak_lead = next((i for i in candidates if lead_mask[i] and sample_mask[i].mean() > 0.95), None)
    peaks = np.array([], dtype=int)
    if peak_lead is not None:
        peaks = _r_peaks(signal_mv[peak_lead], fs)
    if len(peaks) >= 2:
        rr_ms = np.diff(peaks) * 1000.0 / fs
        values.update({
            "heart_rate_bpm": float(60000.0 / np.median(rr_ms)),
            "rr_median_ms": float(np.median(rr_ms)),
            "rr_iqr_ms": float(np.percentile(rr_ms, 75) - np.percentile(rr_ms, 25)),
            "rr_cv": float(np.std(rr_ms) / max(np.mean(rr_ms), 1e-6)),
        })
    else:
        failures.append("insufficient_r_peaks")
        values.update({k: np.nan for k in ["heart_rate_bpm", "rr_median_ms", "rr_iqr_ms", "rr_cv"]})

    for index, lead in enumerate(CANONICAL_LEAD_ORDER):
        valid = sample_mask[index] & np.isfinite(signal_mv[index])
        x = signal_mv[index, valid]
        prefix = lead.lower()
        if not lead_mask[index] or x.size < max(10, fs):
            for suffix in ["range_mv", "r_amp_mv", "s_amp_mv", "rs_ratio", "st60_mv", "t_polarity", "valid_fraction"]:
                values[f"{prefix}__{suffix}"] = np.nan
            continue
        baseline = float(np.median(x))
        values[f"{prefix}__range_mv"] = float(np.ptp(x))
        values[f"{prefix}__valid_fraction"] = float(valid.mean())
        lead_peaks = _r_peaks(signal_mv[index], fs)
        r_values: List[float] = []
        s_values: List[float] = []
        st_values: List[float] = []
        t_values: List[float] = []
        for peak in lead_peaks:
            qrs_start, qrs_stop = peak - int(0.04 * fs), peak + int(0.08 * fs)
            if qrs_start < 0 or qrs_stop >= signal_mv.shape[1]:
                continue
            segment = signal_mv[index, qrs_start:qrs_stop]
            r_values.append(float(np.max(segment) - baseline))
            s_values.append(float(np.min(segment) - baseline))
            st_idx = peak + int(0.14 * fs)
            if st_idx < signal_mv.shape[1]:
                st_values.append(float(signal_mv[index, st_idx] - baseline))
            t_start, t_stop = peak + int(0.18 * fs), peak + int(0.38 * fs)
            if t_stop < signal_mv.shape[1]:
                t_values.append(float(np.mean(signal_mv[index, t_start:t_stop]) - baseline))
        r_amp = float(np.median(r_values)) if r_values else np.nan
        s_amp = float(np.median(s_values)) if s_values else np.nan
        values[f"{prefix}__r_amp_mv"] = r_amp
        values[f"{prefix}__s_amp_mv"] = s_amp
        values[f"{prefix}__rs_ratio"] = float(abs(r_amp) / max(abs(s_amp), 1e-6)) if np.isfinite(r_amp) and np.isfinite(s_amp) else np.nan
        values[f"{prefix}__st60_mv"] = float(np.median(st_values)) if st_values else np.nan
        t_value = float(np.median(t_values)) if t_values else np.nan
        values[f"{prefix}__t_polarity"] = float(np.sign(t_value)) if np.isfinite(t_value) and abs(t_value) > 0.01 else 0.0

    # Optional research-quality delineation. Missing dependency is recorded,
    # never silently replaced with a database lookup.
    try:
        import neurokit2 as nk  # type: ignore
        if peak_lead is not None:
            cleaned = nk.ecg_clean(signal_mv[peak_lead], sampling_rate=fs, method="neurokit")
            _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
            rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            if len(rpeaks):
                _, waves = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="dwt")
                for feature, left, right in [
                    ("pr_interval_ms", "ECG_P_Onsets", "ECG_R_Onsets"),
                    ("qrs_duration_ms", "ECG_R_Onsets", "ECG_R_Offsets"),
                    ("qt_interval_ms", "ECG_R_Onsets", "ECG_T_Offsets"),
                ]:
                    a = np.asarray(waves.get(left, []), dtype=float)
                    b = np.asarray(waves.get(right, []), dtype=float)
                    valid = np.isfinite(a) & np.isfinite(b) & (b > a)
                    values[feature] = float(np.median((b[valid] - a[valid]) * 1000.0 / fs)) if valid.any() else np.nan
    except ImportError:
        failures.append("neurokit2_not_installed")
    except Exception as exc:
        failures.append(f"delineation_failed:{type(exc).__name__}")

    return FeatureBundle(ecg_id=int(ecg_id), values=values, failures=failures)


class FoldLocalTabularTransformer:
    """Median imputation + robust scaling + variance/correlation filtering."""

    def __init__(self, correlation_threshold: float = 0.95):
        self.correlation_threshold = correlation_threshold
        self.columns_: List[str] = []
        self.selected_columns_: List[str] = []
        self.medians_: Optional[pd.Series] = None
        self.centers_: Optional[pd.Series] = None
        self.scales_: Optional[pd.Series] = None
        self.fitted_folds_: List[int] = []

    def fit(
        self,
        frame: pd.DataFrame,
        folds: Sequence[int],
        allowed_folds: Sequence[int],
        labels: Optional[Sequence[int]] = None,
        max_features: int = 256,
    ) -> "FoldLocalTabularTransformer":
        guard_fold_access(allowed_folds, purpose="feature_selection")
        allowed = set(int(v) for v in allowed_folds)
        if not allowed or not allowed.issubset(set(DEV_FOLDS)):
            raise ValueError("Tabular transforms may be fitted only on development folds")
        mask = np.isin(np.asarray(folds), list(allowed))
        train = frame.loc[mask].select_dtypes(include=[np.number]).copy()
        self.columns_ = [c for c in train.columns if not any(token in c.lower() for token in ["label", "target", "diag", "scp", "report", "infarction"])]
        train = train[self.columns_].replace([np.inf, -np.inf], np.nan)
        self.medians_ = train.median()
        filled = train.fillna(self.medians_)
        self.centers_ = filled.median()
        self.scales_ = (filled.quantile(0.75) - filled.quantile(0.25)).replace(0, 1.0)
        scaled = (filled - self.centers_) / self.scales_
        nonconstant = [c for c in scaled.columns if float(scaled[c].var()) > 1e-12]
        corr = scaled[nonconstant].corr().abs()
        keep: List[str] = []
        for column in nonconstant:
            if not any(float(corr.at[column, existing]) >= self.correlation_threshold for existing in keep):
                keep.append(column)
        if labels is not None and len(keep) > max_features:
            try:
                from sklearn.feature_selection import f_classif
            except ImportError as exc:
                raise RuntimeError("scikit-learn is required for fold-local ANOVA selection") from exc
            y = np.asarray(labels)[mask]
            scores, _ = f_classif(scaled[keep].to_numpy(), y)
            scores = np.nan_to_num(scores, nan=-np.inf)
            chosen = np.argsort(scores)[-max_features:]
            keep = [keep[index] for index in sorted(chosen)]
        self.selected_columns_ = keep
        self.fitted_folds_ = sorted(allowed)
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.medians_ is None or self.centers_ is None or self.scales_ is None:
            raise RuntimeError("Transformer is not fitted")
        data = frame.reindex(columns=self.columns_).replace([np.inf, -np.inf], np.nan)
        data = data.fillna(self.medians_)
        scaled = (data - self.centers_) / self.scales_
        return scaled[self.selected_columns_].to_numpy(dtype=np.float32)
