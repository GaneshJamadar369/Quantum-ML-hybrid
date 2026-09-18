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
    extractor: str = "aquire-local-v0.4.0"
    failures: List[str] = field(default_factory=list)


def _r_peaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """Fallback polarity-aware detector; validated extraction uses NeuroKit2."""
    centered = signal - np.median(signal)
    envelope = np.abs(centered)
    prominence = max(float(np.median(np.abs(centered - np.median(centered)))) * 2.0, 0.02)
    peaks, _ = find_peaks(envelope, distance=max(1, int(0.30 * fs)), prominence=prominence)
    return peaks


def _wave_array(waves: dict, key: str, length: int) -> np.ndarray:
    values = np.asarray(waves.get(key, []), dtype=float)
    result = np.full(length, np.nan, dtype=float)
    result[:min(length, len(values))] = values[:length]
    return result


def _detect_delineated_beats(
    signal_mv: np.ndarray,
    sample_mask: np.ndarray,
    lead_mask: np.ndarray,
    fs: int,
    failures: List[str],
    require_delineation: bool,
) -> tuple[np.ndarray, dict, Optional[int]]:
    preferred = CANONICAL_LEAD_ORDER.index("II")
    candidates = [preferred] + [
        index for index in range(len(CANONICAL_LEAD_ORDER)) if index != preferred
    ]
    candidates = [
        index for index in candidates
        if lead_mask[index] and sample_mask[index].mean() > 0.95
    ]
    try:
        import neurokit2 as nk  # type: ignore
    except ImportError as exc:
        failures.append("neurokit2_not_installed")
        if require_delineation:
            raise RuntimeError("NeuroKit2 is required for validated feature extraction") from exc
        if not candidates:
            return np.array([], dtype=int), {}, None
        failures.append("unvalidated_fallback_r_peaks")
        return _r_peaks(signal_mv[candidates[0]], fs), {}, candidates[0]

    for lead_index in candidates:
        try:
            cleaned = nk.ecg_clean(signal_mv[lead_index], sampling_rate=fs, method="neurokit")
            _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
            peaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
            if len(peaks) < 2:
                continue
            rr_ms = np.diff(peaks) * 1000.0 / fs
            plausible = (rr_ms >= 300.0) & (rr_ms <= 2000.0)
            if plausible.mean() < 0.8:
                continue
            _, waves = nk.ecg_delineate(
                cleaned, peaks, sampling_rate=fs, method="dwt", show=False
            )
            onset = _wave_array(waves, "ECG_R_Onsets", len(peaks))
            offset = _wave_array(waves, "ECG_R_Offsets", len(peaks))
            delineated = np.isfinite(onset) & np.isfinite(offset) & (offset > onset)
            if delineated.mean() < 0.5:
                failures.append(f"delineation_lead_{CANONICAL_LEAD_ORDER[lead_index]}:low_coverage")
                continue
            return peaks, waves, lead_index
        except Exception as exc:  # try the next valid lead before failing
            failures.append(f"delineation_lead_{CANONICAL_LEAD_ORDER[lead_index]}:{type(exc).__name__}")
    failures.append("validated_delineation_failed")
    if require_delineation:
        return np.array([], dtype=int), {}, None
    if candidates:
        failures.append("unvalidated_fallback_r_peaks")
        return _r_peaks(signal_mv[candidates[0]], fs), {}, candidates[0]
    return np.array([], dtype=int), {}, None


def extract_deployable_features(
    signal_mv: np.ndarray,
    fs: int,
    ecg_id: int,
    sample_mask: Optional[np.ndarray] = None,
    lead_mask: Optional[np.ndarray] = None,
    require_delineation: bool = False,
) -> FeatureBundle:
    """Extract deterministic features available from a new waveform.

    A common delineated beat time-base is used across all leads. This avoids
    treating S waves or T waves as separate beats and measures ST60 relative to
    a beat-specific pre-QRS baseline and a delineated QRS offset.
    """
    signal_mv = np.asarray(signal_mv, dtype=float)
    sample_mask = np.ones_like(signal_mv, dtype=bool) if sample_mask is None else np.asarray(sample_mask, bool)
    lead_mask = sample_mask.any(axis=1) if lead_mask is None else np.asarray(lead_mask, bool)
    values: Dict[str, float] = {}
    failures: List[str] = []
    values.update({
        key: np.nan for key in [
            "pr_interval_ms", "qrs_duration_ms", "qt_interval_ms",
            "qtc_bazett_ms", "qtc_fridericia_ms", "qtc_framingham_ms",
        ]
    })

    peaks, waves, peak_lead = _detect_delineated_beats(
        signal_mv, sample_mask, lead_mask, fs, failures, require_delineation
    )
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

    fiducial_onset = _wave_array(waves, "ECG_R_Onsets", len(peaks))
    fiducial_offset = _wave_array(waves, "ECG_R_Offsets", len(peaks))
    p_onset = _wave_array(waves, "ECG_P_Onsets", len(peaks))
    t_offset = _wave_array(waves, "ECG_T_Offsets", len(peaks))

    interval_specs = [
        ("pr_interval_ms", p_onset, fiducial_onset, 80.0, 400.0),
        ("qrs_duration_ms", fiducial_onset, fiducial_offset, 40.0, 250.0),
        ("qt_interval_ms", fiducial_onset, t_offset, 200.0, 700.0),
    ]
    for feature, left, right, lower, upper in interval_specs:
        interval_ms = (right - left) * 1000.0 / fs
        valid_interval = (
            np.isfinite(left) & np.isfinite(right)
            & (interval_ms >= lower) & (interval_ms <= upper)
        )
        values[feature] = (
            float(np.median(interval_ms[valid_interval]))
            if valid_interval.any() else np.nan
        )
    qt = values["qt_interval_ms"]
    rr = values.get("rr_median_ms", np.nan)
    if np.isfinite(qt) and np.isfinite(rr) and rr > 0:
        rr_seconds = rr / 1000.0
        values["qtc_bazett_ms"] = float(qt / np.sqrt(rr_seconds))
        values["qtc_fridericia_ms"] = float(qt / np.cbrt(rr_seconds))
        values["qtc_framingham_ms"] = float(qt + 154.0 * (1.0 - rr_seconds))

    for index, lead in enumerate(CANONICAL_LEAD_ORDER):
        valid = sample_mask[index] & np.isfinite(signal_mv[index])
        x = signal_mv[index, valid]
        prefix = lead.lower()
        if not lead_mask[index] or x.size < max(10, fs):
            for suffix in ["range_mv", "r_amp_mv", "s_amp_mv", "rs_ratio", "st60_mv", "t_polarity", "valid_fraction"]:
                values[f"{prefix}__{suffix}"] = np.nan
            continue
        values[f"{prefix}__range_mv"] = float(np.ptp(x))
        values[f"{prefix}__valid_fraction"] = float(valid.mean())
        r_values: List[float] = []
        s_values: List[float] = []
        st_values: List[float] = []
        t_values: List[float] = []
        for beat_index, peak in enumerate(peaks):
            # Approximate windows are allowed only for morphology amplitudes in
            # the explicitly flagged non-validated fallback path. They never
            # create PR/QRS/QT interval values.
            qrs_start = int(round(fiducial_onset[beat_index])) if np.isfinite(fiducial_onset[beat_index]) else peak - int(round(0.05 * fs))
            qrs_stop = int(round(fiducial_offset[beat_index])) if np.isfinite(fiducial_offset[beat_index]) else peak + int(round(0.08 * fs))
            if qrs_start < int(0.20 * fs) or qrs_stop <= qrs_start or qrs_stop >= signal_mv.shape[1]:
                continue
            baseline_start = qrs_start - int(round(0.20 * fs))
            baseline_stop = qrs_start - int(round(0.08 * fs))
            baseline_samples = signal_mv[index, baseline_start:baseline_stop]
            baseline_mask = sample_mask[index, baseline_start:baseline_stop]
            if not baseline_mask.any():
                continue
            baseline = float(np.median(baseline_samples[baseline_mask]))
            if not sample_mask[index, qrs_start:qrs_stop + 1].all():
                continue
            segment = signal_mv[index, qrs_start:qrs_stop + 1]
            # ECGDeli's per-lead R amplitude is signed at the common R
            # fiducial. Sampling the fiducial also preserves dominant-S
            # morphology in V1-V3; taking the positive maximum incorrectly
            # turned those leads into large positive R waves.
            r_values.append(float(signal_mv[index, peak] - baseline))
            s_values.append(float(np.min(segment) - baseline))
            st_idx = qrs_stop + int(round(0.06 * fs))
            if st_idx < signal_mv.shape[1] and sample_mask[index, st_idx]:
                st_values.append(float(signal_mv[index, st_idx] - baseline))
            t_start = qrs_stop + int(round(0.08 * fs))
            t_stop = min(qrs_stop + int(round(0.40 * fs)), signal_mv.shape[1])
            if beat_index + 1 < len(peaks):
                next_onset = fiducial_onset[beat_index + 1]
                next_qrs = int(next_onset) if np.isfinite(next_onset) else int(peaks[beat_index + 1])
                t_stop = min(t_stop, next_qrs - int(round(0.05 * fs)))
            if t_stop > t_start and sample_mask[index, t_start:t_stop].all():
                t_segment = signal_mv[index, t_start:t_stop] - baseline
                t_values.append(float(t_segment[np.argmax(np.abs(t_segment))]))
        r_amp = float(np.median(r_values)) if r_values else np.nan
        s_amp = float(np.median(s_values)) if s_values else np.nan
        values[f"{prefix}__r_amp_mv"] = r_amp
        values[f"{prefix}__s_amp_mv"] = s_amp
        values[f"{prefix}__rs_ratio"] = float(abs(r_amp) / max(abs(s_amp), 1e-6)) if np.isfinite(r_amp) and np.isfinite(s_amp) else np.nan
        values[f"{prefix}__st60_mv"] = float(np.median(st_values)) if st_values else np.nan
        t_value = float(np.median(t_values)) if t_values else np.nan
        values[f"{prefix}__t_polarity"] = float(np.sign(t_value)) if np.isfinite(t_value) and abs(t_value) > 0.01 else 0.0

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
