"""Clinical and statistical evidence gate for deployable ECG features.

This module deliberately separates three questions:

1. Is the development cohort and raw waveform suitable for analysis?
2. Can a candidate feature be measured reproducibly from a new ECG?
3. Does the feature add stable, patient-safe predictive information?

All selection-oriented analyses are restricted to PTB-XL development folds
1--8.  The module only recommends KEEP/REVIEW/REPAIR decisions; final removal
requires fold-local ablation evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
import zlib

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import CANONICAL_LEAD_ORDER, DEV_FOLDS
from .feature_validation import compare_local_to_reference
from .manifest import guard_fold_access


TERRITORIES: Mapping[str, Tuple[str, ...]] = {
    "inferior": ("ii", "iii", "avf"),
    "high_lateral": ("i", "avl"),
    "lateral": ("i", "avl", "v5", "v6"),
    "anterior": ("v1", "v2", "v3", "v4"),
}


@dataclass(frozen=True)
class EvidenceThresholds:
    """Prespecified triage thresholds; they do not directly delete features."""

    minimum_coverage: float = 0.60
    near_zero_variance: float = 1e-12
    redundancy_rho: float = 0.95
    minimum_fold_direction_consistency: float = 0.75
    minimum_effect_size: float = 0.10
    minimum_auc_distance: float = 0.05
    minimum_mutual_information: float = 0.005
    fdr_alpha: float = 0.05


def _atomic_csv(frame: pd.DataFrame, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(output)


def _atomic_json(payload: object, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str))
    temporary.replace(output)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values, preserving NaNs."""

    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    raw = values[valid]
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result[valid] = restored
    return result


def _lead_column(lead: str, suffix: str) -> str:
    return f"{lead.lower()}__{suffix}"


def _available(frame: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _row_slope(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    values = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x = np.arange(1, len(columns) + 1, dtype=float)
    slopes = np.full(len(values), np.nan, dtype=float)
    for index, row in enumerate(values):
        valid = np.isfinite(row)
        if valid.sum() >= 3:
            slopes[index] = float(np.polyfit(x[valid], row[valid], 1)[0])
    return pd.Series(slopes, index=frame.index)


def derive_clinical_composites(features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create explicit, interpretable multi-lead ECG pattern features.

    These are candidate measurements, not diagnostic rules. Their inclusion in
    a model is decided later by development-fold stability and ablation.
    """

    result = pd.DataFrame(index=features.index)
    registry: list[dict] = []

    def add(name: str, values: pd.Series, rationale: str, formula: str) -> None:
        result[name] = pd.to_numeric(values, errors="coerce")
        registry.append({
            "feature": name,
            "source": "clinically_derived",
            "clinical_rationale": rationale,
            "formula": formula,
            "status": "candidate_pending_ablation",
        })

    for territory, leads in TERRITORIES.items():
        st_columns = _available(features, [_lead_column(lead, "st60_mv") for lead in leads])
        t_columns = _available(features, [_lead_column(lead, "t_polarity") for lead in leads])
        r_columns = _available(features, [_lead_column(lead, "r_amp_mv") for lead in leads])
        rs_columns = _available(features, [_lead_column(lead, "rs_ratio") for lead in leads])
        if st_columns:
            st = features[st_columns].apply(pd.to_numeric, errors="coerce")
            add(f"clinical__{territory}__st_mean_mv", st.mean(axis=1),
                "Regional ST displacement is interpreted across anatomically related leads.",
                f"mean({', '.join(st_columns)})")
            add(f"clinical__{territory}__st_absmax_mv", st.abs().max(axis=1),
                "Captures the strongest regional ST deviation without assuming its direction.",
                f"max(abs({', '.join(st_columns)}))")
            add(f"clinical__{territory}__st_abnormal_count", (st.abs() >= 0.10).sum(axis=1),
                "MI patterns commonly involve multiple contiguous leads; 0.10 mV is an exploratory threshold.",
                f"count(abs(ST60) >= 0.10 mV in {territory})")
        if t_columns:
            t = features[t_columns].apply(pd.to_numeric, errors="coerce")
            add(f"clinical__{territory}__t_inversion_fraction", (t < 0).sum(axis=1) / t.notna().sum(axis=1).replace(0, np.nan),
                "Regional T-wave inversion can complement ST and QRS evidence.",
                f"fraction(T polarity < 0 in {territory})")
        if r_columns:
            add(f"clinical__{territory}__r_mean_mv", features[r_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1),
                "Regional R-wave amplitude summarizes depolarization across related leads.",
                f"mean({', '.join(r_columns)})")
        if rs_columns:
            add(f"clinical__{territory}__rs_median", features[rs_columns].apply(pd.to_numeric, errors="coerce").median(axis=1),
                "Regional R/S balance captures depolarization morphology.",
                f"median({', '.join(rs_columns)})")

    inferior = result.get("clinical__inferior__st_mean_mv")
    high_lateral = result.get("clinical__high_lateral__st_mean_mv")
    anterior = result.get("clinical__anterior__st_mean_mv")
    if inferior is not None and high_lateral is not None:
        add("clinical__inferior_high_lateral_st_reciprocity", inferior - high_lateral,
            "Inferior injury may be accompanied by reciprocal high-lateral ST change.",
            "inferior ST mean - high-lateral ST mean")
    if anterior is not None and inferior is not None:
        add("clinical__anterior_inferior_st_contrast", anterior - inferior,
            "Contrasts two anatomical territories without converting the value into a diagnosis.",
            "anterior ST mean - inferior ST mean")

    all_st = _available(features, [_lead_column(lead, "st60_mv") for lead in [x.lower() for x in CANONICAL_LEAD_ORDER]])
    if all_st:
        st = features[all_st].apply(pd.to_numeric, errors="coerce")
        add("clinical__global_st_rms_mv", np.sqrt((st ** 2).mean(axis=1)),
            "Summarizes the total magnitude of ST displacement across the 12-lead system.",
            "sqrt(mean(ST60_lead^2))")
        add("clinical__global_st_positive_count", (st >= 0.10).sum(axis=1),
            "Counts leads with exploratory positive ST displacement.",
            "count(ST60 >= 0.10 mV)")
        add("clinical__global_st_negative_count", (st <= -0.10).sum(axis=1),
            "Counts leads with exploratory negative ST displacement.",
            "count(ST60 <= -0.10 mV)")

    all_t = _available(features, [_lead_column(lead, "t_polarity") for lead in [x.lower() for x in CANONICAL_LEAD_ORDER]])
    if all_t:
        t = features[all_t].apply(pd.to_numeric, errors="coerce")
        add("clinical__global_t_inversion_count", (t < 0).sum(axis=1),
            "Counts the spatial extent of T-wave inversion.",
            "count(T polarity < 0)")

    precordial_r = [_lead_column(f"v{number}", "r_amp_mv") for number in range(1, 7)]
    if len(_available(features, precordial_r)) == 6:
        add("clinical__r_progression_slope_v1_v6", _row_slope(features, precordial_r),
            "Loss or reversal of normal R-wave progression can support anterior infarction patterns.",
            "least-squares slope of R amplitude from V1 to V6")
    precordial_ratio = [_lead_column(f"v{number}", "rs_ratio") for number in range(1, 7)]
    if len(_available(features, precordial_ratio)) == 6:
        ratios = features[precordial_ratio].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        transition = np.full(len(ratios), np.nan)
        for index, row in enumerate(ratios):
            candidates = np.where(np.isfinite(row) & (row >= 1.0))[0]
            if len(candidates):
                transition[index] = float(candidates[0] + 1)
        add("clinical__precordial_transition_lead", pd.Series(transition, index=features.index),
            "The lead where R becomes at least as large as S summarizes precordial transition.",
            "first V lead with R/S >= 1")

    net_i_columns = [_lead_column("i", "r_amp_mv"), _lead_column("i", "s_amp_mv")]
    net_avf_columns = [_lead_column("avf", "r_amp_mv"), _lead_column("avf", "s_amp_mv")]
    if all(column in features for column in net_i_columns + net_avf_columns):
        net_i = pd.to_numeric(features[net_i_columns[0]], errors="coerce") + pd.to_numeric(features[net_i_columns[1]], errors="coerce")
        net_avf = pd.to_numeric(features[net_avf_columns[0]], errors="coerce") + pd.to_numeric(features[net_avf_columns[1]], errors="coerce")
        add("clinical__frontal_axis_proxy_deg", np.degrees(np.arctan2(net_avf, net_i)),
            "A reproducible frontal-plane axis proxy can identify conduction and hypertrophy confounding.",
            "atan2(net QRS aVF, net QRS I)")

    registry.extend([
        {
            "feature": "pathological_q_wave_amplitude_and_duration",
            "source": "deferred_raw_waveform",
            "clinical_rationale": "Pathological Q waves are clinically relevant for infarction patterns.",
            "formula": "requires validated Q onset/offset and amplitude delineation",
            "status": "DEFERRED_UNTIL_MEASUREMENT_VALIDATION",
        },
        {
            "feature": "sex_and_age_specific_st_thresholds",
            "source": "deferred_clinical_rule",
            "clinical_rationale": "Clinical ST thresholds depend on lead, age and sex.",
            "formula": "prespecified guideline thresholds; never learned from fold 9/10",
            "status": "DEFERRED_UNTIL_PROTOCOL_APPROVAL",
        },
    ])
    return result, pd.DataFrame(registry)


def build_existing_feature_registry(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature in features.select_dtypes(include=[np.number]).columns:
        if feature in {"ecg_id", "patient_id", "strat_fold", "mi_label"}:
            continue
        if "interval" in feature or feature.startswith("qtc_"):
            category = "fiducial_interval"
            rationale = "Summarizes atrial, ventricular or repolarization timing."
        elif feature in {"heart_rate_bpm", "rr_median_ms", "rr_iqr_ms", "rr_cv"}:
            category = "rhythm"
            rationale = "Controls for rhythm and rate-related ECG variation."
        elif feature.endswith("__st60_mv"):
            category = "st_segment"
            rationale = "Lead-specific ST displacement is relevant to ischemic and infarction patterns."
        elif feature.endswith("__t_polarity"):
            category = "t_wave"
            rationale = "Lead-specific T-wave direction represents repolarization morphology."
        elif feature.endswith(("__r_amp_mv", "__s_amp_mv", "__rs_ratio", "__range_mv")):
            category = "qrs_amplitude"
            rationale = "Captures lead-specific depolarization morphology and voltage."
        elif feature.endswith("__valid_fraction"):
            category = "measurement_quality"
            rationale = "Records whether the measurement is supported by sufficient valid samples."
        else:
            category = "other"
            rationale = "Requires clinical review."
        rows.append({
            "feature": feature,
            "source": "existing_deployable",
            "category": category,
            "clinical_rationale": rationale,
            "status": "candidate_pending_evidence",
        })
    return pd.DataFrame(rows)


def cohort_statistical_audit(manifest: pd.DataFrame, output_dir: Path) -> dict:
    """Describe development cohort balance without accessing folds 9 or 10."""

    frame = manifest.copy()
    guard_fold_access(frame["strat_fold"].to_numpy(), purpose="feature_selection")
    if not set(pd.to_numeric(frame.strat_fold).unique()).issubset(set(DEV_FOLDS)):
        raise ValueError("Feature evidence cohort must contain development folds 1-8 only")
    patient_fold = frame.groupby("patient_id").strat_fold.nunique()
    if (patient_fold > 1).any():
        raise ValueError("Patient leakage detected across development folds")
    report = {
        "records": int(len(frame)),
        "patients": int(frame.patient_id.nunique()),
        "mi_records": int(frame.mi_label.sum()),
        "non_mi_records": int((frame.mi_label == 0).sum()),
        "mi_prevalence": float(frame.mi_label.mean()),
        "hard_negative_records": int(frame.get("hard_negative", False).sum()),
        "folds": sorted(pd.to_numeric(frame.strat_fold).astype(int).unique().tolist()),
        "patient_fold_leakage": False,
    }
    group_columns = [
        column for column in ["strat_fold", "sex", "label_quality_group", "hard_negative"]
        if column in frame
    ]
    rows = []
    for group in group_columns:
        for value, part in frame.groupby(group, dropna=False):
            rows.append({
                "group": group,
                "value": str(value),
                "records": int(len(part)),
                "patients": int(part.patient_id.nunique()),
                "mi_records": int(part.mi_label.sum()),
                "mi_prevalence": float(part.mi_label.mean()),
            })
    if "age" in frame:
        age_group = pd.cut(pd.to_numeric(frame.age, errors="coerce"), [-np.inf, 39, 59, 79, np.inf], labels=["<40", "40-59", "60-79", "80+"])
        for value, part in frame.assign(age_group=age_group).groupby("age_group", observed=False):
            rows.append({
                "group": "age_group", "value": str(value), "records": int(len(part)),
                "patients": int(part.patient_id.nunique()), "mi_records": int(part.mi_label.sum()),
                "mi_prevalence": float(part.mi_label.mean()) if len(part) else np.nan,
            })
    output_dir = Path(output_dir)
    _atomic_json(report, output_dir / "cohort_statistical_audit.json")
    _atomic_csv(pd.DataFrame(rows), output_dir / "cohort_balance.csv")
    return report


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {key: np.nan for key in ["median", "q1", "q3", "p05", "p95", "minimum", "maximum"]}
    return {
        "median": float(np.median(values)),
        "q1": float(np.percentile(values, 25)),
        "q3": float(np.percentile(values, 75)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def raw_waveform_statistical_audit(
    hdf5_path: Path,
    output_dir: Path,
    batch_size: int = 256,
    max_records: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Audit accepted waveforms before tabular feature extraction."""

    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError("h5py is required for raw waveform evidence") from exc

    leads = [lead.lower() for lead in CANONICAL_LEAD_ORDER]
    record_rows: list[dict] = []
    lead_values: Dict[Tuple[str, int, str], list[float]] = {}
    with h5py.File(hdf5_path, "r") as h5:
        n_records = int(h5["accepted_signal"].shape[0])
        n_records = min(n_records, max_records) if max_records else n_records
        folds = np.asarray(h5["strat_fold"][:n_records], dtype=int)
        guard_fold_access(folds, purpose="feature_selection")
        if not set(np.unique(folds)).issubset(set(DEV_FOLDS)):
            raise ValueError("Raw feature audit may only access folds 1-8")
        for start in range(0, n_records, batch_size):
            stop = min(n_records, start + batch_size)
            signal = np.asarray(h5["accepted_signal"][start:stop], dtype=float)
            minimal = np.asarray(h5["minimal_signal"][start:stop], dtype=float)
            masks = np.asarray(h5["sample_mask"][start:stop], dtype=bool)
            lead_masks = np.asarray(h5["lead_mask"][start:stop], dtype=bool)
            labels = np.asarray(h5["mi_label"][start:stop], dtype=int)
            valid_signal = np.where(masks, signal, np.nan)
            per_lead_range = np.nanmax(valid_signal, axis=2) - np.nanmin(valid_signal, axis=2)
            per_lead_rms = np.sqrt(np.nanmean(valid_signal ** 2, axis=2))
            per_lead_baseline = np.abs(np.nanmedian(valid_signal, axis=2))
            per_lead_valid = masks.mean(axis=2)
            correction_rms = np.sqrt(np.nanmean((signal - minimal) ** 2, axis=(1, 2)))
            max_abs_amplitude = np.nanmax(np.abs(valid_signal), axis=(1, 2))
            for offset in range(stop - start):
                row = {
                    "array_index": start + offset,
                    "ecg_id": int(h5["ecg_id"][start + offset]),
                    "patient_id": int(h5["patient_id"][start + offset]),
                    "mi_label": int(labels[offset]),
                    "strat_fold": int(folds[start + offset]),
                    "valid_lead_count": int(lead_masks[offset].sum()),
                    "valid_sample_fraction": float(masks[offset].mean()),
                    "median_lead_range_mv": float(np.nanmedian(per_lead_range[offset])),
                    "median_lead_rms_mv": float(np.nanmedian(per_lead_rms[offset])),
                    "median_abs_baseline_mv": float(np.nanmedian(per_lead_baseline[offset])),
                    "max_abs_amplitude_mv": float(max_abs_amplitude[offset]),
                    "correction_rms_change_mv": float(correction_rms[offset]),
                }
                record_rows.append(row)
            batch_metrics = {
                "range_mv": per_lead_range,
                "rms_mv": per_lead_rms,
                "abs_baseline_mv": per_lead_baseline,
                "valid_fraction": per_lead_valid,
            }
            for label in (0, 1):
                label_mask = labels == label
                for lead_index, lead in enumerate(leads):
                    for metric, values in batch_metrics.items():
                        lead_values.setdefault((lead, label, metric), []).extend(
                            values[label_mask, lead_index].astype(float).tolist()
                        )

    lead_rows = []
    for (lead, label, metric), values in lead_values.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        lead_rows.append({"lead": lead, "mi_label": label, "metric": metric, "n": len(finite), **_distribution(finite)})
    record_frame = pd.DataFrame(record_rows)
    lead_frame = pd.DataFrame(lead_rows)
    output_dir = Path(output_dir)
    _atomic_csv(record_frame, output_dir / "raw_waveform_record_audit.csv")
    _atomic_csv(lead_frame, output_dir / "raw_waveform_lead_distributions.csv")
    raw_metric_columns = [
        "valid_lead_count", "valid_sample_fraction", "median_lead_range_mv",
        "median_lead_rms_mv", "median_abs_baseline_mv", "max_abs_amplitude_mv",
        "correction_rms_change_mv",
    ]
    raw_evidence = pd.DataFrame([
        _feature_evidence(metric, record_frame[metric], record_frame.mi_label.to_numpy())
        for metric in raw_metric_columns
    ])
    raw_evidence["fdr_q_value"] = benjamini_hochberg(raw_evidence.p_value)
    _atomic_csv(raw_evidence, output_dir / "raw_waveform_class_evidence.csv")

    lead_evidence_rows = []
    for (lead, metric), part in lead_frame.groupby(["lead", "metric"]):
        positive = np.asarray(lead_values.get((lead, 1, metric), []), dtype=float)
        negative = np.asarray(lead_values.get((lead, 0, metric), []), dtype=float)
        positive = positive[np.isfinite(positive)]
        negative = negative[np.isfinite(negative)]
        p_value = delta = np.nan
        if len(positive) and len(negative):
            test = mannwhitneyu(positive, negative, alternative="two-sided")
            p_value = float(test.pvalue)
            delta = float(2.0 * test.statistic / (len(positive) * len(negative)) - 1.0)
        lead_evidence_rows.append({
            "lead": lead, "metric": metric, "n_mi": len(positive), "n_non_mi": len(negative),
            "median_mi": float(np.median(positive)) if len(positive) else np.nan,
            "median_non_mi": float(np.median(negative)) if len(negative) else np.nan,
            "cliffs_delta": delta, "p_value": p_value,
        })
    lead_evidence = pd.DataFrame(lead_evidence_rows)
    lead_evidence["fdr_q_value"] = benjamini_hochberg(lead_evidence.p_value)
    _atomic_csv(lead_evidence, output_dir / "raw_lead_class_evidence.csv")
    return record_frame, lead_frame


def _safe_auc(y: np.ndarray, score: np.ndarray) -> Tuple[float, float, float]:
    if len(np.unique(y)) < 2 or len(score) < 3:
        return np.nan, np.nan, np.nan
    auc = float(roc_auc_score(y, score))
    ap_positive = float(average_precision_score(y, score))
    ap_negative = float(average_precision_score(y, -score))
    return auc, max(auc, 1.0 - auc), max(ap_positive, ap_negative)


def _feature_evidence(feature: str, series: pd.Series, labels: np.ndarray) -> dict:
    values = pd.to_numeric(series, errors="coerce").to_numpy(float)
    finite = np.isfinite(values)
    y = labels[finite]
    x = values[finite]
    positive = x[y == 1]
    negative = x[y == 0]
    p_value = delta = delta_ci_low = delta_ci_high = np.nan
    if len(positive) and len(negative):
        test = mannwhitneyu(positive, negative, alternative="two-sided")
        p_value = float(test.pvalue)
        delta = float(2.0 * test.statistic / (len(positive) * len(negative)) - 1.0)
        rng = np.random.default_rng(zlib.crc32(feature.encode("utf-8")))
        positive_sample_size = min(len(positive), 1500)
        negative_sample_size = min(len(negative), 1500)
        bootstrapped = []
        for _ in range(100):
            pos = rng.choice(positive, positive_sample_size, replace=True)
            neg = rng.choice(negative, negative_sample_size, replace=True)
            statistic = mannwhitneyu(pos, neg, alternative="two-sided").statistic
            bootstrapped.append(2.0 * statistic / (len(pos) * len(neg)) - 1.0)
        delta_ci_low, delta_ci_high = np.percentile(bootstrapped, [2.5, 97.5])
    auc, auc_strength, auprc_strength = _safe_auc(y, x)
    missing_score = (~finite).astype(float)
    missing_auc = float(roc_auc_score(labels, missing_score)) if len(np.unique(missing_score)) > 1 else 0.5
    return {
        "feature": feature,
        "n": int(len(x)),
        "coverage": float(finite.mean()),
        "missingness": float(1.0 - finite.mean()),
        "unique_values": int(pd.Series(x).nunique()),
        "variance": float(np.var(x)) if len(x) else np.nan,
        "median_all": float(np.median(x)) if len(x) else np.nan,
        "iqr_all": float(np.percentile(x, 75) - np.percentile(x, 25)) if len(x) else np.nan,
        "median_mi": float(np.median(positive)) if len(positive) else np.nan,
        "median_non_mi": float(np.median(negative)) if len(negative) else np.nan,
        "median_difference": float(np.median(positive) - np.median(negative)) if len(positive) and len(negative) else np.nan,
        "cliffs_delta": delta,
        "cliffs_delta_ci_low": float(delta_ci_low),
        "cliffs_delta_ci_high": float(delta_ci_high),
        "p_value": p_value,
        "univariate_auroc": auc,
        "univariate_auroc_strength": auc_strength,
        "univariate_auprc_best_direction": auprc_strength,
        "missingness_auroc": missing_auc,
    }


def _sensitivity_at_specificity(
    y: np.ndarray,
    probability: np.ndarray,
    target: float = 0.90,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    from sklearn.metrics import roc_curve

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y, probability, sample_weight=sample_weight
    )
    valid = np.where((1.0 - false_positive_rate) >= target)[0]
    return float(np.max(true_positive_rate[valid])) if len(valid) else 0.0


def _ablation_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss

    return {
        "auprc": float(average_precision_score(y, probability, sample_weight=sample_weight)),
        "auroc": float(roc_auc_score(y, probability, sample_weight=sample_weight)),
        "brier": float(brier_score_loss(y, probability, sample_weight=sample_weight)),
        "log_loss": float(log_loss(y, probability, labels=[0, 1], sample_weight=sample_weight)),
        "sensitivity_at_90_specificity": _sensitivity_at_specificity(
            y, probability, sample_weight=sample_weight
        ),
    }


def run_clinical_group_ablation(
    base_features: pd.DataFrame,
    derived_features: pd.DataFrame,
    labels: Sequence[int],
    folds: Sequence[int],
    patient_ids: Sequence[int],
    output_dir: Path,
    seed: int = 42,
    bootstrap_repeats: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-safe OOF screening of clinically defined feature groups.

    This is a prespecified logistic-regression ablation, not champion-model
    selection. It tests whether multi-lead clinical patterns add information
    beyond the original deployable measurements.
    """

    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import RobustScaler

    labels_array = np.asarray(labels, dtype=int)
    folds_array = np.asarray(folds, dtype=int)
    patients_array = np.asarray(patient_ids)
    guard_fold_access(folds_array, purpose="feature_selection")
    if not set(np.unique(folds_array)).issubset(set(DEV_FOLDS)):
        raise ValueError("Clinical ablation accepts development folds 1-8 only")
    patient_folds = pd.DataFrame({"patient": patients_array, "fold": folds_array}).groupby("patient").fold.nunique()
    if (patient_folds > 1).any():
        raise ValueError("Patient crosses OOF folds")

    groups = {
        "existing_94": base_features.select_dtypes(include=[np.number]),
        "clinical_composites_only": derived_features.select_dtypes(include=[np.number]),
        "existing_plus_clinical": pd.concat([
            base_features.select_dtypes(include=[np.number]),
            derived_features.select_dtypes(include=[np.number]),
        ], axis=1),
    }
    estimator = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True), RobustScaler(),
        LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        ),
    )
    prediction_rows = []
    probabilities: Dict[str, np.ndarray] = {}
    metric_rows = []
    for group, matrix in groups.items():
        values = matrix.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        probability = np.full(len(values), np.nan)
        for held_out in sorted(np.unique(folds_array)):
            train = folds_array != held_out
            valid = folds_array == held_out
            model = clone(estimator)
            model.fit(values[train], labels_array[train])
            probability[valid] = model.predict_proba(values[valid])[:, 1]
        if not np.isfinite(probability).all():
            raise RuntimeError(f"Incomplete OOF predictions for {group}")
        probabilities[group] = probability
        metric_rows.append({"feature_group": group, **_ablation_metrics(labels_array, probability)})
        prediction_rows.extend({
            "row_index": index, "patient_id": patients_array[index],
            "fold": int(folds_array[index]), "label": int(labels_array[index]),
            "feature_group": group, "probability": float(probability[index]),
        } for index in range(len(values)))

    base_probability = probabilities["existing_94"]
    combined_probability = probabilities["existing_plus_clinical"]
    unique_patients, patient_inverse = np.unique(patients_array, return_inverse=True)
    rng = np.random.default_rng(seed)
    bootstrap_rows = []
    for repeat in range(bootstrap_repeats):
        patient_counts = rng.multinomial(
            len(unique_patients), np.full(len(unique_patients), 1.0 / len(unique_patients))
        )
        record_weights = patient_counts[patient_inverse].astype(float)
        represented = record_weights > 0
        if len(np.unique(labels_array[represented])) < 2:
            continue
        base_metrics = _ablation_metrics(labels_array, base_probability, record_weights)
        combined_metrics = _ablation_metrics(labels_array, combined_probability, record_weights)
        bootstrap_rows.append({
            "repeat": repeat,
            "delta_auprc": combined_metrics["auprc"] - base_metrics["auprc"],
            "delta_auroc": combined_metrics["auroc"] - base_metrics["auroc"],
            "delta_brier": combined_metrics["brier"] - base_metrics["brier"],
            "delta_sensitivity_at_90_specificity": (
                combined_metrics["sensitivity_at_90_specificity"]
                - base_metrics["sensitivity_at_90_specificity"]
            ),
        })
    bootstrap = pd.DataFrame(bootstrap_rows)
    delta_summary = {}
    for column in [name for name in bootstrap if name.startswith("delta_")]:
        delta_summary[column] = {
            "median": float(bootstrap[column].median()),
            "ci_low": float(bootstrap[column].quantile(0.025)),
            "ci_high": float(bootstrap[column].quantile(0.975)),
        }
    promote = bool(
        delta_summary.get("delta_auprc", {}).get("ci_low", -np.inf) > 0
        and delta_summary.get("delta_brier", {}).get("ci_high", np.inf) <= 0
    )
    conclusion = {
        "decision": "PROMOTE_CLINICAL_COMPOSITES" if promote else "DO_NOT_PROMOTE_WITHOUT_MORE_EVIDENCE",
        "criterion": "95% patient-bootstrap CI: delta AUPRC > 0 and delta Brier <= 0",
        "delta_summary": delta_summary,
        "screening_model": "fold-safe class-weighted logistic regression",
        "champion_selection": False,
    }
    output_dir = Path(output_dir)
    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.DataFrame(prediction_rows)
    _atomic_csv(metrics_frame, output_dir / "clinical_group_ablation_metrics.csv")
    _atomic_csv(predictions_frame, output_dir / "clinical_group_ablation_predictions.csv")
    _atomic_csv(bootstrap, output_dir / "clinical_group_ablation_patient_bootstrap.csv")
    _atomic_json(conclusion, output_dir / "clinical_group_ablation_conclusion.json")
    return metrics_frame, predictions_frame


def _mutual_information(frame: pd.DataFrame, labels: np.ndarray, seed: int) -> Dict[str, float]:
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable = [column for column in numeric if numeric[column].notna().any()]
    if not usable:
        return {}
    filled = numeric[usable].fillna(numeric[usable].median()).fillna(0.0)
    # Treat all measurements as continuous. Even polarity/count features are
    # numeric clinical measurements, and this avoids imposing nominal-category
    # geometry on values such as -1/0/+1.
    scores = mutual_info_classif(filled.to_numpy(), labels, discrete_features=False, random_state=seed)
    return {column: float(score) for column, score in zip(usable, scores)}


def _fold_stability(frame: pd.DataFrame, labels: np.ndarray, folds: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for feature in frame.columns:
        values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(float)
        for fold in sorted(np.unique(folds)):
            mask = (folds == fold) & np.isfinite(values)
            y = labels[mask]
            x = values[mask]
            positive, negative = x[y == 1], x[y == 0]
            auc, strength, _ = _safe_auc(y, x)
            difference = float(np.median(positive) - np.median(negative)) if len(positive) and len(negative) else np.nan
            rows.append({
                "feature": feature, "fold": int(fold), "n": int(mask.sum()),
                "median_difference": difference, "univariate_auroc": auc,
                "univariate_auroc_strength": strength,
            })
    detail = pd.DataFrame(rows)
    summaries = []
    for feature, part in detail.groupby("feature"):
        valid = part.median_difference.dropna()
        if len(valid):
            dominant = np.sign(valid.median())
            consistency = float((np.sign(valid) == dominant).mean()) if dominant else float((valid == 0).mean())
        else:
            consistency = np.nan
        summaries.append({
            "feature": feature,
            "folds_with_evidence": int(len(valid)),
            "direction_consistency": consistency,
            "auc_strength_mean": float(part.univariate_auroc_strength.mean()),
            "auc_strength_min": float(part.univariate_auroc_strength.min()),
            "auc_strength_max": float(part.univariate_auroc_strength.max()),
        })
    return detail, pd.DataFrame(summaries)


def _redundancy_pairs(frame: pd.DataFrame, threshold: float, sample_limit: int, seed: int) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable = [column for column in numeric if numeric[column].notna().any() and numeric[column].nunique() > 1]
    sampled = numeric[usable]
    if len(sampled) > sample_limit:
        sampled = sampled.sample(sample_limit, random_state=seed)
    correlation = sampled.corr(method="spearman", min_periods=max(20, min(100, len(sampled) // 10))).abs()
    rows = []
    for left_index, left in enumerate(usable):
        for right in usable[left_index + 1:]:
            value = correlation.at[left, right]
            if np.isfinite(value) and value >= threshold:
                rows.append({"feature_a": left, "feature_b": right, "abs_spearman_rho": float(value)})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "abs_spearman_rho"])


def analyse_extracted_features(
    features: pd.DataFrame,
    labels: Sequence[int],
    folds: Sequence[int],
    output_dir: Path,
    thresholds: EvidenceThresholds = EvidenceThresholds(),
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate post-extraction evidence without touching held-out folds."""

    labels_array = np.asarray(labels, dtype=int)
    folds_array = np.asarray(folds, dtype=int)
    guard_fold_access(folds_array, purpose="feature_selection")
    if len(features) != len(labels_array) or len(labels_array) != len(folds_array):
        raise ValueError("Feature, label and fold lengths differ")
    if not set(np.unique(folds_array)).issubset(set(DEV_FOLDS)):
        raise ValueError("Feature evidence accepts development folds 1-8 only")
    numeric = features.select_dtypes(include=[np.number]).copy()
    forbidden = [column for column in numeric if any(token in column.lower() for token in ["label", "target", "diag", "scp", "report", "infarction", "patient_id", "strat_fold", "ecg_id"])]
    numeric = numeric.drop(columns=forbidden, errors="ignore")
    evidence = pd.DataFrame([_feature_evidence(feature, numeric[feature], labels_array) for feature in numeric])
    evidence["fdr_q_value"] = benjamini_hochberg(evidence.p_value)
    mi = _mutual_information(numeric, labels_array, seed)
    evidence["mutual_information"] = evidence.feature.map(mi).fillna(0.0)
    fold_detail, stability = _fold_stability(numeric, labels_array, folds_array)
    evidence = evidence.merge(stability, on="feature", how="left", validate="one_to_one")
    redundancy = _redundancy_pairs(numeric, thresholds.redundancy_rho, 5000, seed)
    redundant = set(redundancy.feature_a).union(redundancy.feature_b) if len(redundancy) else set()

    decisions = []
    for row in evidence.itertuples(index=False):
        reasons = []
        decision = "KEEP_FOR_NESTED_ABLATION"
        if row.coverage < thresholds.minimum_coverage:
            decision = "REPAIR_OR_EXCLUDE"
            reasons.append("coverage_below_prespecified_minimum")
        if not np.isfinite(row.variance) or row.variance <= thresholds.near_zero_variance or row.unique_values <= 1:
            decision = "EXCLUDE_NONINFORMATIVE"
            reasons.append("near_zero_variance")
        if np.isfinite(row.direction_consistency) and row.direction_consistency < thresholds.minimum_fold_direction_consistency:
            decision = "REVIEW_UNSTABLE"
            reasons.append("effect_direction_not_stable_across_folds")
        if row.feature in redundant:
            reasons.append("high_redundancy_cluster_member")
            if decision == "KEEP_FOR_NESTED_ABLATION":
                decision = "REDUNDANCY_REVIEW"
        signal = (
            (np.isfinite(row.cliffs_delta) and abs(row.cliffs_delta) >= thresholds.minimum_effect_size)
            or (np.isfinite(row.univariate_auroc_strength) and row.univariate_auroc_strength - 0.5 >= thresholds.minimum_auc_distance)
            or row.mutual_information >= thresholds.minimum_mutual_information
        )
        if not signal:
            reasons.append("weak_univariate_signal_requires_multivariable_ablation")
        if np.isfinite(row.fdr_q_value) and row.fdr_q_value > thresholds.fdr_alpha:
            reasons.append("not_fdr_significant")
        decisions.append({
            "feature": row.feature, "recommendation": decision,
            "evidence_reasons": ";".join(reasons) if reasons else "passed_statistical_triage",
            "automatic_removal": False,
        })
    decisions_frame = pd.DataFrame(decisions)
    output_dir = Path(output_dir)
    _atomic_csv(evidence, output_dir / "feature_statistical_evidence.csv")
    _atomic_csv(fold_detail, output_dir / "feature_fold_evidence.csv")
    _atomic_csv(redundancy, output_dir / "feature_redundancy_pairs.csv")
    _atomic_csv(decisions_frame, output_dir / "feature_decision_register.csv")
    _atomic_json({
        "thresholds": asdict(thresholds),
        "records": len(features),
        "features_evaluated": len(numeric.columns),
        "selection_scope": "PTB-XL development folds 1-8 only",
        "automatic_feature_removal": False,
        "decision_rule": "measurement validity first; final inclusion requires fold-local ablation",
    }, output_dir / "feature_evidence_protocol.json")
    return evidence, redundancy, decisions_frame


def extractor_failure_report(features: pd.DataFrame, output: Path) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    if "extractor_failures" in features:
        for raw in features.extractor_failures.fillna("[]"):
            try:
                values = json.loads(raw) if isinstance(raw, str) else list(raw)
            except (json.JSONDecodeError, TypeError):
                values = ["unparseable_failure_metadata"]
            for value in values:
                key = str(value).split(":", 1)[0]
                counts[key] = counts.get(key, 0) + 1
    frame = pd.DataFrame([
        {"failure": key, "records": value, "fraction": value / max(len(features), 1)}
        for key, value in sorted(counts.items())
    ], columns=["failure", "records", "fraction"])
    _atomic_csv(frame, Path(output))
    return frame


def run_feature_evidence_gate(
    features: pd.DataFrame,
    metadata: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: Path,
    primary_hdf5: Optional[Path] = None,
    reference_features: Optional[Path] = None,
    reference_pairs: Optional[Mapping[str, str]] = None,
    max_raw_records: Optional[int] = None,
) -> dict:
    """Run the complete pre/post-extraction evidence gate."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame in (features, metadata, manifest):
        if "ecg_id" in frame.columns:
            frame.set_index("ecg_id", inplace=True)
        frame.index = pd.to_numeric(frame.index, errors="raise").astype(int)
        if not frame.index.is_unique:
            raise ValueError("ecg_id must be unique in every feature-evidence input")
    joined = metadata.join(manifest, how="inner", rsuffix="__manifest", validate="one_to_one")
    joined = joined[joined.strat_fold.isin(DEV_FOLDS) & joined.eligibility.eq("PRIMARY")]
    joined = joined.join(features, how="inner", validate="one_to_one")
    if not len(joined):
        raise ValueError("No accepted development records align across inputs")
    cohort_statistical_audit(joined, output_dir / "pre_extraction")
    if primary_hdf5 is not None:
        raw_waveform_statistical_audit(primary_hdf5, output_dir / "pre_extraction", max_records=max_raw_records)

    feature_columns = [column for column in features.columns if column in joined.columns]
    base = joined[feature_columns].copy()
    derived, derived_registry = derive_clinical_composites(base)
    combined = pd.concat([base, derived], axis=1)
    existing_registry = build_existing_feature_registry(base)
    registry = pd.concat([existing_registry, derived_registry], ignore_index=True, sort=False)
    _atomic_csv(registry, output_dir / "candidate_feature_registry.csv")
    extractor_failure_report(base, output_dir / "post_extraction" / "extractor_failures.csv")
    evidence, redundancy, decisions = analyse_extracted_features(
        combined,
        joined.mi_label.to_numpy(),
        joined.strat_fold.to_numpy(),
        output_dir / "post_extraction",
    )
    ablation_metrics, _ = run_clinical_group_ablation(
        base,
        derived,
        joined.mi_label.to_numpy(),
        joined.strat_fold.to_numpy(),
        joined.patient_id.to_numpy(),
        output_dir / "clinical_ablation",
    )
    combined_output = combined.copy()
    combined_output.insert(0, "ecg_id", combined_output.index)
    _atomic_csv(combined_output, output_dir / "deployable_features_with_clinical_composites.csv")

    reference_rows = 0
    if reference_features is not None and reference_pairs:
        raw_reference_columns = sorted({
            column.removeprefix("ref_ecgdeli__") for column in reference_pairs.values()
        })
        reference = pd.read_csv(
            reference_features,
            usecols=lambda column: column == "ecg_id" or column in raw_reference_columns,
        ).set_index("ecg_id")
        reference = reference.add_prefix("ref_ecgdeli__")
        reference_joined = base.join(reference, how="left", validate="one_to_one")
        result = compare_local_to_reference(reference_joined, reference_pairs, output_dir / "measurement_validation")
        reference_rows = int(len(result))

    summary = {
        "status": "EVIDENCE_GENERATED_REQUIRES_CLINICAL_REVIEW",
        "development_records": int(len(joined)),
        "base_numeric_features": int(len(existing_registry)),
        "derived_candidate_features": int(len(derived.columns)),
        "features_statistically_evaluated": int(len(evidence)),
        "redundancy_pairs": int(len(redundancy)),
        "repair_or_exclude_recommendations": int(decisions.recommendation.isin(["REPAIR_OR_EXCLUDE", "EXCLUDE_NONINFORMATIVE"]).sum()),
        "reference_pairs_evaluated": reference_rows,
        "clinical_feature_groups_screened": int(len(ablation_metrics)),
        "folds_used": sorted(joined.strat_fold.astype(int).unique().tolist()),
        "fold_9_accessed": False,
        "fold_10_accessed": False,
        "ready_for_automatic_feature_deletion": False,
        "next_required_action": "clinical review, repair invalid measurements, then freeze the approved feature manifest",
    }
    _atomic_json(summary, output_dir / "feature_evidence_summary.json")
    return summary
