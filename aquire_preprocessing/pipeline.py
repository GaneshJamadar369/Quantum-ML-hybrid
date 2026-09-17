"""Research-grade preprocessing orchestration and incremental persistence."""

from __future__ import annotations

import json
import hashlib
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PATHS, PIPELINE_VERSION, QC
from .contracts import Eligibility, GateState, ProcessedECG
from .manifest import guard_fold_access
from .features import extract_deployable_features
from .morphology_gate import GateResult, evaluate_gate
from .normalization import LeadRobustScaler
from .quality import QCResult, assess_quality
from .router import route_record
from .resampling import resample_500_to_100, resample_corrected_pair
from .storage import HDF5RecordWriter
from .structural import validate_record
from .views import generate_views


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def process_record(ecg_id: int, row: pd.Series, sampling_rate: int = 100) -> Optional[ProcessedECG]:
    structural = validate_record(
        int(ecg_id),
        sampling_rate=sampling_rate,
        metadata_row=row,
    )
    if not structural.is_valid or structural.signal_mv is None:
        return None

    qc = assess_quality(
        structural.signal_mv,
        int(ecg_id),
        fs=sampling_rate,
        sample_mask=structural.sample_mask,
        lead_mask=structural.lead_mask,
    )
    routing = route_record(qc)
    original_sample_mask = qc.sample_mask.copy()
    original_lead_mask = qc.lead_mask.copy()
    views = generate_views(
        structural.signal_mv,
        routing,
        int(ecg_id),
        fs=sampling_rate,
        sample_mask=qc.sample_mask,
        lead_mask=qc.lead_mask,
    )
    gate: Optional[GateResult] = None
    minimal_output = views.signal_minimal.copy()
    accepted = views.signal_minimal.copy()
    accepted_sample_mask = original_sample_mask
    accepted_lead_mask = original_lead_mask
    if views.view_b_differs:
        gate = evaluate_gate(
            views.signal_minimal,
            views.signal_corrected,
            int(ecg_id),
            fs=sampling_rate,
            lead_mask=views.lead_mask,
        )
        if gate.state == GateState.PASS.value:
            accepted = views.signal_corrected.copy()
            accepted_sample_mask = views.sample_mask
            accepted_lead_mask = views.lead_mask
    elif qc.qc_status == "FAIL":
        accepted_sample_mask = views.sample_mask
        accepted_lead_mask = views.lead_mask

    native_gate = gate
    if sampling_rate == 500:
        resample_mask = views.sample_mask if qc.qc_status == "FAIL" else original_sample_mask
        resample_leads = views.lead_mask if qc.qc_status == "FAIL" else original_lead_mask
        minimal_output, minimal_mask, minimal_leads = resample_500_to_100(
            views.signal_minimal, resample_mask, resample_leads
        )
        if views.view_b_differs and native_gate and native_gate.state == GateState.PASS.value:
            resampled = resample_corrected_pair(
                views.signal_minimal, views.signal_corrected, int(ecg_id),
                sample_mask=views.sample_mask, lead_mask=views.lead_mask,
            )
            accepted = resampled.signal_100hz
            accepted_sample_mask = resampled.sample_mask_100hz
            accepted_lead_mask = resampled.lead_mask
            gate = resampled.gate
        else:
            accepted = minimal_output.copy()
            accepted_sample_mask = minimal_mask
            accepted_lead_mask = minimal_leads
    eligibility = Eligibility.QUARANTINE if qc.qc_status == "FAIL" else Eligibility.PRIMARY
    label_metadata = {
        key: row.get(key)
        for key in [
            "mi_scp_codes", "annotation_likelihood_max",
            "annotation_likelihood_known", "label_quality_group",
            "hard_negative_groups", "validated_by_human_flag",
            "second_opinion_flag",
            "baseline_drift", "static_noise", "burst_noise", "electrodes_problems",
        ]
        if key in row.index
    }
    provenance = {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_version": row.get("dataset_version", "unknown"),
        "source_paths": structural.source_paths,
        "source_checksum": structural.source_checksum,
        "original_shape": structural.original_shape,
        "sampling_rate": structural.sampling_rate,
        "model_sampling_rate": 100,
        "transformations": structural.transformations,
        "processing_route": routing.processing_route,
        "corrections_applied": views.corrections_applied,
        "native_morphology_gate": asdict(native_gate) if native_gate is not None else None,
    }
    return ProcessedECG(
        ecg_id=int(ecg_id),
        patient_id=int(row["patient_id"]),
        minimal_signal=minimal_output,
        accepted_signal=accepted,
        sample_mask=accepted_sample_mask,
        lead_mask=accepted_lead_mask,
        qc_result=qc,
        morphology_result=gate,
        eligibility=eligibility,
        provenance=provenance,
        mi_label=int(row["mi_label"]),
        strat_fold=int(row["strat_fold"]),
        hard_negative=bool(row.get("hard_negative", False)),
        label_metadata=label_metadata,
    )


def processed_to_contract(record: ProcessedECG) -> dict:
    gate = record.morphology_result
    return {
        "ecg_id": record.ecg_id,
        "patient_id": record.patient_id,
        "signal_minimal": record.minimal_signal,
        "signal_corrected": record.accepted_signal,
        "lead_mask": record.lead_mask,
        "sample_mask": record.sample_mask,
        "qc_status": record.qc_result.qc_status,
        "processing_route": record.provenance.get("processing_route", []),
        "mi_label": record.mi_label,
        "hard_negative": record.hard_negative,
        "strat_fold": record.strat_fold,
        "source_checksum": record.provenance.get("source_checksum", ""),
        "minimal_checksum": hashlib.sha256(
            np.ascontiguousarray(record.minimal_signal).tobytes()
        ).hexdigest(),
        "accepted_checksum": hashlib.sha256(
            np.ascontiguousarray(record.accepted_signal).tobytes()
        ).hexdigest(),
        "pipeline_version": PIPELINE_VERSION,
        "eligibility": record.eligibility.value,
        "gate_state": gate.state if gate else GateState.PASS.value,
    }


def _metadata_row(record: ProcessedECG, array_index: int) -> dict:
    qc: QCResult = record.qc_result
    gate = record.morphology_result
    return {
        "array_index": int(array_index),
        "ecg_id": record.ecg_id,
        "patient_id": record.patient_id,
        "mi_label": record.mi_label,
        "hard_negative": record.hard_negative,
        "strat_fold": record.strat_fold,
        "eligibility": record.eligibility.value,
        "qc_status": qc.qc_status,
        "failed_leads": qc.n_failed_leads,
        "warned_leads": qc.n_warned_leads,
        "gate_state": gate.state if gate else GateState.PASS.value,
        "gate_utility": gate.utility_score if gate else 0.0,
        "source_checksum": record.provenance.get("source_checksum", ""),
        "pipeline_version": PIPELINE_VERSION,
        "processing_route": json.dumps(record.provenance.get("processing_route", [])),
        "label_quality_group": record.label_metadata.get("label_quality_group"),
        "annotation_likelihood_max": record.label_metadata.get("annotation_likelihood_max"),
    }


def _qc_rows(record: ProcessedECG) -> List[dict]:
    rows = []
    for lead, quality in record.qc_result.per_lead.items():
        row = {
            "ecg_id": record.ecg_id, "patient_id": record.patient_id,
            "strat_fold": record.strat_fold, "mi_label": record.mi_label,
            "hard_negative": record.hard_negative, "lead": lead,
            "qc_status": record.qc_result.qc_status,
            "lead_status": quality.status,
            "is_flat": quality.is_flat, "is_clipped": quality.is_clipped,
            "missing_fraction": quality.missing_fraction,
            "baseline_wander_ratio": quality.baseline_wander_ratio,
            "powerline_snr_db": quality.powerline_snr_db,
            "powerline_supported": quality.powerline_supported,
            "hf_noise_ratio": quality.hf_noise_ratio,
            "amplitude_range_mv": quality.amplitude_range_mv,
            "amplitude_invalid_score": max(
                QC.amplitude_min_mv - quality.amplitude_range_mv,
                quality.amplitude_range_mv - QC.amplitude_max_mv,
                0.0,
            ),
            "is_plausible_amplitude": quality.is_plausible_amplitude,
            "issues": json.dumps(quality.issues),
        }
        row.update({
            key: record.label_metadata.get(key)
            for key in ["baseline_drift", "static_noise", "burst_noise", "electrodes_problems"]
        })
        rows.append(row)
    return rows


def _morphology_rows(record: ProcessedECG) -> List[dict]:
    gate = record.morphology_result
    if gate is None:
        return []
    rows = []
    for lead, metrics in gate.per_lead_metrics.items():
        row = asdict(metrics)
        row.update({
            "ecg_id": record.ecg_id, "patient_id": record.patient_id,
            "strat_fold": record.strat_fold, "gate_state": gate.state,
            "gate_utility": gate.utility_score,
            "artifact_reduction": gate.artifact_reduction,
        })
        rows.append(row)
    return rows


def process_batch(
    manifest: pd.DataFrame,
    sampling_rate: int = 100,
    max_records: Optional[int] = None,
    save_output: bool = True,
    collect_records: bool = False,
    purpose: str = "preprocessing_tuning",
) -> Tuple[List[ProcessedECG], dict]:
    guard_fold_access(manifest["strat_fold"].to_numpy(), purpose=purpose)
    ecg_ids = manifest.index.tolist()[:max_records]
    start = time.perf_counter()
    records: List[ProcessedECG] = []
    metadata: List[dict] = []
    feature_rows: List[dict] = []
    qc_rows: List[dict] = []
    morphology_rows: List[dict] = []
    counts = {"structural_failures": 0, "primary": 0, "quarantine": 0, "gate_rejections": 0}

    primary_writer = quarantine_writer = None
    if save_output:
        n_samples = 1000
        role = {
            "preprocessing_tuning": "development",
            "calibration_evaluation": "calibration",
            "final_locked_evaluation": "locked_test",
        }.get(purpose, purpose.replace(" ", "_"))
        rate_label = "100hz" if sampling_rate == 100 else "500to100hz"
        primary_writer = HDF5RecordWriter(PATHS.output_root / f"primary_{role}_{rate_label}.h5", n_samples)
        quarantine_writer = HDF5RecordWriter(PATHS.output_root / f"quarantine_{role}_{rate_label}.h5", n_samples)
        primary_index = {
            int(ecg_id): index for index, ecg_id in enumerate(primary_writer.handle["ecg_id"][:].tolist())
        }
        quarantine_index = {
            int(ecg_id): index for index, ecg_id in enumerate(quarantine_writer.handle["ecg_id"][:].tolist())
        }
        completed = set(primary_index) | set(quarantine_index)
    else:
        completed = set()
        primary_index = quarantine_index = {}

    try:
        for ecg_id in ecg_ids:
            already_completed = int(ecg_id) in completed
            record = process_record(int(ecg_id), manifest.loc[ecg_id], sampling_rate)
            if record is None:
                if already_completed:
                    raise RuntimeError(f"Structural result changed while resuming ecg_id={ecg_id}")
                counts["structural_failures"] += 1
                metadata.append({"array_index": -1, "ecg_id": int(ecg_id), "eligibility": Eligibility.REJECTED.value})
                continue
            if record.eligibility == Eligibility.PRIMARY:
                counts["primary"] += 1
                if already_completed:
                    if int(ecg_id) not in primary_index:
                        raise RuntimeError(f"Eligibility changed while resuming ecg_id={ecg_id}")
                    index = primary_index[int(ecg_id)]
                else:
                    index = primary_writer.append(record) if primary_writer else counts["primary"] - 1
            else:
                counts["quarantine"] += 1
                if already_completed:
                    if int(ecg_id) not in quarantine_index:
                        raise RuntimeError(f"Eligibility changed while resuming ecg_id={ecg_id}")
                    index = quarantine_index[int(ecg_id)]
                else:
                    index = quarantine_writer.append(record) if quarantine_writer else counts["quarantine"] - 1
            if record.morphology_result and record.morphology_result.state != GateState.PASS.value:
                counts["gate_rejections"] += 1
            metadata.append(_metadata_row(record, index))
            qc_rows.extend(_qc_rows(record))
            morphology_rows.extend(_morphology_rows(record))
            if record.eligibility == Eligibility.PRIMARY:
                bundle = extract_deployable_features(
                    record.accepted_signal,
                    100,
                    record.ecg_id,
                    sample_mask=record.sample_mask,
                    lead_mask=record.lead_mask,
                )
                feature_rows.append({
                    "ecg_id": record.ecg_id,
                    **bundle.values,
                    "extractor_failures": json.dumps(bundle.failures),
                })
            if collect_records:
                records.append(record)
    except Exception:
        if primary_writer:
            primary_writer.close(commit=False)
        if quarantine_writer:
            quarantine_writer.close(commit=False)
        raise
    else:
        if primary_writer:
            primary_writer.close(commit=True)
        if quarantine_writer:
            quarantine_writer.close(commit=True)

    if save_output:
        PATHS.output_root.mkdir(parents=True, exist_ok=True)
        _atomic_csv(pd.DataFrame(metadata), PATHS.output_root / f"processing_metadata_{role}.csv")
        _atomic_csv(pd.DataFrame(feature_rows), PATHS.output_root / f"deployable_features_{role}.csv")
        _atomic_csv(pd.DataFrame(qc_rows), PATHS.output_root / f"qc_metrics_{role}.csv")
        _atomic_csv(pd.DataFrame(morphology_rows), PATHS.output_root / f"morphology_metrics_{role}.csv")
    elapsed = time.perf_counter() - start
    summary = {
        "total_requested": len(ecg_ids),
        "total_completed": sum(counts.values()) - counts["gate_rejections"],
        **counts,
        "seconds": elapsed,
        "records_per_second": (len(ecg_ids) - counts["structural_failures"]) / elapsed if elapsed else 0.0,
        "purpose": purpose,
    }
    return records, summary


def fit_fold_local_normalizers(
    records: List[ProcessedECG],
    output_dir: Path,
) -> Dict[int, LeadRobustScaler]:
    """Fit one scaler per held-out development fold; never touches folds 9/10."""
    if not records:
        raise ValueError("No records supplied")
    folds = np.asarray([r.strat_fold for r in records])
    guard_fold_access(folds, purpose="feature_selection")
    signals = np.stack([r.accepted_signal for r in records])
    sample_masks = np.stack([r.sample_mask for r in records])
    lead_masks = np.stack([r.lead_mask for r in records])
    patients = np.asarray([r.patient_id for r in records])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scalers: Dict[int, LeadRobustScaler] = {}
    for held_out in sorted(set(folds)):
        allowed = [int(v) for v in sorted(set(folds)) if int(v) != int(held_out)]
        scaler = LeadRobustScaler().fit(
            signals,
            folds=folds,
            allowed_folds=allowed,
            sample_masks=sample_masks,
            lead_masks=lead_masks,
            patient_ids=patients,
        )
        scaler.save(output_dir / f"normalizer_holdout_fold_{int(held_out)}.json")
        scalers[int(held_out)] = scaler
    return scalers
