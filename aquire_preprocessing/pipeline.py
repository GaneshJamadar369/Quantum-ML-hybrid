"""
End-to-end preprocessing pipeline orchestrator.

Implements the full pipeline from signal-data-preprocessing-plan.md:
1. Structural validation → canonical (12, 1000) tensor
2. Lead-level QC fingerprint
3. Cross-lead physics checks
4. Router decision
5. Generate View A (minimal) and View B (corrected)
6. Morphology gate validation
7. Emit the output contract dict (per spec Section 15)
8. Batch mode: process all records
"""

import hashlib
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    CANONICAL_LEAD_ORDER,
    DATASET_VERSION_PTBXL,
    NUM_LEADS,
    PATHS,
    PIPELINE_VERSION,
    SAMPLES_100HZ,
)
from .manifest import build_manifest, compute_waveform_hash
from .structural import StructuralValidation, validate_record
from .quality import QCResult, assess_quality
from .router import RoutingDecision, route_record
from .views import SignalViews, generate_views
from .morphology_gate import GateResult, evaluate_gate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output contract (Section 15 of the plan)
# ---------------------------------------------------------------------------

def build_output_contract(
    ecg_id: int,
    patient_id: int,
    views: SignalViews,
    qc: QCResult,
    routing: RoutingDecision,
    gate: Optional[GateResult],
    mi_label: int,
    label_confidence: float,
    hard_negative: bool,
    strat_fold: int,
    source_checksum: str,
) -> dict:
    """Build the versioned output contract for a single record.

    This is the standardized schema from plan Section 15.
    """
    # If the gate rejected the correction, use View A for both
    if gate is not None and not gate.gate_passed:
        final_corrected = views.signal_minimal.copy()
        gate_action = "correction_rejected"
    else:
        final_corrected = views.signal_corrected
        gate_action = "correction_accepted" if views.view_b_differs else "no_correction"

    return {
        "ecg_id": int(ecg_id),
        "patient_id": int(patient_id),
        "signal_minimal": views.signal_minimal,           # float32[12, 1000]
        "signal_corrected": final_corrected,              # float32[12, 1000]
        "lead_mask": views.lead_mask,                     # bool[12]
        "sample_mask": views.sample_mask,                 # bool[12, 1000]
        "qc_status": qc.qc_status,                       # PASS | WARN | FAIL
        "qc_metrics": {
            "per_lead": {
                lead: {
                    "status": lq.status,
                    "is_flat": lq.is_flat,
                    "is_clipped": lq.is_clipped,
                    "baseline_wander_ratio": lq.baseline_wander_ratio,
                    "powerline_snr_db": lq.powerline_snr_db,
                    "hf_noise_ratio": lq.hf_noise_ratio,
                    "amplitude_range_mv": lq.amplitude_range_mv,
                }
                for lead, lq in qc.per_lead.items()
            },
            "cross_lead": {
                "einthoven_residual_mv": qc.cross_lead.einthoven_residual_mv
                if qc.cross_lead else None,
                "goldberger_avr_mv": qc.cross_lead.goldberger_avr_residual_mv
                if qc.cross_lead else None,
                "goldberger_avl_mv": qc.cross_lead.goldberger_avl_residual_mv
                if qc.cross_lead else None,
                "goldberger_avf_mv": qc.cross_lead.goldberger_avf_residual_mv
                if qc.cross_lead else None,
            },
            "morphology_preservation": {
                "gate_passed": gate.gate_passed if gate else True,
                "utility_score": gate.utility_score if gate else 0.0,
                "gate_action": gate_action,
            },
        },
        "processing_route": routing.processing_route,
        "mi_label": int(mi_label),
        "label_confidence": float(label_confidence),
        "hard_negative": bool(hard_negative),
        "strat_fold": int(strat_fold),
        "source_checksum": source_checksum,
        "dataset_version": DATASET_VERSION_PTBXL,
        "pipeline_version": PIPELINE_VERSION,
    }


# ---------------------------------------------------------------------------
# Single-record pipeline
# ---------------------------------------------------------------------------

def process_record(
    ecg_id: int,
    patient_id: int,
    mi_label: int,
    label_confidence: float,
    hard_negative: bool,
    strat_fold: int,
    sampling_rate: int = 100,
) -> dict:
    """Process a single ECG record through the full pipeline.

    Returns the output contract dict or None if structural validation fails.
    """
    # 1. Structural validation
    sv = validate_record(ecg_id, sampling_rate)
    if not sv.is_valid:
        logger.warning("ecg_id %d: structural validation FAILED — %s", ecg_id, sv.errors)
        return None

    signal = sv.signal
    source_checksum = compute_waveform_hash(signal)

    # 2. Quality assessment
    qc = assess_quality(signal, ecg_id, fs=sampling_rate)

    # 3. Processing router
    routing = route_record(qc)

    # 4. Generate views
    views = generate_views(signal, routing, ecg_id, fs=sampling_rate)

    # 5. Morphology gate (only if correction was applied)
    gate = None
    if views.view_b_differs:
        gate = evaluate_gate(
            views.signal_minimal,
            views.signal_corrected,
            ecg_id,
            fs=sampling_rate,
        )
        # If gate rejects, fall back to View A
        if not gate.gate_passed:
            logger.info(
                "ecg_id %d: morphology gate rejected correction, using View A",
                ecg_id,
            )

    # 6. Build output contract
    contract = build_output_contract(
        ecg_id=ecg_id,
        patient_id=patient_id,
        views=views,
        qc=qc,
        routing=routing,
        gate=gate,
        mi_label=mi_label,
        label_confidence=label_confidence,
        hard_negative=hard_negative,
        strat_fold=strat_fold,
        source_checksum=source_checksum,
    )

    return contract


# ---------------------------------------------------------------------------
# Batch pipeline
# ---------------------------------------------------------------------------

def process_batch(
    manifest: pd.DataFrame,
    sampling_rate: int = 100,
    max_records: Optional[int] = None,
    save_output: bool = True,
) -> Tuple[List[dict], dict]:
    """Process all ECG records through the full pipeline.

    Parameters
    ----------
    manifest : pd.DataFrame
        Patient manifest indexed by ecg_id with mi_label, etc.
    sampling_rate : int
        Target sampling rate.
    max_records : int, optional
        Limit processing to this many records (for testing).
    save_output : bool
        Whether to save results to disk.

    Returns
    -------
    contracts : list of dict
        Output contracts for all successfully processed records.
    summary : dict
        Processing summary statistics.
    """
    ecg_ids = manifest.index.tolist()
    if max_records is not None:
        ecg_ids = ecg_ids[:max_records]

    n_total = len(ecg_ids)
    logger.info("Starting batch processing of %d records", n_total)

    contracts = []
    n_pass = 0
    n_warn = 0
    n_fail = 0
    n_struct_fail = 0
    n_gate_rejected = 0

    start_time = time.time()

    for idx, ecg_id in enumerate(ecg_ids):
        if (idx + 1) % 500 == 0 or idx == 0:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                "Processing %d/%d (%.1f records/sec)...",
                idx + 1, n_total, rate,
            )

        row = manifest.loc[ecg_id]
        contract = process_record(
            ecg_id=ecg_id,
            patient_id=int(row["patient_id"]),
            mi_label=int(row.get("mi_label", 0)),
            label_confidence=float(row.get("label_confidence", 1.0)),
            hard_negative=bool(row.get("hard_negative", False)),
            strat_fold=int(row["strat_fold"]),
            sampling_rate=sampling_rate,
        )

        if contract is None:
            n_struct_fail += 1
            continue

        contracts.append(contract)

        if contract["qc_status"] == "PASS":
            n_pass += 1
        elif contract["qc_status"] == "WARN":
            n_warn += 1
        elif contract["qc_status"] == "FAIL":
            n_fail += 1

        if (
            contract["qc_metrics"]["morphology_preservation"]["gate_action"]
            == "correction_rejected"
        ):
            n_gate_rejected += 1

    elapsed = time.time() - start_time

    summary = {
        "total_attempted": n_total,
        "total_processed": len(contracts),
        "structural_failures": n_struct_fail,
        "qc_pass": n_pass,
        "qc_warn": n_warn,
        "qc_fail": n_fail,
        "gate_rejections": n_gate_rejected,
        "processing_time_seconds": elapsed,
        "records_per_second": len(contracts) / elapsed if elapsed > 0 else 0,
    }

    # --- Save outputs ---
    if save_output and contracts:
        out_dir = PATHS.output_root
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save signals as numpy arrays
        signals_minimal = np.stack(
            [c["signal_minimal"] for c in contracts], axis=0
        )
        signals_corrected = np.stack(
            [c["signal_corrected"] for c in contracts], axis=0
        )
        lead_masks = np.stack([c["lead_mask"] for c in contracts], axis=0)
        sample_masks = np.stack([c["sample_mask"] for c in contracts], axis=0)

        np.save(out_dir / "signals_minimal.npy", signals_minimal)
        np.save(out_dir / "signals_corrected.npy", signals_corrected)
        np.save(out_dir / "lead_masks.npy", lead_masks)
        np.save(out_dir / "sample_masks.npy", sample_masks)

        # Save metadata as CSV
        meta_rows = []
        for c in contracts:
            meta_rows.append({
                "ecg_id": c["ecg_id"],
                "patient_id": c["patient_id"],
                "qc_status": c["qc_status"],
                "mi_label": c["mi_label"],
                "label_confidence": c["label_confidence"],
                "hard_negative": c["hard_negative"],
                "strat_fold": c["strat_fold"],
                "source_checksum": c["source_checksum"],
                "gate_action": c["qc_metrics"]["morphology_preservation"]["gate_action"],
                "utility_score": c["qc_metrics"]["morphology_preservation"]["utility_score"],
                "einthoven_residual": c["qc_metrics"]["cross_lead"]["einthoven_residual_mv"],
                "pipeline_version": c["pipeline_version"],
            })
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(out_dir / "processing_metadata.csv", index=False)

        logger.info(
            "Saved outputs to %s: signals (%s), metadata (%d rows)",
            out_dir, signals_minimal.shape, len(meta_df),
        )

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("AQUIRE-Med Preprocessing Pipeline Summary")
    print("=" * 60)
    print(f"Records attempted:    {summary['total_attempted']}")
    print(f"Records processed:    {summary['total_processed']}")
    print(f"Structural failures:  {summary['structural_failures']}")
    print(f"QC PASS:              {summary['qc_pass']}")
    print(f"QC WARN:              {summary['qc_warn']}")
    print(f"QC FAIL:              {summary['qc_fail']}")
    print(f"Gate rejections:      {summary['gate_rejections']}")
    print(f"Processing time:      {summary['processing_time_seconds']:.1f}s")
    print(f"Speed:                {summary['records_per_second']:.1f} records/sec")
    print("=" * 60 + "\n")

    return contracts, summary
