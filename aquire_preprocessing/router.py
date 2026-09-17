"""
Artifact-aware processing router.

Implements plan phase 6:
- Routes each record to the correct processing path based on QC results
- Deterministic, versioned, and logged in processing_route[]
- Clean signals → minimal path (no filtering)
- Artifact-specific corrections only when detected
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .config import QC
from .quality import QCResult

logger = logging.getLogger(__name__)


class ProcessingAction(Enum):
    """Available processing actions for the router."""
    MINIMAL = "minimal"                        # Clean signal → no correction
    BASELINE_CORRECTION = "baseline_correction" # Zero-phase baseline removal
    POWERLINE_REMOVAL = "powerline_removal"    # Targeted 50/60 Hz notch
    SHORT_GAP_REPAIR = "short_gap_repair"      # Limited interpolation + mask
    LEAD_MASK = "lead_mask"                    # Mark lead as unusable
    FAIL = "fail"                              # Entire record unusable


@dataclass
class RoutingDecision:
    """The processing route for a single ECG record."""
    ecg_id: int
    overall_action: str = "minimal"  # primary action label
    per_lead_actions: dict = field(default_factory=dict)  # lead_name → [actions]
    processing_route: List[str] = field(default_factory=list)  # ordered log
    requires_correction: bool = False


def route_record(qc_result: QCResult) -> RoutingDecision:
    """Determine the processing route for a record based on its QC assessment.

    Parameters
    ----------
    qc_result : QCResult
        Output from quality.assess_quality().

    Returns
    -------
    RoutingDecision
        Contains per-lead actions and the ordered processing route log.
    """
    decision = RoutingDecision(ecg_id=qc_result.ecg_id)

    # --- Record-level FAIL ---
    if qc_result.qc_status == "FAIL":
        decision.overall_action = ProcessingAction.FAIL.value
        decision.processing_route.append(
            f"RECORD_FAIL: {'; '.join(qc_result.summary_issues)}"
        )
        logger.warning(
            "ecg_id %d routed to FAIL: %s",
            qc_result.ecg_id, qc_result.summary_issues,
        )
        return decision

    # --- Per-lead routing ---
    any_correction_needed = False

    for lead_name, lq in qc_result.per_lead.items():
        actions = []

        # FAIL lead → mask it
        if lq.status == "FAIL":
            actions.append(ProcessingAction.LEAD_MASK.value)
            any_correction_needed = True
            decision.processing_route.append(
                f"{lead_name}: LEAD_MASK ({'; '.join(lq.issues)})"
            )

        elif lq.status == "WARN":
            # Route based on which specific artifact was detected

            # Baseline wander
            if lq.baseline_wander_ratio > QC.baseline_wander_power_ratio_max:
                actions.append(ProcessingAction.BASELINE_CORRECTION.value)
                decision.processing_route.append(
                    f"{lead_name}: BASELINE_CORRECTION "
                    f"(wander_ratio={lq.baseline_wander_ratio:.3f})"
                )
                any_correction_needed = True

            # Powerline interference
            if lq.powerline_supported and lq.powerline_snr_db > QC.powerline_snr_db_min:
                actions.append(ProcessingAction.POWERLINE_REMOVAL.value)
                decision.processing_route.append(
                    f"{lead_name}: POWERLINE_REMOVAL "
                    f"(snr={lq.powerline_snr_db:.1f} dB)"
                )
                any_correction_needed = True

            # Clipping (no correction possible, just warn)
            if lq.is_clipped:
                decision.processing_route.append(
                    f"{lead_name}: WARN_CLIPPED (no correction applied)"
                )

            # Amplitude issues (no correction, just warn)
            if not lq.is_plausible_amplitude:
                decision.processing_route.append(
                    f"{lead_name}: WARN_AMPLITUDE "
                    f"(range={lq.amplitude_range_mv:.3f} mV)"
                )

            # High-frequency noise
            if lq.hf_noise_ratio > QC.hf_noise_power_ratio_max:
                decision.processing_route.append(
                    f"{lead_name}: WARN_HF_NOISE "
                    f"(ratio={lq.hf_noise_ratio:.3f})"
                )
                # Note: We do NOT apply aggressive low-pass filtering
                # to protect QRS morphology (plan Section 7)

            # If no specific correction identified, just log the warning
            if not actions:
                actions.append(ProcessingAction.MINIMAL.value)
                decision.processing_route.append(
                    f"{lead_name}: MINIMAL (warned but no correction needed)"
                )

        else:
            # PASS → minimal
            actions.append(ProcessingAction.MINIMAL.value)

        decision.per_lead_actions[lead_name] = actions

    # --- Cross-lead physics warnings ---
    if qc_result.cross_lead and not qc_result.cross_lead.all_passed:
        for issue in qc_result.cross_lead.issues:
            decision.processing_route.append(f"CROSS_LEAD_WARN: {issue}")
        # Cross-lead issues create warnings but never auto-relabel leads

    # --- Set overall action ---
    if any_correction_needed:
        decision.overall_action = "corrected"
        decision.requires_correction = True
    else:
        decision.overall_action = "minimal"
        decision.requires_correction = False

    if not decision.processing_route:
        decision.processing_route.append("ALL_PASS: minimal processing")

    return decision


def summarize_routing(
    decisions: List[RoutingDecision],
) -> dict:
    """Summarize routing decisions across a batch of records.

    Returns a dict with counts per action category.
    """
    summary = {
        "total": len(decisions),
        "minimal": 0,
        "corrected": 0,
        "failed": 0,
        "baseline_corrections": 0,
        "powerline_removals": 0,
        "lead_masks": 0,
    }

    for d in decisions:
        if d.overall_action == ProcessingAction.FAIL.value:
            summary["failed"] += 1
        elif d.requires_correction:
            summary["corrected"] += 1
        else:
            summary["minimal"] += 1

        for lead, actions in d.per_lead_actions.items():
            if ProcessingAction.BASELINE_CORRECTION.value in actions:
                summary["baseline_corrections"] += 1
            if ProcessingAction.POWERLINE_REMOVAL.value in actions:
                summary["powerline_removals"] += 1
            if ProcessingAction.LEAD_MASK.value in actions:
                summary["lead_masks"] += 1

    return summary
