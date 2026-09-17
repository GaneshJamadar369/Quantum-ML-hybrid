"""
AQUIRE-Med Signal & Data Preprocessing Pipeline
=================================================
SIH 26139 — Morphology-preserving, quality-aware ECG preprocessing
for MI-pattern detection from 12-lead ECGs.

Primary dataset: PTB-XL v1.0.3
Companion:       PTB-XL+ v1.0.1
"""

__version__ = "0.2.0"
__pipeline_version__ = "aquire-preproc-v0.2.0"

from .contracts import Eligibility, GateState, ProcessedECG, ValidatedECG

__all__ = ["Eligibility", "GateState", "ProcessedECG", "ValidatedECG"]
