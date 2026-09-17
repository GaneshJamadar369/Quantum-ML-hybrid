"""
AQUIRE-Med Preprocessing Pipeline — Kaggle Notebook Version
============================================================
SIH 26139 — Morphology-preserving, quality-aware ECG preprocessing
for MI-pattern detection from 12-lead ECGs.

Datasets required (attach via + Add Input):
  1. khyeh0719/ptb-xl-dataset
  2. bjoernjostein/ptb-xl

Enable GPU: Notebook options → Accelerator → GPU T4 x2
"""

# =============================================================================
# CELL 1: Imports & Configuration
# =============================================================================
import os
import ast
import json
import time
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wfdb
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt, iirnotch, find_peaks, welch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aquire")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PIPELINE_VERSION = "aquire-preproc-v0.1.0"
DATASET_VERSION_PTBXL = "1.0.3"

# Paths (Kaggle)
PTBXL_ROOT = Path("/kaggle/input/ptb-xl-dataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1")
PTBXLP_ROOT = Path("/kaggle/input/ptb-xl/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1")
OUTPUT_ROOT = Path("/kaggle/working/processed")

# Canonical lead order
CANONICAL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
NUM_LEADS = 12
SAMPLES_100HZ = 1000
FS = 100

# Lead name aliases
LEAD_ALIASES = {
    "i": "I", "ii": "II", "iii": "III",
    "avr": "aVR", "avl": "aVL", "avf": "aVF",
    "v1": "V1", "v2": "V2", "v3": "V3", "v4": "V4", "v5": "V5", "v6": "V6",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    "I": "I", "II": "II", "III": "III",
    "aVR": "aVR", "aVL": "aVL", "aVF": "aVF",
    "V1": "V1", "V2": "V2", "V3": "V3", "V4": "V4", "V5": "V5", "V6": "V6",
}

# MI superclass SCP codes
MI_SCP_CODES = ["IMI", "AMI", "ASMI", "ILMI", "IPMI", "INJAL", "INJAS", "INJIN", "INJLA", "PMI", "LMI"]
HARD_NEG_CODES = ["STTC", "NST_", "ISCA", "ISCI", "LVH", "LBBB"]

# Fold assignments
DEV_FOLDS = [1, 2, 3, 4, 5, 6, 7, 8]
CAL_FOLD = 9
TEST_FOLD = 10

# QC thresholds (frozen)
FLATLINE_MIN_MS = 200.0
FLATLINE_VAR_EPS = 1e-6
CLIPPING_FRAC_MAX = 0.01
MISSING_FRAC_MAX = 0.05
AMP_MIN_MV = 0.05
AMP_MAX_MV = 6.0
BW_POWER_MAX = 0.4
PL_SNR_MIN = 10.0
HF_POWER_MAX = 0.3
EINTHOVEN_MAX_MV = 0.15
GOLDBERGER_MAX_MV = 0.15

# Morphology tolerances (frozen)
MAX_RPEAK_SHIFT = 1       # samples (10 ms at 100 Hz)
MAX_ST_SHIFT_MV = 0.05    # millivolts
MAX_QRS_WIDTH_MS = 10.0   # milliseconds
MAX_QRS_AMP_MV = 0.10     # millivolts

# Feature denylist patterns (columns that leak diagnostic info)
DENYLIST = ["diag", "statement", "scp_code", "report", "reason", "infarction", "validated_by", "likelihood"]

print("✓ Configuration loaded")
print(f"  PTB-XL:  {PTBXL_ROOT}")
print(f"  PTB-XL+: {PTBXLP_ROOT}")
print(f"  Output:  {OUTPUT_ROOT}")

# =============================================================================
# CELL 2: Verify Dataset Access
# =============================================================================
print("\n" + "="*60)
print("Dataset Verification")
print("="*60)

# Check PTB-XL
ptbxl_csv = PTBXL_ROOT / "ptbxl_database.csv"
scp_csv = PTBXL_ROOT / "scp_statements.csv"
records_dir = PTBXL_ROOT / "records100"

assert ptbxl_csv.exists(), f"Missing: {ptbxl_csv}"
assert scp_csv.exists(), f"Missing: {scp_csv}"
assert records_dir.exists(), f"Missing: {records_dir}"
print(f"✓ PTB-XL database:    {ptbxl_csv}")
print(f"✓ SCP statements:     {scp_csv}")
print(f"✓ Records (100Hz):    {records_dir}")

# Check PTB-XL+
feat_12sl = PTBXLP_ROOT / "features" / "12sl_features.csv"
feat_ecgdeli = PTBXLP_ROOT / "features" / "ecgdeli_features.csv"
feat_desc = PTBXLP_ROOT / "features" / "feature_description.csv"

assert feat_12sl.exists(), f"Missing: {feat_12sl}"
assert feat_ecgdeli.exists(), f"Missing: {feat_ecgdeli}"
print(f"✓ 12SL features:      {feat_12sl}")
print(f"✓ ECGDeli features:   {feat_ecgdeli}")
print(f"✓ Feature description: {feat_desc}")
print("="*60)

# =============================================================================
# CELL 3: Build Patient Manifest & MI Labels
# =============================================================================
print("\n" + "="*60)
print("Phase 1-3: Building Patient Manifest")
print("="*60)

# Load base tables
df = pd.read_csv(ptbxl_csv, index_col="ecg_id")
df.scp_codes = df.scp_codes.apply(ast.literal_eval)
scp_df = pd.read_csv(scp_csv, index_col=0)
print(f"Loaded {len(df)} ECG records")
print(f"Loaded {len(scp_df)} SCP code definitions")

# Build MI label
mi_code_set = set(MI_SCP_CODES)
hard_neg_set = set(HARD_NEG_CODES)

mi_labels, mi_codes_list, confidences, hard_negs = [], [], [], []
for ecg_id, row in df.iterrows():
    scp_dict = row["scp_codes"]
    present_mi = {c: l for c, l in scp_dict.items() if c in mi_code_set}
    
    is_mi = 1 if present_mi else 0
    mi_labels.append(is_mi)
    mi_codes_list.append(list(present_mi.keys()))
    confidences.append(max(present_mi.values()) / 100.0 if present_mi else 1.0)
    hard_negs.append(is_mi == 0 and bool(set(scp_dict.keys()) & hard_neg_set))

df["mi_label"] = mi_labels
df["mi_scp_codes"] = mi_codes_list
df["label_confidence"] = confidences
df["hard_negative"] = hard_negs

n_mi = sum(mi_labels)
n_hard = sum(hard_negs)
print(f"\nMI Label Distribution:")
print(f"  MI positive:    {n_mi} ({n_mi/len(df)*100:.1f}%)")
print(f"  MI negative:    {len(df)-n_mi} ({(len(df)-n_mi)/len(df)*100:.1f}%)")
print(f"  Hard negatives: {n_hard}")

# Validate patient folds (zero leakage)
dev_patients = set(df[df["strat_fold"].isin(DEV_FOLDS)]["patient_id"].unique())
val_patients = set(df[df["strat_fold"] == CAL_FOLD]["patient_id"].unique())
test_patients = set(df[df["strat_fold"] == TEST_FOLD]["patient_id"].unique())

assert len(dev_patients & val_patients) == 0, "LEAKAGE: Dev ↔ Val!"
assert len(dev_patients & test_patients) == 0, "LEAKAGE: Dev ↔ Test!"
assert len(val_patients & test_patients) == 0, "LEAKAGE: Val ↔ Test!"

print(f"\nFold Validation (ZERO patient overlap ✓):")
print(f"  Dev (1-8):   {df['strat_fold'].isin(DEV_FOLDS).sum()} records, {len(dev_patients)} patients")
print(f"  Val (9):     {(df['strat_fold']==CAL_FOLD).sum()} records, {len(val_patients)} patients")
print(f"  Test (10):   {(df['strat_fold']==TEST_FOLD).sum()} records, {len(test_patients)} patients")

# =============================================================================
# CELL 4: Join PTB-XL+ Features
# =============================================================================
print("\n" + "="*60)
print("Phase 3: Joining PTB-XL+ Features")
print("="*60)

def is_denied(col):
    return any(p in col.lower() for p in DENYLIST)

# Load 12SL features
df_12sl = pd.read_csv(feat_12sl, index_col="ecg_id")
denied_12sl = [c for c in df_12sl.columns if is_denied(c)]
allowed_12sl = [c for c in df_12sl.columns if not is_denied(c)]
if denied_12sl:
    print(f"  DENYLIST: Dropping {len(denied_12sl)} columns from 12SL: {denied_12sl[:5]}...")
    df_12sl = df_12sl.drop(columns=denied_12sl)

# Load ECGDeli features
df_ecgdeli = pd.read_csv(feat_ecgdeli, index_col="ecg_id")
denied_deli = [c for c in df_ecgdeli.columns if is_denied(c)]
allowed_deli = [c for c in df_ecgdeli.columns if not is_denied(c)]
if denied_deli:
    print(f"  DENYLIST: Dropping {len(denied_deli)} columns from ECGDeli: {denied_deli[:5]}...")
    df_ecgdeli = df_ecgdeli.drop(columns=denied_deli)

# Join
df = df.join(df_12sl, how="left", rsuffix="_12sl")
df = df.join(df_ecgdeli, how="left", rsuffix="_deli")

print(f"✓ 12SL features:   {len(allowed_12sl)} allowed, {len(denied_12sl)} denied")
print(f"✓ ECGDeli features: {len(allowed_deli)} allowed, {len(denied_deli)} denied")
print(f"✓ Combined manifest: {len(df)} records × {df.shape[1]} columns")

# =============================================================================
# CELL 5: Structural Validation & WFDB Loading
# =============================================================================
print("\n" + "="*60)
print("Phase 2: Structural Validation (WFDB Loading)")
print("="*60)

def resolve_record_path(ecg_id, fs=100):
    folder = f"{(ecg_id // 1000) * 1000:05d}"
    suffix = "lr" if fs == 100 else "hr"
    return str(PTBXL_ROOT / f"records{fs}" / folder / f"{ecg_id:05d}_{suffix}")

def canonicalize_leads(names):
    canonical, unknown = [], []
    for n in names:
        clean = n.strip()
        resolved = LEAD_ALIASES.get(clean)
        if resolved is None:
            unknown.append(clean)
            canonical.append(clean)
        else:
            canonical.append(resolved)
    return canonical, unknown

def validate_and_load(ecg_id, fs=100):
    """Load and validate a single ECG record → float32[12, 1000]."""
    path = resolve_record_path(ecg_id, fs)
    try:
        record = wfdb.rdrecord(path)
    except Exception as e:
        return None, f"WFDB error: {e}"
    
    if record.n_sig != NUM_LEADS:
        return None, f"Expected {NUM_LEADS} leads, got {record.n_sig}"
    
    canonical, unknown = canonicalize_leads(record.sig_name)
    if unknown:
        return None, f"Unknown leads: {unknown}"
    
    signal = record.p_signal  # (samples, leads) in mV
    if signal is None:
        return None, "Signal is None"
    
    n_samples = signal.shape[0]
    if n_samples != SAMPLES_100HZ:
        if n_samples > SAMPLES_100HZ:
            signal = signal[:SAMPLES_100HZ, :]
        else:
            signal = np.pad(signal, ((0, SAMPLES_100HZ - n_samples), (0, 0)))
    
    # Replace NaN/Inf
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Reorder to canonical
    reorder = [canonical.index(lead) for lead in CANONICAL_LEADS]
    signal = signal[:, reorder]
    
    # Transpose to (12, 1000) and cast
    return signal.T.astype(np.float32), None

# Test with first 5 records
print("Testing structural validation on first 5 records...")
for eid in [1, 2, 3, 4, 5]:
    sig, err = validate_and_load(eid)
    if err:
        print(f"  ecg_id {eid}: ✗ {err}")
    else:
        print(f"  ecg_id {eid}: ✓ shape={sig.shape}, range=[{sig.min():.3f}, {sig.max():.3f}] mV")

# =============================================================================
# CELL 6: Quality Assessment (Lead-Level + Cross-Lead Physics)
# =============================================================================
print("\n" + "="*60)
print("Phase 4-5: Quality Assessment")
print("="*60)

def check_flatline(lead, fs=100):
    min_samples = int(FLATLINE_MIN_MS / 1000.0 * fs)
    if np.var(lead) < FLATLINE_VAR_EPS:
        return True
    for start in range(0, len(lead) - min_samples + 1, min_samples // 2):
        if np.var(lead[start:start+min_samples]) < FLATLINE_VAR_EPS:
            return True
    return False

def check_clipping(lead):
    n = len(lead)
    return (np.sum(lead == np.min(lead)) / n > CLIPPING_FRAC_MAX or
            np.sum(lead == np.max(lead)) / n > CLIPPING_FRAC_MAX)

def check_baseline_wander(lead, fs=100):
    if len(lead) < fs:
        return 0.0
    freqs, psd = welch(lead, fs=fs, nperseg=min(256, len(lead)))
    total = np.sum(psd)
    if total < 1e-12:
        return 0.0
    return np.sum(psd[freqs < 0.5]) / total

def check_powerline(lead, fs=100):
    if len(lead) < fs:
        return 0.0
    freqs, psd = welch(lead, fs=fs, nperseg=min(256, len(lead)))
    snr = 0.0
    median_p = np.median(psd)
    if median_p < 1e-12:
        return 0.0
    for f in [50.0]:
        if f < fs / 2:
            idx = np.argmin(np.abs(freqs - f))
            s = 10 * np.log10(psd[idx] / median_p)
            snr = max(snr, s)
    return snr

def check_amplitude(lead):
    amp = float(np.max(lead) - np.min(lead))
    return amp, AMP_MIN_MV <= amp <= AMP_MAX_MV

def check_einthoven(signal):
    """II ≈ I + III"""
    return float(np.mean(np.abs(signal[1] - (signal[0] + signal[2]))))

def check_goldberger(signal):
    """aVR ≈ -(I+II)/2, aVL ≈ I-II/2, aVF ≈ II-I/2"""
    avr_r = float(np.mean(np.abs(signal[3] - (-(signal[0]+signal[1])/2))))
    avl_r = float(np.mean(np.abs(signal[4] - (signal[0]-signal[1]/2))))
    avf_r = float(np.mean(np.abs(signal[5] - (signal[1]-signal[0]/2))))
    return avr_r, avl_r, avf_r

def assess_quality(signal, ecg_id):
    """Full QC assessment → dict with status, per_lead, cross_lead."""
    lead_mask = np.ones(NUM_LEADS, dtype=bool)
    n_fail, n_warn = 0, 0
    per_lead = {}
    issues = []
    
    for i, name in enumerate(CANONICAL_LEADS):
        lead = signal[i]
        flat = check_flatline(lead)
        clip = check_clipping(lead)
        bw = check_baseline_wander(lead)
        pl = check_powerline(lead)
        amp, amp_ok = check_amplitude(lead)
        
        if flat:
            status = "FAIL"
            lead_mask[i] = False
            n_fail += 1
        elif clip or not amp_ok or bw > BW_POWER_MAX or pl > PL_SNR_MIN:
            status = "WARN"
            n_warn += 1
        else:
            status = "PASS"
        
        per_lead[name] = {
            "status": status, "flat": flat, "clip": clip,
            "bw_ratio": round(bw, 4), "pl_snr": round(pl, 1),
            "amp_mv": round(amp, 3), "amp_ok": amp_ok,
        }
    
    # Cross-lead physics
    einth = check_einthoven(signal)
    avr_r, avl_r, avf_r = check_goldberger(signal)
    physics_ok = (einth < EINTHOVEN_MAX_MV and 
                  avr_r < GOLDBERGER_MAX_MV and
                  avl_r < GOLDBERGER_MAX_MV and
                  avf_r < GOLDBERGER_MAX_MV)
    
    if n_fail >= 3:
        overall = "FAIL"
    elif n_fail > 0 or n_warn > 0 or not physics_ok:
        overall = "WARN"
    else:
        overall = "PASS"
    
    return {
        "ecg_id": ecg_id,
        "qc_status": overall,
        "n_failed": n_fail,
        "n_warned": n_warn,
        "lead_mask": lead_mask,
        "per_lead": per_lead,
        "einthoven_mv": round(einth, 4),
        "physics_ok": physics_ok,
    }

# Test QC on a few records
for eid in [1, 100, 500]:
    sig, err = validate_and_load(eid)
    if sig is not None:
        qc = assess_quality(sig, eid)
        print(f"  ecg_id {eid}: QC={qc['qc_status']}, "
              f"Failed={qc['n_failed']}, Warned={qc['n_warned']}, "
              f"Einthoven={qc['einthoven_mv']:.4f} mV")

# =============================================================================
# CELL 7: Artifact-Aware Router + Morphology-Preserving Filters
# =============================================================================
print("\n" + "="*60)
print("Phase 6-7: Router + Filters")
print("="*60)

def correct_baseline(lead, fs=100, cutoff=0.05):
    """Zero-phase Butterworth high-pass at 0.05 Hz (conservative)."""
    nyq = fs / 2.0
    if cutoff >= nyq:
        return lead.copy()
    b, a = butter(2, cutoff / nyq, btype="high")
    try:
        return filtfilt(b, a, lead, padtype="odd", padlen=6).astype(np.float32)
    except ValueError:
        return lead.copy()

def remove_powerline_noise(lead, fs=100, freq=50.0, Q=30.0):
    """Zero-phase IIR notch at 50 Hz."""
    if freq >= fs / 2:
        return lead.copy()
    b, a = iirnotch(freq, Q, fs)
    try:
        return filtfilt(b, a, lead).astype(np.float32)
    except ValueError:
        return lead.copy()

def route_and_correct(signal, qc_result):
    """Apply router-indicated corrections → View B."""
    corrected = signal.copy()
    route_log = []
    any_correction = False
    
    if qc_result["qc_status"] == "FAIL":
        route_log.append("RECORD_FAIL")
        return corrected, route_log, False
    
    for i, name in enumerate(CANONICAL_LEADS):
        lq = qc_result["per_lead"][name]
        
        if lq["status"] == "FAIL":
            route_log.append(f"{name}: MASKED")
            continue
        
        if lq["bw_ratio"] > BW_POWER_MAX:
            corrected[i] = correct_baseline(corrected[i])
            route_log.append(f"{name}: BASELINE_CORRECTED")
            any_correction = True
        
        if lq["pl_snr"] > PL_SNR_MIN:
            corrected[i] = remove_powerline_noise(corrected[i])
            route_log.append(f"{name}: POWERLINE_REMOVED")
            any_correction = True
    
    if not route_log:
        route_log.append("ALL_CLEAN: minimal")
    
    return corrected, route_log, any_correction

print("✓ Router and filters ready")

# =============================================================================
# CELL 8: Morphology-Preservation Gate
# =============================================================================
print("\n" + "="*60)
print("Phase 9: Morphology Gate")
print("="*60)

def detect_r_peaks(lead, fs=100):
    min_dist = int(0.2 * fs)
    height_th = 0.5 * np.std(lead)
    peaks, _ = find_peaks(lead, distance=min_dist, height=height_th)
    return peaks

def measure_st_level(lead, peak_idx, fs=100):
    st_point = peak_idx + int(0.14 * fs)  # J+60ms
    if st_point >= len(lead):
        return 0.0
    w = int(0.02 * fs)
    return float(np.mean(lead[max(0,st_point-w):min(len(lead),st_point+w)]))

def morphology_gate(view_a, view_b, ecg_id):
    """Check if correction preserved diagnostic morphology."""
    if np.allclose(view_a, view_b, atol=1e-7):
        return True, "no_correction", 0.0
    
    violations = 0
    for i in range(NUM_LEADS):
        peaks_a = detect_r_peaks(view_a[i])
        peaks_b = detect_r_peaks(view_b[i])
        
        if len(peaks_a) > 0 and len(peaks_b) > 0:
            # R-peak shift
            n = min(len(peaks_a), len(peaks_b))
            for j in range(n):
                shift = min(np.abs(peaks_b - peaks_a[j]))
                if shift > MAX_RPEAK_SHIFT:
                    violations += 1
                    break
            
            # ST-level shift
            st_a = measure_st_level(view_a[i], peaks_a[0])
            st_b = measure_st_level(view_b[i], peaks_b[0])
            if abs(st_b - st_a) > MAX_ST_SHIFT_MV:
                violations += 1
    
    passed = violations == 0
    action = "accepted" if passed else "rejected"
    return passed, action, violations

print("✓ Morphology gate ready")

# =============================================================================
# CELL 9: Process ALL Records
# =============================================================================
print("\n" + "="*60)
print("FULL PIPELINE EXECUTION")
print("="*60)

ecg_ids = df.index.tolist()
n_total = len(ecg_ids)

# Storage
all_minimal = []
all_corrected = []
all_meta = []
n_pass, n_warn, n_fail, n_struct_fail, n_gate_reject = 0, 0, 0, 0, 0

start_time = time.time()

for idx, ecg_id in enumerate(ecg_ids):
    if (idx + 1) % 2000 == 0 or idx == 0:
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        print(f"  Processing {idx+1}/{n_total} ({rate:.0f} rec/sec)...")
    
    # 1. Structural validation
    signal, err = validate_and_load(ecg_id)
    if signal is None:
        n_struct_fail += 1
        continue
    
    view_a = signal.copy()  # View A: minimal
    
    # 2. Quality assessment
    qc = assess_quality(signal, ecg_id)
    
    # 3. Router + correction → View B
    view_b, route_log, corrected = route_and_correct(signal, qc)
    
    # 4. Morphology gate
    gate_action = "no_correction"
    if corrected:
        gate_passed, gate_action, violations = morphology_gate(view_a, view_b, ecg_id)
        if not gate_passed:
            view_b = view_a.copy()  # Fall back to View A
            gate_action = "rejected"
            n_gate_reject += 1
    
    # 5. Track QC status
    if qc["qc_status"] == "PASS":
        n_pass += 1
    elif qc["qc_status"] == "WARN":
        n_warn += 1
    else:
        n_fail += 1
    
    # 6. Store
    all_minimal.append(view_a)
    all_corrected.append(view_b)
    
    row = df.loc[ecg_id]
    all_meta.append({
        "ecg_id": ecg_id,
        "patient_id": int(row["patient_id"]),
        "mi_label": int(row["mi_label"]),
        "label_confidence": float(row["label_confidence"]),
        "hard_negative": bool(row["hard_negative"]),
        "strat_fold": int(row["strat_fold"]),
        "qc_status": qc["qc_status"],
        "einthoven_mv": qc["einthoven_mv"],
        "gate_action": gate_action,
        "route": "; ".join(route_log),
        "source_hash": hashlib.sha256(signal.tobytes()).hexdigest()[:16],
    })

elapsed = time.time() - start_time

# =============================================================================
# CELL 10: Save Outputs & Print Summary
# =============================================================================
print("\n" + "="*60)
print("SAVING OUTPUTS")
print("="*60)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Stack into arrays
signals_minimal = np.stack(all_minimal, axis=0)   # (N, 12, 1000)
signals_corrected = np.stack(all_corrected, axis=0)

np.save(OUTPUT_ROOT / "signals_minimal.npy", signals_minimal)
np.save(OUTPUT_ROOT / "signals_corrected.npy", signals_corrected)
print(f"✓ Signals minimal:   {signals_minimal.shape}")
print(f"✓ Signals corrected: {signals_corrected.shape}")

# Save metadata
meta_df = pd.DataFrame(all_meta)
meta_df.to_csv(OUTPUT_ROOT / "processing_metadata.csv", index=False)
print(f"✓ Metadata:          {len(meta_df)} rows")

# Save manifest
save_df = df.copy()
for col in ["mi_scp_codes"]:
    if col in save_df.columns:
        save_df[col] = save_df[col].apply(str)
save_df.to_csv(OUTPUT_ROOT / "patient_manifest.csv")
print(f"✓ Manifest:          {len(save_df)} rows")

# =============================================================================
# CELL 11: Final Summary
# =============================================================================
print("\n" + "="*60)
print("AQUIRE-Med Preprocessing Pipeline — COMPLETE")
print("="*60)
print(f"Total attempted:      {n_total}")
print(f"Successfully processed: {len(all_meta)}")
print(f"Structural failures:  {n_struct_fail}")
print(f"QC PASS:              {n_pass}")
print(f"QC WARN:              {n_warn}")
print(f"QC FAIL:              {n_fail}")
print(f"Gate rejections:      {n_gate_reject}")
print(f"Processing time:      {elapsed:.1f}s ({len(all_meta)/elapsed:.0f} rec/sec)")
print(f"Output shape:         {signals_minimal.shape}")
print(f"\nMI Label Distribution (processed):")
mi_processed = sum(m["mi_label"] for m in all_meta)
print(f"  MI=1: {mi_processed} ({mi_processed/len(all_meta)*100:.1f}%)")
print(f"  MI=0: {len(all_meta)-mi_processed} ({(len(all_meta)-mi_processed)/len(all_meta)*100:.1f}%)")
print(f"\nFold Distribution (processed):")
for fold in sorted(set(m["strat_fold"] for m in all_meta)):
    n = sum(1 for m in all_meta if m["strat_fold"] == fold)
    print(f"  Fold {fold}: {n} records")
print("="*60)
print("✓ All outputs saved to /kaggle/working/processed/")
print("✓ Ready for Milestone 2 (Classical Model Training)")
