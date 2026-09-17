"""
Patient manifest builder and MI label mapper.

Implements plan phases 1–3, 13–14:
- Load ptbxl_database.csv and scp_statements.csv
- Build binary MI superclass label from SCP codes
- Compute label_confidence, hard_negative flag
- Validate patient-aware fold integrity (zero overlap)
- Left-join PTB-XL+ feature tables on ecg_id
- Feature allowlist/denylist audit
- Exact-duplicate waveform hash detection
"""

import ast
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .config import (
    PATHS,
    MI_SCP_CODES,
    HARD_NEGATIVE_SCP_CODES,
    DEV_FOLDS,
    CALIBRATION_FOLD,
    LOCKED_TEST_FOLD,
    CANONICAL_LEAD_ORDER,
    DATASET_VERSION_PTBXL,
    DATASET_VERSION_PTBXLP,
    PIPELINE_VERSION,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature denylist: columns from PTB-XL+ that leak diagnostic information
# and must NEVER be used as model inputs.
# ---------------------------------------------------------------------------

FEATURE_DENYLIST_PATTERNS: List[str] = [
    "diag",          # diagnostic statements / codes
    "statement",     # report text fragments
    "scp_code",      # target-derived
    "report",        # free-text diagnosis
    "reason",        # clinical reason for recording
    "infarction",    # infarction stage (label leakage)
    "validated_by",  # annotation meta
    "likelihood",    # annotation likelihood is stored separately
]


# ===================================================================
# 1. Load PTB-XL metadata
# ===================================================================

def load_ptbxl_database(path: Optional[Path] = None) -> pd.DataFrame:
    """Load ptbxl_database.csv with parsed SCP codes.

    Returns a DataFrame indexed by ecg_id with the scp_codes column
    parsed from string-encoded dicts to actual Python dicts.
    """
    csv_path = path or PATHS.ptbxl_database_csv
    logger.info("Loading PTB-XL database from %s", csv_path)

    df = pd.read_csv(csv_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    logger.info("Loaded %d ECG records", len(df))
    return df


def load_scp_statements(path: Optional[Path] = None) -> pd.DataFrame:
    """Load scp_statements.csv for SCP code → superclass mapping."""
    csv_path = path or PATHS.scp_statements_csv
    logger.info("Loading SCP statements from %s", csv_path)
    df = pd.read_csv(csv_path, index_col=0)
    return df


# ===================================================================
# 2. MI label mapping
# ===================================================================

def build_mi_label(
    df: pd.DataFrame,
    scp_df: pd.DataFrame,
    mi_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add binary MI label, contributing SCP codes, label confidence,
    and hard-negative flag to the manifest.

    Parameters
    ----------
    df : pd.DataFrame
        PTB-XL database indexed by ecg_id, with parsed scp_codes column.
    scp_df : pd.DataFrame
        SCP statements table for superclass lookup.
    mi_codes : list of str, optional
        SCP codes belonging to MI superclass. Defaults to MI_SCP_CODES.

    Returns
    -------
    pd.DataFrame
        Input dataframe augmented with:
        - mi_label: int (0 or 1)
        - mi_scp_codes: list of contributing MI SCP codes
        - label_confidence: float (max likelihood among contributing codes)
        - hard_negative: bool
        - overlapping_superclasses: list of non-MI superclasses present
    """
    if mi_codes is None:
        mi_codes = MI_SCP_CODES

    mi_code_set = set(mi_codes)
    hard_neg_set = set(HARD_NEGATIVE_SCP_CODES)

    mi_labels = []
    mi_contributing_codes = []
    label_confidences = []
    hard_negatives = []
    overlapping_classes = []

    for ecg_id, row in df.iterrows():
        scp_dict = row["scp_codes"]  # e.g. {"NORM": 100.0, "IMI": 80.0}

        # Find MI codes present in this record
        present_mi = {
            code: likelihood
            for code, likelihood in scp_dict.items()
            if code in mi_code_set
        }

        # MI label
        is_mi = 1 if len(present_mi) > 0 else 0
        mi_labels.append(is_mi)
        mi_contributing_codes.append(list(present_mi.keys()))

        # Label confidence = max likelihood among contributing codes
        if present_mi:
            label_confidences.append(max(present_mi.values()) / 100.0)
        else:
            label_confidences.append(1.0)  # confident non-MI

        # Hard negative: non-MI record that has mimicking conditions
        present_codes = set(scp_dict.keys())
        is_hard_neg = (is_mi == 0) and bool(present_codes & hard_neg_set)
        hard_negatives.append(is_hard_neg)

        # Overlapping non-MI superclasses
        # Look up diagnostic_class for each present code
        overlap = set()
        for code in scp_dict.keys():
            if code in scp_df.index and code not in mi_code_set:
                dc = scp_df.loc[code].get("diagnostic_class", None)
                if pd.notna(dc):
                    overlap.add(str(dc))
        overlapping_classes.append(sorted(overlap))

    df = df.copy()
    df["mi_label"] = mi_labels
    df["mi_scp_codes"] = mi_contributing_codes
    df["label_confidence"] = label_confidences
    df["hard_negative"] = hard_negatives
    df["overlapping_superclasses"] = overlapping_classes

    n_mi = sum(mi_labels)
    n_non_mi = len(mi_labels) - n_mi
    n_hard_neg = sum(hard_negatives)
    logger.info(
        "MI label mapping: %d MI, %d non-MI (%d hard negatives)",
        n_mi, n_non_mi, n_hard_neg,
    )

    return df


# ===================================================================
# 3. Patient-aware fold validation
# ===================================================================

def validate_patient_folds(df: pd.DataFrame) -> Dict[str, any]:
    """Verify zero patient_id overlap between dev/val/test folds.

    Returns a dict with fold sizes, patient counts, and any violations.
    Raises ValueError if patient leakage is detected.
    """
    dev_patients = set(
        df[df["strat_fold"].isin(DEV_FOLDS)]["patient_id"].unique()
    )
    val_patients = set(
        df[df["strat_fold"] == CALIBRATION_FOLD]["patient_id"].unique()
    )
    test_patients = set(
        df[df["strat_fold"] == LOCKED_TEST_FOLD]["patient_id"].unique()
    )

    dev_val_overlap = dev_patients & val_patients
    dev_test_overlap = dev_patients & test_patients
    val_test_overlap = val_patients & test_patients

    report = {
        "dev_records": int(df["strat_fold"].isin(DEV_FOLDS).sum()),
        "val_records": int((df["strat_fold"] == CALIBRATION_FOLD).sum()),
        "test_records": int((df["strat_fold"] == LOCKED_TEST_FOLD).sum()),
        "dev_patients": len(dev_patients),
        "val_patients": len(val_patients),
        "test_patients": len(test_patients),
        "dev_val_overlap": len(dev_val_overlap),
        "dev_test_overlap": len(dev_test_overlap),
        "val_test_overlap": len(val_test_overlap),
        "leakage_detected": False,
    }

    if dev_val_overlap or dev_test_overlap or val_test_overlap:
        report["leakage_detected"] = True
        msg = (
            f"PATIENT LEAKAGE DETECTED! "
            f"Dev↔Val: {len(dev_val_overlap)}, "
            f"Dev↔Test: {len(dev_test_overlap)}, "
            f"Val↔Test: {len(val_test_overlap)}"
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        "Fold validation PASSED — Dev: %d patients (%d records), "
        "Val: %d patients (%d records), Test: %d patients (%d records), "
        "Zero overlap.",
        report["dev_patients"], report["dev_records"],
        report["val_patients"], report["val_records"],
        report["test_patients"], report["test_records"],
    )
    return report


# ===================================================================
# 4. PTB-XL+ feature table join
# ===================================================================

def _is_denied_column(col_name: str) -> bool:
    """Check if a column name matches any denylist pattern."""
    col_lower = col_name.lower()
    return any(pat in col_lower for pat in FEATURE_DENYLIST_PATTERNS)


def join_ptbxl_plus_features(
    df: pd.DataFrame,
    load_12sl: bool = True,
    load_ecgdeli: bool = True,
    load_unig: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Left-join PTB-XL+ feature tables on ecg_id.

    Performs a feature allowlist/denylist audit and drops any columns
    that match the denylist patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Manifest indexed by ecg_id.
    load_12sl, load_ecgdeli, load_unig : bool
        Which feature tables to load.

    Returns
    -------
    df_joined : pd.DataFrame
        Manifest with PTB-XL+ features appended.
    audit : dict
        Keys: 'allowed', 'denied' — lists of column names.
    """
    audit = {"allowed": [], "denied": []}

    feature_tables = []
    if load_12sl and PATHS.features_12sl_csv.exists():
        logger.info("Loading 12SL features from %s", PATHS.features_12sl_csv)
        ft = pd.read_csv(PATHS.features_12sl_csv, index_col="ecg_id")
        feature_tables.append(("12sl", ft))

    if load_ecgdeli and PATHS.features_ecgdeli_csv.exists():
        logger.info("Loading ecgdeli features from %s", PATHS.features_ecgdeli_csv)
        ft = pd.read_csv(PATHS.features_ecgdeli_csv, index_col="ecg_id")
        feature_tables.append(("ecgdeli", ft))

    if load_unig and PATHS.features_unig_csv.exists():
        logger.info("Loading unig features from %s", PATHS.features_unig_csv)
        ft = pd.read_csv(PATHS.features_unig_csv, index_col="ecg_id")
        feature_tables.append(("unig", ft))

    for table_name, ft in feature_tables:
        # Audit columns
        denied_cols = [c for c in ft.columns if _is_denied_column(c)]
        allowed_cols = [c for c in ft.columns if not _is_denied_column(c)]
        audit["denied"].extend(denied_cols)
        audit["allowed"].extend(allowed_cols)

        if denied_cols:
            logger.warning(
                "DENYLIST: Dropping %d columns from %s: %s",
                len(denied_cols), table_name, denied_cols,
            )
            ft = ft.drop(columns=denied_cols)

        # Left join
        n_before = len(df)
        df = df.join(ft, how="left", rsuffix=f"_{table_name}")
        n_matched = df[ft.columns[0]].notna().sum() if len(ft.columns) > 0 else 0
        logger.info(
            "Joined %s: %d/%d ecg_ids matched",
            table_name, n_matched, n_before,
        )

    logger.info(
        "Feature audit: %d allowed, %d denied",
        len(audit["allowed"]), len(audit["denied"]),
    )
    return df, audit


# ===================================================================
# 5. Duplicate detection
# ===================================================================

def compute_waveform_hash(signal: np.ndarray) -> str:
    """Compute SHA-256 hash of a waveform array for duplicate detection."""
    return hashlib.sha256(signal.tobytes()).hexdigest()


def detect_exact_duplicates(
    hashes: Dict[int, str],
) -> List[Tuple[int, int]]:
    """Find pairs of ecg_ids with identical waveform hashes.

    Parameters
    ----------
    hashes : dict
        Mapping ecg_id → SHA-256 hash string.

    Returns
    -------
    list of (ecg_id_a, ecg_id_b) duplicate pairs.
    """
    hash_to_ids: Dict[str, List[int]] = {}
    for ecg_id, h in hashes.items():
        hash_to_ids.setdefault(h, []).append(ecg_id)

    duplicates = []
    for h, ids in hash_to_ids.items():
        if len(ids) > 1:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    duplicates.append((ids[i], ids[j]))

    if duplicates:
        logger.warning("Found %d exact duplicate waveform pairs", len(duplicates))
    else:
        logger.info("No exact duplicate waveforms detected")

    return duplicates


# ===================================================================
# 6. Full manifest builder
# ===================================================================

def build_manifest(
    join_features: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """Build the complete patient manifest with MI labels, fold validation,
    and optional PTB-XL+ feature join.

    Returns the manifest DataFrame indexed by ecg_id.
    """
    # Step 1: Load base tables
    df = load_ptbxl_database()
    scp_df = load_scp_statements()

    # Step 2: Build MI label
    df = build_mi_label(df, scp_df)

    # Step 3: Validate patient folds
    fold_report = validate_patient_folds(df)

    # Step 4: Join PTB-XL+ features
    audit = None
    if join_features:
        df, audit = join_ptbxl_plus_features(df)

    # Step 5: Add metadata columns
    df["dataset_version"] = DATASET_VERSION_PTBXL
    df["pipeline_version"] = PIPELINE_VERSION

    # Step 6: Save
    if save:
        out_dir = PATHS.manifests_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "patient_manifest.csv"
        # For columns containing lists, convert to string for CSV
        save_df = df.copy()
        for col in ["mi_scp_codes", "overlapping_superclasses"]:
            if col in save_df.columns:
                save_df[col] = save_df[col].apply(str)
        save_df.to_csv(out_path)
        logger.info("Saved manifest to %s (%d records)", out_path, len(df))

    # Print summary
    print("\n" + "=" * 60)
    print("AQUIRE-Med Patient Manifest Summary")
    print("=" * 60)
    print(f"Total records:     {len(df)}")
    print(f"MI positive:       {df['mi_label'].sum()}")
    print(f"MI negative:       {(df['mi_label'] == 0).sum()}")
    print(f"Hard negatives:    {df['hard_negative'].sum()}")
    print(f"Dev folds (1-8):   {fold_report['dev_records']} records, "
          f"{fold_report['dev_patients']} patients")
    print(f"Val fold (9):      {fold_report['val_records']} records, "
          f"{fold_report['val_patients']} patients")
    print(f"Test fold (10):    {fold_report['test_records']} records, "
          f"{fold_report['test_patients']} patients")
    print(f"Patient leakage:   {'NONE ✓' if not fold_report['leakage_detected'] else 'DETECTED ✗'}")
    if audit:
        print(f"PTB-XL+ features:  {len(audit['allowed'])} allowed, "
              f"{len(audit['denied'])} denied")
    print("=" * 60 + "\n")

    return df
