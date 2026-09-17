"""Immutable PTB-XL manifest, phenotype mapping and PTB-XL+ alignment."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import (
    CANONICAL_LEAD_ORDER,
    CALIBRATION_FOLD,
    DATASET_VERSION_PTBXL,
    DEV_FOLDS,
    EXPECTED_MI_SCP_CODES,
    HARD_NEGATIVE_DIAGNOSTIC_CLASSES,
    LOCKED_TEST_FOLD,
    PATHS,
    PIPELINE_VERSION,
)
from .registry import verify_ptbxl_identity, verify_waveform_references

FORBIDDEN_FEATURE_TOKENS = {
    "diag", "statement", "scp", "report", "reason", "infarction",
    "validated", "likelihood", "snomed", "label", "target", "class",
}
FEATURE_DENYLIST_PATTERNS = sorted(FORBIDDEN_FEATURE_TOKENS)


def load_ptbxl_database(path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = Path(path or PATHS.ptbxl_database_csv)
    df = pd.read_csv(csv_path, index_col="ecg_id")
    if not df.index.is_unique:
        raise ValueError("PTB-XL ecg_id is not unique")
    df["scp_codes"] = df["scp_codes"].apply(
        lambda value: ast.literal_eval(value) if isinstance(value, str) else value
    )
    return df


def load_scp_statements(path: Optional[Path] = None) -> pd.DataFrame:
    df = pd.read_csv(Path(path or PATHS.scp_statements_csv), index_col=0)
    if not df.index.is_unique:
        raise ValueError("scp_statements code index is not unique")
    return df


def derive_mi_codes(scp_df: pd.DataFrame, assert_release: bool = True) -> List[str]:
    if "diagnostic_class" not in scp_df.columns:
        raise ValueError("scp_statements.csv lacks diagnostic_class")
    codes = sorted(scp_df.index[scp_df["diagnostic_class"].eq("MI")].astype(str))
    if assert_release and set(codes) != set(EXPECTED_MI_SCP_CODES):
        raise ValueError(
            "Mounted SCP-to-MI mapping does not match PTB-XL v1.0.3. "
            f"Expected {sorted(EXPECTED_MI_SCP_CODES)}, found {codes}"
        )
    return codes


def _diagnostic_groups(codes: Iterable[str], scp_df: pd.DataFrame) -> List[str]:
    groups = set()
    for code in codes:
        if code in scp_df.index:
            group = scp_df.at[code, "diagnostic_class"]
            if pd.notna(group):
                groups.add(str(group))
    return sorted(groups)


def _metadata_bool(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def build_mi_label(
    df: pd.DataFrame,
    scp_df: pd.DataFrame,
    mi_codes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Create MI/non-MI labels without treating annotation certainty as risk."""
    mi_set = set(mi_codes or derive_mi_codes(scp_df))
    rows = []
    for _, row in df.iterrows():
        scp = row["scp_codes"]
        if not isinstance(scp, dict):
            scp = {}
        present_mi = {str(code): float(value) for code, value in scp.items() if code in mi_set}
        groups = _diagnostic_groups(scp.keys(), scp_df)
        non_mi_groups = sorted(set(groups) - {"MI"})
        likelihoods = list(present_mi.values())
        likelihood_known = bool(likelihoods) and any(v > 0 for v in likelihoods)
        likelihood_max = max(likelihoods) / 100.0 if likelihood_known else np.nan
        mi_label = int(bool(present_mi))
        hard_groups = sorted(set(non_mi_groups) & set(HARD_NEGATIVE_DIAGNOSTIC_CLASSES))
        validated_human = _metadata_bool(row.get("validated_by_human", False))
        second_opinion = _metadata_bool(row.get("second_opinion", False))
        if mi_label and likelihood_known and likelihood_max >= 0.8 and validated_human:
            quality = "high"
        elif mi_label and likelihood_known:
            quality = "supported"
        elif mi_label:
            quality = "uncertain"
        else:
            quality = "negative_unquantified"
        rows.append({
            "mi_label": mi_label,
            "mi_scp_codes": sorted(present_mi),
            "annotation_likelihood_max": likelihood_max,
            "annotation_likelihood_known": likelihood_known,
            "label_quality_group": quality,
            "second_opinion_flag": second_opinion,
            "validated_by_human_flag": validated_human,
            "diagnostic_superclasses": groups,
            "overlapping_superclasses": non_mi_groups,
            "hard_negative_groups": hard_groups,
            "hard_negative": bool(not mi_label and hard_groups),
        })
    return pd.concat([df.copy(), pd.DataFrame(rows, index=df.index)], axis=1)


def split_role(fold: int) -> str:
    fold = int(fold)
    if fold in DEV_FOLDS:
        return "development"
    if fold == CALIBRATION_FOLD:
        return "calibration"
    if fold == LOCKED_TEST_FOLD:
        return "locked_test"
    raise ValueError(f"Invalid PTB-XL strat_fold: {fold}")


def validate_patient_folds(df: pd.DataFrame) -> Dict[str, object]:
    groups = {
        "dev": set(df.loc[df.strat_fold.isin(DEV_FOLDS), "patient_id"]),
        "cal": set(df.loc[df.strat_fold.eq(CALIBRATION_FOLD), "patient_id"]),
        "test": set(df.loc[df.strat_fold.eq(LOCKED_TEST_FOLD), "patient_id"]),
    }
    overlaps = {
        "dev_cal": groups["dev"] & groups["cal"],
        "dev_test": groups["dev"] & groups["test"],
        "cal_test": groups["cal"] & groups["test"],
    }
    if any(overlaps.values()):
        raise ValueError("Patient leakage detected: " + str({k: len(v) for k, v in overlaps.items()}))
    return {
        "dev_records": int(df.strat_fold.isin(DEV_FOLDS).sum()),
        "val_records": int(df.strat_fold.eq(CALIBRATION_FOLD).sum()),
        "test_records": int(df.strat_fold.eq(LOCKED_TEST_FOLD).sum()),
        "dev_patients": len(groups["dev"]),
        "val_patients": len(groups["cal"]),
        "test_patients": len(groups["test"]),
        "dev_val_overlap": 0,
        "dev_test_overlap": 0,
        "val_test_overlap": 0,
        "leakage_detected": False,
    }


def guard_fold_access(folds: Iterable[int], purpose: str) -> None:
    folds = {int(v) for v in folds}
    if LOCKED_TEST_FOLD in folds and purpose not in {"final_locked_evaluation", "data_integrity"}:
        raise PermissionError(f"Fold 10 access blocked for purpose={purpose!r}")
    if CALIBRATION_FOLD in folds and purpose in {"tuning", "feature_selection", "qc_calibration"}:
        raise PermissionError(f"Fold 9 access blocked for purpose={purpose!r}")


def _feature_description_allowlist(path: Path) -> Optional[set[str]]:
    if not path.exists():
        return None
    description = pd.read_csv(path)
    if "id" not in description.columns:
        raise ValueError("feature_description.csv has no canonical 'id' column")
    values: set[str] = set()
    for value in description["id"].dropna().astype(str):
        if value.endswith("_X"):
            values.update(value[:-1] + lead for lead in CANONICAL_LEAD_ORDER)
        else:
            values.add(value)
    return values


def _is_forbidden_feature(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in FORBIDDEN_FEATURE_TOKENS)


def _is_denied_column(name: str) -> bool:
    """Compatibility alias retained for callers and leakage tests."""
    return _is_forbidden_feature(name)


def join_ptbxl_plus_features(
    df: pd.DataFrame,
    load_12sl: bool = True,
    load_ecgdeli: bool = True,
    load_unig: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Align source-prefixed reference features with cardinality checks."""
    described_names = _feature_description_allowlist(PATHS.feature_description_csv)
    specs = [
        ("12sl", PATHS.features_12sl_csv, load_12sl, "benchmark_only"),
        ("ecgdeli", PATHS.features_ecgdeli_csv, load_ecgdeli, "open_reference"),
        ("unig", PATHS.features_unig_csv, load_unig, "benchmark_only"),
    ]
    audit: Dict[str, object] = {"sources": {}, "allowed": [], "denied": []}
    joined = df.copy()
    original_index = joined.index.copy()
    for source, path, enabled, role in specs:
        if not enabled or not path.exists():
            continue
        if described_names is None:
            raise FileNotFoundError(
                "PTB-XL+ feature_description.csv is required for the explicit allowlist"
            )
        table = pd.read_csv(path, index_col="ecg_id")
        if not table.index.is_unique:
            raise ValueError(f"PTB-XL+ {source} ecg_id is not unique")
        denied = [c for c in table.columns if _is_forbidden_feature(c)]
        allowed = [c for c in table.columns if c not in denied]
        selected = [c for c in allowed if c in described_names]
        undescribed = sorted(set(allowed) - set(selected))
        if not selected:
            raise ValueError(f"Explicit PTB-XL+ allowlist selected no {source} features")
        table = table[selected].add_prefix(f"ref_{source}__")
        joined = joined.join(table, how="left", validate="one_to_one")
        matched = int(table.index.intersection(df.index).nunique())
        audit["sources"][source] = {
            "rows": int(len(table)), "matched": matched,
            "coverage": matched / len(df) if len(df) else 0.0,
            "allowed_count": len(selected), "denied": denied,
            "undescribed_count": len(undescribed),
            "undescribed_examples": undescribed[:20],
            "deployment_role": role,
        }
        audit["allowed"].extend(table.columns.tolist())
        audit["denied"].extend(f"{source}:{name}" for name in denied)
    if len(joined) != len(df) or not joined.index.equals(original_index):
        raise AssertionError("PTB-XL+ join changed manifest cardinality or order")
    return joined, audit


def compute_waveform_hash(signal: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(signal).tobytes()).hexdigest()


def detect_exact_duplicates(hashes: Dict[int, str]) -> List[Tuple[int, int]]:
    reverse: Dict[str, List[int]] = {}
    for ecg_id, digest in hashes.items():
        reverse.setdefault(digest, []).append(int(ecg_id))
    return [
        (ids[i], ids[j])
        for ids in reverse.values() if len(ids) > 1
        for i in range(len(ids)) for j in range(i + 1, len(ids))
    ]


def build_manifest(
    join_features: bool = False,
    save: bool = True,
    verify_release: bool = True,
) -> pd.DataFrame:
    if verify_release:
        verify_ptbxl_identity(PATHS.ptbxl_root, strict=True)
    df = load_ptbxl_database()
    scp = load_scp_statements()
    df = build_mi_label(df, scp)
    fold_report = validate_patient_folds(df)
    df["split_role"] = df["strat_fold"].map(split_role)
    df["dataset_version"] = DATASET_VERSION_PTBXL
    df["pipeline_version"] = PIPELINE_VERSION
    waveform_report = verify_waveform_references(PATHS.ptbxl_root, df)
    if not waveform_report["verified"]:
        raise FileNotFoundError(
            f"{waveform_report['missing_pair_count']} manifest waveform references are unresolved"
        )
    audit = None
    if join_features:
        df, audit = join_ptbxl_plus_features(df)
    if save:
        PATHS.manifests_dir.mkdir(parents=True, exist_ok=True)
        serial = df.copy()
        for column in [
            "scp_codes", "mi_scp_codes", "diagnostic_superclasses",
            "overlapping_superclasses", "hard_negative_groups",
        ]:
            if column in serial:
                serial[column] = serial[column].map(json.dumps)
        serial.to_csv(PATHS.manifests_dir / "patient_manifest.csv")
        (PATHS.manifests_dir / "fold_report.json").write_text(json.dumps(fold_report, indent=2))
        (PATHS.manifests_dir / "waveform_reference_report.json").write_text(
            json.dumps(waveform_report, indent=2)
        )
        if audit is not None:
            (PATHS.manifests_dir / "ptbxl_plus_audit.json").write_text(json.dumps(audit, indent=2))
    return df
