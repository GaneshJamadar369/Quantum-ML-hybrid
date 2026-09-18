import numpy as np
import pandas as pd

from aquire_preprocessing.feature_reextraction import select_patient_stratified_sample
from aquire_preprocessing.measurement_calibration import run_measurement_calibration


def test_patient_stratified_sample_has_no_duplicate_patients():
    rows = []
    for fold in range(1, 9):
        for cohort in range(3):
            for repetition in range(5):
                patient = fold * 1000 + cohort * 100 + repetition
                rows.extend([
                    {
                        "array_index": len(rows), "ecg_id": len(rows) + 1,
                        "patient_id": patient, "strat_fold": fold,
                        "mi_label": int(cohort == 0),
                        "hard_negative": bool(cohort == 1), "eligibility": "PRIMARY",
                    },
                    {
                        "array_index": len(rows) + 1, "ecg_id": len(rows) + 2,
                        "patient_id": patient, "strat_fold": fold,
                        "mi_label": int(cohort == 0),
                        "hard_negative": bool(cohort == 1), "eligibility": "PRIMARY",
                    },
                ])
    selected = select_patient_stratified_sample(pd.DataFrame(rows), 72, seed=3)
    assert len(selected) == 72
    assert selected.patient_id.is_unique
    assert set(selected.strat_fold) == set(range(1, 9))
    cohorts = np.where(selected.mi_label.eq(1), "MI", np.where(selected.hard_negative, "HARD", "OTHER"))
    assert set(cohorts) == {"MI", "HARD", "OTHER"}


def test_measurement_calibration_passes_exact_measurements(tmp_path):
    records = 80
    rng = np.random.default_rng(8)
    local = pd.DataFrame({
        "ecg_id": np.arange(records),
        "patient_id": np.arange(1000, 1000 + records),
        "strat_fold": np.tile(np.arange(1, 9), 10),
        "rr_median_ms": rng.normal(800, 70, records),
        "i__r_amp_mv": rng.normal(0.8, 0.2, records),
    })
    reference = pd.DataFrame({
        "ecg_id": local.ecg_id,
        "RR_Mean_Global": local.rr_median_ms + rng.normal(0, 2, records),
        "R_Amp_I": local.i__r_amp_mv + rng.normal(0, 0.01, records),
    })
    local_path, reference_path = tmp_path / "local.csv", tmp_path / "reference.csv"
    local.to_csv(local_path, index=False)
    reference.to_csv(reference_path, index=False)
    result = run_measurement_calibration(
        local_path,
        reference_path,
        {"rr_median_ms": "ref_ecgdeli__RR_Mean_Global", "i__r_amp_mv": "ref_ecgdeli__R_Amp_I"},
        {
            "rr_median_ms": {"group": "timing", "minimum_coverage": 0.9, "minimum_correlation": 0.9, "maximum_median_absolute_error": 10},
            "i__r_amp_mv": {"group": "amplitude", "minimum_coverage": 0.9, "minimum_correlation": 0.9, "maximum_median_absolute_error": 0.05},
        },
        tmp_path / "calibration",
    )
    # The production gate requires six amplitude leads, so this deliberately
    # small fixture stops even though both individual measurements pass.
    assert result["timing_pairs_passed"] == 1
    assert result["amplitude_pairs_passed"] == 1
    metrics = pd.read_csv(tmp_path / "calibration" / "measurement_calibration_metrics.csv")
    assert metrics.point_estimate_pass.all()
