import numpy as np

from run_waveform_representation_export import patient_balanced_weights


def test_patient_balanced_weights_equalize_repeated_records_before_hard_negative_weighting():
    patients = np.asarray([1, 1, 1, 2, 3, 3])
    labels = np.asarray([1, 1, 1, 0, 1, 1])
    hard_negative = np.zeros(6, dtype=bool)
    weights = patient_balanced_weights(patients, labels, hard_negative)
    totals = [weights[patients == patient].sum() for patient in (1, 2, 3)]
    assert np.allclose(totals, totals[0])
    assert np.isclose(weights.mean(), 1.0)


def test_patient_balanced_weights_upweight_hard_negative_records():
    patients = np.asarray([1, 2, 3, 4])
    labels = np.asarray([1, 0, 0, 1])
    hard_negative = np.asarray([False, True, False, False])
    weights = patient_balanced_weights(patients, labels, hard_negative)
    assert weights[1] > weights[2]
    assert np.isclose(weights.mean(), 1.0)
