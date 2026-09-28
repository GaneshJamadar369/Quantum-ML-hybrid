"""Versioned scientific content shown by the prototype UI."""

from __future__ import annotations


MODEL_CARD = {
    "name": "AQUIRE-Med fixed parallel hybrid prototype",
    "task": "MI-pattern versus non-MI-pattern classification from one 10-second 12-lead ECG",
    "intended_use": "Research screening and Smart India Hackathon demonstration",
    "not_intended_for": [
        "acute MI diagnosis",
        "future cardiovascular event prediction",
        "treatment selection",
        "autonomous clinical decisions",
    ],
    "dataset": {
        "primary": "PTB-XL v1.0.3",
        "reference": "PTB-XL+ v1.0.1",
        "development_records": 17348,
        "development_patients": 14958,
        "development_folds": [1, 2, 3, 4, 5, 6, 7, 8],
    },
    "claim_boundary": "Quantum advantage is not established; the matched overall-classical fusion is slightly stronger in AUPRC.",
}

ARCHITECTURE = {
    "routing": "fixed_parallel_all_eligible_inputs",
    "common": ["upload", "structural_validation", "quality_control", "minimal_signal"],
    "routes": {
        "quantum": {
            "active": True,
            "steps": ["transformer_h128", "robust_scaling", "pls_q4", "angle_map", "four_qubit_vqc"],
            "output": "s_q",
        },
        "classical": {
            "active": True,
            "steps": ["106_morphology_features", "imputation", "hist_gradient_boosting"],
            "output": "s_c",
        },
        "fusion": {
            "active": True,
            "formula": "sigmoid(beta_0 + beta_q * s_q + beta_c * s_c)",
            "output": "mi_pattern_probability",
        },
    },
    "prohibited": ["dynamic_routing", "predictor_selection_gate", "residual_target_learning"],
}

BENCHMARKS = {
    "scope": "PTB-XL development OOF, folds 1-8",
    "primary_metric": "AUPRC",
    "stabilized_two_seed": [
        {"name": "Quantum only", "auprc": 0.8298004791, "auroc": 0.92202, "brier": 0.09247, "sensitivity_at_90_specificity": 0.76763},
        {"name": "Overall classical", "auprc": 0.8365327466, "auroc": 0.92736, "brier": 0.08940, "sensitivity_at_90_specificity": 0.77656},
        {"name": "Hybrid fusion", "auprc": 0.8359958579, "auroc": 0.92659, "brier": 0.08961, "sensitivity_at_90_specificity": 0.77793},
    ],
    "five_seed_auprc": {
        "quantum_only": 0.8275012717,
        "overall_classical": 0.8376586666,
        "hybrid_fusion": 0.8360706890,
    },
    "operating_point": {
        "target_specificity": 0.90,
        "overall_classical": {"true_positives": 3392, "false_negatives": 976, "false_positives": 1298, "true_negatives": 11682},
        "hybrid_fusion": {"true_positives": 3398, "false_negatives": 970, "false_positives": 1298, "true_negatives": 11682},
    },
    "threshold_selection": {
        "frozen_threshold": 0.43531340285804376,
        "calibration_fold": 9,
        "target_specificity": 0.90,
        "observed_specificity": 0.9004276114844227,
        "observed_sensitivity": 0.7392923649906891,
        "reasoning": (
            "The decision threshold was selected once on patient-separated fold 9 to target "
            "approximately 90% specificity, limiting false-positive alerts while preserving "
            "sensitivity. It is frozen for all submitted ECGs and is not adjusted per patient."
        ),
    },
    "confusion_matrix": {
        "scope": "PTB-XL development OOF folds 1-8 at the matched approximately 90% specificity operating point",
        "true_positives": {"count": 3398, "rate": 0.7779304029304029, "denominator": "all MI-pattern ECGs"},
        "false_negatives": {"count": 970, "rate": 0.22206959706959706, "denominator": "all MI-pattern ECGs"},
        "false_positives": {"count": 1298, "rate": 0.10, "denominator": "all non-MI-pattern ECGs"},
        "true_negatives": {"count": 11682, "rate": 0.90, "denominator": "all non-MI-pattern ECGs"},
        "positive_predictive_value": 0.7235945485519591,
        "negative_predictive_value": 0.9233322794815049,
        "accuracy": 0.8692644685266313,
    },
    "interpretation": "Hybrid fusion recovered six additional MI-pattern records at the matched development operating point, while the overall classical replacement remained slightly stronger in global AUPRC.",
}
