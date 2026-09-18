import numpy as np
import pandas as pd
from typing import Dict, List

PRESPECIFIED_INTERACTIONS = [
    ("iii__st60_mv", "avl__st60_mv", "mul", "interaction__st_reciprocity_iii_avl"),
    ("avf__r_amp_mv", "v5__r_amp_mv", "mul", "interaction__inferior_lateral_voltage"),
    ("clinical__inferior__st_mean_mv", "clinical__inferior__r_mean_mv", "ratio", "interaction__inferior_st_r_ratio"),
    ("rr_median_ms", "ii__r_amp_mv", "mul", "interaction__rate_amplitude"),
    ("v4__r_amp_mv", "v1__r_amp_mv", "diff", "interaction__r_progression_v1_v4"),
]

class ClinicalInteractionEngineer:
    """Creates prespecified pairwise interactions only — no combinatorial explosion."""
    def __init__(self):
        self.applied_interactions_ = []
        self.correlations_ = {}

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray):
        self.applied_interactions_ = []
        self.correlations_ = {}
        
        # We compute the features and their correlation with the label
        X_inter = self.transform(X_train)
        
        for _, _, _, name in PRESPECIFIED_INTERACTIONS:
            if name in X_inter.columns:
                self.applied_interactions_.append(name)
                # Compute pearson correlation with target (handling NaNs and 0 std)
                val = X_inter[name].values
                mask = ~np.isnan(val)
                if mask.sum() > 1 and np.std(val[mask]) > 1e-8:
                    corr = np.corrcoef(val[mask], y_train[mask])[0, 1]
                    self.correlations_[name] = float(corr)
                else:
                    self.correlations_[name] = 0.0
                    
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        
        for col1, col2, op, name in PRESPECIFIED_INTERACTIONS:
            if col1 in X.columns and col2 in X.columns:
                if op == "mul":
                    X_out[name] = X[col1] * X[col2]
                elif op == "diff":
                    X_out[name] = X[col1] - X[col2]
                elif op == "ratio":
                    # add small epsilon to denominator to prevent division by zero
                    X_out[name] = X[col1] / (X[col2] + 1e-6)
                    
        return X_out

    def audit_report(self) -> Dict[str, float]:
        return self.correlations_
