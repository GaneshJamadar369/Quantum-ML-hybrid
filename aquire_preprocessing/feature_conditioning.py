import numpy as np
import pandas as pd
from typing import Dict

class FoldLocalOutlierClipper:
    """Winsorises each feature to [P1, P99] computed on the training fold."""
    def __init__(self, p_lower=1.0, p_upper=99.0):
        self.p_lower = p_lower
        self.p_upper = p_upper
        self.bounds_ = {}
        self.clip_rates_ = {}
        self.columns_ = []
        
    def fit(self, X_train: pd.DataFrame):
        self.columns_ = X_train.columns.tolist()
        self.bounds_ = {}
        self.clip_rates_ = {}
        
        for col in self.columns_:
            values = X_train[col].dropna().values
            if len(values) == 0:
                self.bounds_[col] = (np.nan, np.nan)
                self.clip_rates_[col] = 0.0
                continue
                
            lower = np.percentile(values, self.p_lower)
            upper = np.percentile(values, self.p_upper)
            self.bounds_[col] = (lower, upper)
            
            # calculate clip rate
            clipped = np.sum((values < lower) | (values > upper))
            self.clip_rates_[col] = clipped / len(values)
            
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col in self.columns_:
            if col in self.bounds_ and not np.isnan(self.bounds_[col][0]):
                lower, upper = self.bounds_[col]
                X_out[col] = np.clip(X_out[col], lower, upper)
        return X_out
        
    def audit_report(self) -> Dict[str, float]:
        return self.clip_rates_
