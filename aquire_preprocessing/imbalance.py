import numpy as np

def fold_safe_smote_enn(X_train, y_train, random_state=42):
    """SMOTE + Edited Nearest Neighbours cleaning. Never touches validation fold."""
    try:
        from imblearn.combine import SMOTEENN
    except ImportError as exc:
        raise RuntimeError("imbalanced-learn is required for SMOTE-ENN; install requirements.lock") from exc
        
    smote_enn = SMOTEENN(random_state=random_state, n_jobs=-1)
    return smote_enn.fit_resample(X_train, y_train)

def hard_negative_weights(y_train, hard_negative_train):
    """Upweight abnormal-ECG hard-negative non-MI records."""
    w = np.ones(len(y_train), dtype=float)
    w[(y_train == 0) & hard_negative_train] = 1.5
    return w
