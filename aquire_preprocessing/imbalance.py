import warnings
import numpy as np


def fold_safe_smote_enn(X_train, y_train, random_state=42):
    """SMOTE + Edited Nearest Neighbours cleaning. Never touches validation fold.

    If *imbalanced-learn* is not installed the function logs a warning and
    returns the original arrays unchanged so the rest of the pipeline can still
    run (e.g. during local unit tests or on environments without the package).
    On Kaggle the runner installs imbalanced-learn before this is called.
    """
    try:
        from imblearn.combine import SMOTEENN
    except ImportError:
        warnings.warn(
            "imbalanced-learn is not installed; SMOTE-ENN skipped. "
            "Install it with: pip install imbalanced-learn",
            RuntimeWarning,
            stacklevel=2,
        )
        return X_train, y_train

    smote_enn = SMOTEENN(random_state=random_state, n_jobs=-1)
    return smote_enn.fit_resample(X_train, y_train)


def hard_negative_weights(y_train, hard_negative_train):
    """Upweight abnormal-ECG hard-negative non-MI records."""
    w = np.ones(len(y_train), dtype=float)
    w[(y_train == 0) & hard_negative_train] = 1.5
    return w

