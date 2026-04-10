"""
preprocessing_utils.py
-----------------------
Shared preprocessing functions for Lead Scoring v2 model pipeline.

These functions are used inside sklearn FunctionTransformer steps.
They MUST live in a standalone module (not __main__) so that joblib
can serialize and deserialize the pipeline from ANY entry point
(train script, compare script, API server, etc.).
"""

import numpy as np


def log_transform_days(X):
    """
    Apply log1p to Days_Since_Created (column index 1) while keeping
    Call_Attempt_Count (column index 0) raw.

    Used inside sklearn.preprocessing.FunctionTransformer within the
    numeric preprocessing pipeline.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, 2)
        Column 0 = Call_Attempt_Count (kept as-is)
        Column 1 = Days_Since_Created  (log1p transformed)

    Returns
    -------
    np.ndarray, shape (n_samples, 2)
    """
    return np.column_stack([
        X[:, 0],            # Call_Attempt_Count — keep raw
        np.log1p(X[:, 1]),   # Days_Since_Created — log transform (right-skewed)
    ])
