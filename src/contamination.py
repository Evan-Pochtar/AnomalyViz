import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def estimateOutlierContamination(df: pd.DataFrame) -> float:
    """
    Automatically estimate the contamination rate (outlier proportion) in a dataset.
    
    Uses Isolation Forest with 'auto' contamination to detect outliers and calculate
    the actual contamination rate. This helps determine appropriate contamination
    parameters for other outlier detection algorithms.

    Args:
        df (pd.DataFrame): Input dataset to analyze for outlier contamination
        
    Returns:
        float: Estimated contamination rate as a proportion (0.0 to 1.0)
               where 0.1 means approximately 10% outliers
    """

    n_samples = len(df)
    max_rows = 100000
    if n_samples > max_rows:
        df = df.sample(max_rows)
        n_samples = max_rows

    X = df.values

    clf = IsolationForest(n_estimators=100, contamination='auto', n_jobs=-1)
    preds = clf.fit_predict(X)
    outlier_flags = (preds == -1)

    outliers = np.count_nonzero(outlier_flags)
    contamination = outliers / float(n_samples)
    return contamination
