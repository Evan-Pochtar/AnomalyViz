import numpy as np
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.select_dtypes(include=[np.number]).dropna()
    if df.empty:
        raise ValueError("No numeric columns found in the dataset for outlier detection.")
    return df