import numpy as np
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")
    if df.shape[0] == 0:
        raise ValueError("DataFrame has no rows")
    if df.shape[1] == 0:
        raise ValueError("DataFrame has no columns")
    
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df[~numeric_df.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
    
    if numeric_df.empty:
        raise ValueError("No numeric columns with valid finite values found")
    
    return numeric_df
