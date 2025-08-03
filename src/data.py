import numpy as np
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None")
    
    if len(df) == 0:
        raise ValueError("DataFrame has no rows")
        
    if len(df.columns) == 0:
        raise ValueError("DataFrame has no columns")
    
    df = df.select_dtypes(include=[np.number]).dropna()
    if df.empty:
        raise ValueError("No numeric columns found in the dataset for outlier detection.")
    
    return df
