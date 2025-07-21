import numpy as np

def clean(df):
    df = df.select_dtypes(include=[np.number]).dropna()
    if df.empty:
        raise ValueError("No numeric columns found in the dataset for outlier detection.")
    return df