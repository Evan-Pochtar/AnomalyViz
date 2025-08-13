import numpy as np
import pandas as pd

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare dataset for outlier detection by extracting valid numeric data.
    
    Outlier detection algorithms require numeric data with finite values. This function
    performs essential preprocessing to ensure the dataset is suitable for analysis
    by filtering to numeric columns and removing invalid values.
    
    Data quality checks:
    - Rejects empty or None DataFrames
    - Ensures at least one row and column exist
    - Verifies numeric data is available after filtering
    - Removes problematic values that break algorithms
    
    Args:
        df (pd.DataFrame): Input DataFrame to clean and prepare
        
    Returns:
        pd.DataFrame: Cleaned DataFrame containing only:
            - Numeric columns (integer and float types)
            - Rows with finite values only (no NaN, inf, -inf)
            - At least one row and column of valid data
            
    Raises:
        ValueError: If input DataFrame is empty, None, has no rows/columns,
          or contains no valid numeric data after cleaning
    """

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