import numpy as np
import pandas as pd

def createSampleDataset(n_samples: int = 1000, n_features: int = 10, contamination: float = 0.1, random_state: int = 1) -> pd.DataFrame:
    """
    Generate a synthetic dataset with known outliers for testing outlier detection algorithms.
    
    Creates a realistic multi-dimensional dataset with both normal data points and
    deliberately introduced anomalies. This is essential for validating outlier
    detection algorithms since we know the ground truth labels.
    
    Args:
        n_samples (int): Total number of data points to generate (default: 1000)
        n_features (int): Number of numeric features in the dataset (default: 10)
        contamination (float): Proportion of anomalies (0.0 to 1.0, default: 0.1)
        random_state (int): Random seed for reproducible results (default: 1)
        
    Returns:
        pd.DataFrame: Generated dataset containing:
            - feature_0 to feature_{n_features-1}: Numeric features
            - is_anomaly: Ground truth labels (0=normal, 1=anomaly)
            - category_A: Categorical feature with 3 levels
            - category_B: Categorical feature with 2 levels  
            - integer_feature: Random integer feature (1-99)
    """

    np.random.seed(random_state)
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal
    
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,
        cov=np.eye(n_features) * 2,
        size=n_anomalies
    )
    
    X = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['is_anomaly'] = labels
    
    df['category_A'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=n_samples)
    df['category_B'] = np.random.choice(['Group_X', 'Group_Y'], size=n_samples)
    
    df['integer_feature'] = np.random.randint(1, 100, size=n_samples)
    
    return df
