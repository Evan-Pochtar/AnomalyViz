import numpy as np
import pandas as pd

def createSampleDataset(n_samples: int = 1000, n_features: int = 10, contamination: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    n_normal = int(n_samples * (1 - contamination))
    n_anomalies = n_samples - n_normal
    
    normal_data = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    anomaly_data = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,  # Shifted mean
        cov=np.eye(n_features) * 2,    # Larger variance
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
