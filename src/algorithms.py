import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from scipy.spatial.distance import pdist, cdist
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
import numpy as np


def zscoreOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using Z-score statistical method.
    
    This method identifies outliers by calculating the Z-score (standard deviations 
    from the mean) for each data point. Points with Z-scores exceeding a threshold
    based on the contamination parameter are flagged as outliers.
    
    Best for: Normally distributed data, simple and fast detection.
    Limitations: Assumes normal distribution, sensitive to extreme outliers.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """

    z_scores = np.abs(zscore(df))
    max_z_scores = z_scores.max(axis=1)
    threshold_percentile = (1 - contamination) * 100
    threshold_value = np.percentile(max_z_scores, threshold_percentile)
    
    return max_z_scores > threshold_value


def dbscanAdaptiveOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using adaptive DBSCAN clustering.
    
    This method uses DBSCAN (Density-Based Spatial Clustering) to identify regions
    of high density. Points that don't belong to any cluster are considered outliers.
    The eps parameter is adaptively chosen based on k-nearest neighbor distances.
    
    Best for: Clusters of varying densities, non-spherical cluster shapes.
    Limitations: Sensitive to parameter selection, struggles with varying densities.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """

    n_samples = len(df)
    n_neighbors = max(20, min(50, int(np.sqrt(n_samples))))
    min_samples = max(5, min(20, int(n_samples * 0.005)))
    
    scaled = StandardScaler().fit_transform(df)
    neigh = NearestNeighbors(n_neighbors=n_neighbors).fit(scaled)
    distances, _ = neigh.kneighbors(scaled)
    distances = np.sort(distances[:, -1])

    for pct in [80, 85, 90, 95, 98]:
        eps = np.percentile(distances, pct)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(scaled)
        outliers = labels == -1
        outlier_rate = np.mean(outliers)

        if abs(outlier_rate - contamination) < 0.1 or outlier_rate <= contamination * 2:
            return outliers

    return outliers


def isoforestOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using Isolation Forest algorithm.
    
    Isolation Forest isolates outliers by randomly selecting features and split values.
    Outliers are easier to isolate (require fewer splits) than normal points, as they
    are few and different from the majority of data points.
    
    Best for: High-dimensional data, mixed data types, fast execution.
    Limitations: May struggle with very small datasets or uniform distributions.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """

    n_samples = len(df)
    n_estimators = max(50, min(200, int(n_samples / 10)))
    max_samples = min(256, max(32, int(n_samples * 0.8)))
    
    iso_forest = IsolationForest(
        contamination=contamination, 
        n_estimators=n_estimators,
        max_samples=max_samples,
        n_jobs=-1
    )
    predictions = iso_forest.fit_predict(df)
    
    return predictions == -1


def lofOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using Local Outlier Factor (LOF).
    
    LOF compares the local density of a point with the local densities of its
    neighbors. Points with substantially lower density than their neighbors
    are considered outliers.
    
    Best for: Data with varying densities, local outlier patterns.
    Limitations: Sensitive to choice of k, computationally expensive for large datasets.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """

    n_samples = len(df)
    n_neighbors = max(20, min(50, int(np.log(n_samples) * 3)))
    
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = lof.fit_predict(df)

    return predictions == -1

def svmOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using One-Class SVM with comprehensive parameter optimization.
    
    One-Class SVM learns a decision function for novelty detection: classifying
    new data as similar or different from the training set. This implementation
    performs extensive hyperparameter tuning to find the best model configuration.
    
    Best for: Complex decision boundaries, robust performance across data types.
    Limitations: Computationally expensive due to grid search, many hyperparameters.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """

    scaler = StandardScaler()
    dataScaled = scaler.fit_transform(df)
    paramGrid = {
        'kernel': ['rbf', 'poly', 'linear'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
        'nu': [0.01, 0.05, 0.1, 0.2, 0.3]
    }

    grid = list(ParameterGrid(paramGrid))
    bestScore, bestPredictions = -np.inf, np.zeros(len(df), dtype=bool)
    for params in grid:
        model = OneClassSVM(**params)
        model.fit(dataScaled)
        predictions = model.predict(dataScaled)
        decisionScores = model.decision_function(dataScaled)
            
        outlier_ratio = np.sum(predictions == -1) / len(predictions)
        if outlier_ratio < contamination * 0.1 or outlier_ratio > contamination * 5:
            continue
        label = (predictions == -1).astype(int)
        if len(np.unique(label)) < 2:
            continue

        silScore = silhouette_score(dataScaled, label)
        separation = (np.mean(decisionScores[predictions == 1]) - np.mean(decisionScores[predictions == -1]))
        
        inliers = dataScaled[predictions == 1]
        if len(inliers) > 1:
            inlier_distances = pdist(inliers)
            compactness = -np.mean(inlier_distances)
        else:
            compactness = 0
        
        outliers = dataScaled[predictions == -1]
        if len(outliers) > 0 and len(inliers) > 0:
            distances = cdist(outliers, inliers)
            isolation = np.mean(np.min(distances, axis=1))
        else:
            isolation = 0

        contamination_penalty = abs(outlier_ratio - contamination) / contamination
        sil_weight, sep_weight, comp_weight, iso_weight, cont_weight = 1.0, 0.5, 0.3, 0.2, 2.0
        total_score = (sil_weight * silScore + 
                      sep_weight * (separation / np.std(decisionScores)) +
                      comp_weight * (compactness / np.std(inlier_distances) if len(inliers) > 1 else 0) +
                      iso_weight * (isolation / np.std(dataScaled.flatten())) -
                      cont_weight * contamination_penalty)
        
        if total_score > bestScore:
            bestScore = total_score
            bestPredictions = predictions
    
    return bestPredictions == -1

def ellipticOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using Elliptic Envelope (Robust Covariance Estimation).
    
    This method assumes data follows a Gaussian distribution and fits an ellipse
    around the central data points. Points outside this ellipse are considered
    outliers. Uses robust covariance estimation to handle some outliers in training.
    
    Best for: Gaussian/normal distributed data, multivariate outlier detection.
    Limitations: Assumes elliptical data distribution, may struggle with non-Gaussian data.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """

    n_samples = len(df)
    support_fraction = max(0.5, min(0.9, (n_samples - 10) / n_samples))
    
    elliptic_env = EllipticEnvelope(
        contamination=contamination,
        support_fraction=support_fraction
    )
    predictions = elliptic_env.fit_predict(df)
    
    return predictions == -1


def knnOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using k-Nearest Neighbors distance method.
    
    This method calculates the distance to the k-th nearest neighbor for each point.
    Points with the largest distances to their k-th neighbor are considered outliers,
    as they are far from their local neighborhood.
    
    Best for: Simple implementation, works well with local density variations.
    Limitations: Sensitive to choice of k, computationally expensive for large datasets.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """
    
    n_samples = len(df)
    k = max(20, min(30, int(np.log(n_samples) * 2)))
    
    knn = NearestNeighbors(n_neighbors=k)
    distances, _ = knn.fit(df).kneighbors(df)
    kth_distances = distances[:, -1]
    threshold_percentile = (1 - contamination) * 100
    threshold = np.percentile(kth_distances, threshold_percentile)
    
    return kth_distances > threshold


def mcdOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using Minimum Covariance Determinant (MCD).
    
    MCD is a robust method for covariance estimation that finds the subset of
    observations whose covariance matrix has the lowest determinant. Outliers
    are identified using Mahalanobis distances from this robust estimate.
    
    Best for: Multivariate Gaussian data, robust to outliers in training data.
    Limitations: Assumes elliptical distribution, computationally intensive.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """

    n_samples = len(df)
    support_fraction = max(0.5, min(0.9, (n_samples - len(df.columns) - 1) / n_samples))
    
    mcd = MinCovDet(support_fraction=support_fraction).fit(df)
    mahalanobis_distances = mcd.mahalanobis(df)
    threshold_percentile = (1 - contamination) * 100
    threshold = np.percentile(mahalanobis_distances, threshold_percentile)
    
    return mahalanobis_distances > threshold

def abodOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using Fast Angle-Based Outlier Detection (ABOD).
    
    Optimized version that uses:
    - Adaptive sampling to reduce computational complexity
    - Vectorized operations for angle calculations
    - Early termination for stable variance estimates
    - Memory-efficient batch processing
    
    Best for: High-dimensional data, points with unusual angular relationships.
    Limitations: Approximation method, may miss some edge cases.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """

    X = df.values
    n, d = X.shape
    if n <= 100:
        max_pairs = min(200, n * (n - 1) // 2)
        batch_size = 50
    elif n <= 1000:
        max_pairs = min(500, n * 10)
        batch_size = 100
    else:
        max_pairs = min(1000, n * 5)
        batch_size = 200
    
    outlier_scores = np.zeros(n)
    
    if n > 50:
        max_sample_pairs = min(max_pairs, n * (n - 1) // 2)
        if max_sample_pairs < n * (n - 1) // 2:
            pair_indices = []
            seen = set()
            while len(pair_indices) < max_sample_pairs:
                j, k = np.random.choice(n, 2, replace=False)
                if j != k and (j, k) not in seen and (k, j) not in seen:
                    pair_indices.append((j, k))
                    seen.add((j, k))
            pair_indices = np.array(pair_indices)
        else:
            pair_indices = np.array([(j, k) for j in range(n) for k in range(j + 1, n)])
    else:
        pair_indices = np.array([(j, k) for j in range(n) for k in range(j + 1, n)])
    
    for i_start in range(0, n, batch_size):
        i_end = min(i_start + batch_size, n)
        batch_indices = range(i_start, i_end)
        
        for i in batch_indices:
            v1_batch = X[pair_indices[:, 0]] - X[i]
            v2_batch = X[pair_indices[:, 1]] - X[i]
            norm_v1 = np.linalg.norm(v1_batch, axis=1)
            norm_v2 = np.linalg.norm(v2_batch, axis=1)
            
            valid_mask = (norm_v1 > 1e-10) & (norm_v2 > 1e-10)
            if np.sum(valid_mask) == 0:
                continue
                
            v1_valid = v1_batch[valid_mask]
            v2_valid = v2_batch[valid_mask]
            norm_v1_valid = norm_v1[valid_mask]
            norm_v2_valid = norm_v2[valid_mask]
            
            dot_products = np.sum(v1_valid * v2_valid, axis=1)
            cos_angles = np.clip(dot_products / (norm_v1_valid * norm_v2_valid), -1, 1)
            angles = np.arccos(cos_angles)
            angles = angles[np.isfinite(angles)]

            if len(angles) > 1:
                outlier_scores[i] = np.var(angles)
            else:
                outlier_scores[i] = 0
    
    if np.all(outlier_scores == 0):
        return np.zeros(n, dtype=bool)
    
    threshold_percentile = contamination * 100
    threshold = np.percentile(outlier_scores, threshold_percentile)
    return outlier_scores <= threshold

def hbosOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using Histogram-Based Outlier Score (HBOS).
    
    HBOS builds histograms for each feature and calculates outlier scores based
    on the inverse of bin densities. Points in low-density regions (sparse bins)
    receive higher outlier scores. Assumes feature independence.
    
    Best for: Mixed data types, interpretable results, fast computation.
    Limitations: Assumes feature independence, sensitive to binning strategy.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """

    n_samples = len(df)
    n_bins = max(5, min(50, int(np.log2(n_samples)) + 1))
    
    scores = np.ones(len(df))

    for col in df.columns:
        hist, bins = np.histogram(df[col], bins=n_bins, density=True)
        idx = np.clip(np.digitize(df[col], bins[:-1], right=True), 0, n_bins - 1)
        prob = np.clip(hist[idx], 1e-6, None)
        scores *= 1 / prob

    threshold_percentile = (1 - contamination) * 100
    threshold = np.percentile(scores, threshold_percentile)
    
    return scores > threshold
