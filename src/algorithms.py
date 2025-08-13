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
from sklearn.metrics import pairwise_distances
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
    data_array = df.values
    n_samples, n_features = data_array.shape

    if n_samples < 2:
        return np.zeros(n_samples, dtype=bool)
    
    z_scores_per_feature = np.zeros_like(data_array)
    for i in range(n_features):
        feature_data = data_array[:, i]
        feature_mean = np.mean(feature_data)
        feature_std = np.std(feature_data, ddof=0)
    
        if feature_std < 1e-10:
            z_scores_per_feature[:, i] = 0.0
        else:
            relative_deviations = np.abs(feature_data - feature_mean) / (np.abs(feature_mean) + 1e-10)
            max_relative_deviation = np.max(relative_deviations)
            
            if max_relative_deviation < 1e-12:
                z_scores_per_feature[:, i] = 0.0
            else:
                z_scores_per_feature[:, i] = np.abs((feature_data - feature_mean) / feature_std)
    
    max_z_scores = np.max(z_scores_per_feature, axis=1)
    if np.all(max_z_scores == 0):
        return np.zeros(n_samples, dtype=bool)
    
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
    max_samples = min(256, max(n_samples, int(n_samples * 0.8)))
    
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
    n_neighbors = max(min(n_samples, 20), min(50, int(np.log(n_samples) * 3)))
    
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
                      comp_weight * (compactness / (np.std(inlier_distances) +  1e-8) if len(inliers) > 1 else 0) +
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

    n_samples, n_features = df.shape
    if n_samples < 2:
        return np.zeros(n_samples, dtype=bool)
    
    data_array = df.values
    feature_variances = np.var(data_array, axis=0)
    if (feature_variances < 1e-10).any():
        valid_features = feature_variances >= 1e-10
        if valid_features.sum() == 0:
            return np.zeros(n_samples, dtype=bool)
        data_for_covariance = data_array[:, valid_features]
    else:
        data_for_covariance = data_array
    
    if data_for_covariance.shape[1] >= n_samples:
        distances = pairwise_distances(data_array)
        mean_distances = np.mean(distances, axis=1)
        threshold = np.percentile(mean_distances, (1 - contamination) * 100)
        return mean_distances > threshold
    
    try:
        sample_cov = np.cov(data_for_covariance.T)
        if data_for_covariance.shape[1] > 1:
            cond_number = np.linalg.cond(sample_cov)
            if cond_number > 1e12:
                distances = pairwise_distances(data_array)
                mean_distances = np.mean(distances, axis=1)
                threshold = np.percentile(mean_distances, (1 - contamination) * 100)
                return mean_distances > threshold
    except np.linalg.LinAlgError:
        distances = pairwise_distances(data_array)
        mean_distances = np.mean(distances, axis=1)
        threshold = np.percentile(mean_distances, (1 - contamination) * 100)
        return mean_distances > threshold
    
    support_fraction = max(0.5, min(0.9, (n_samples - n_features - 1) / n_samples))
    
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
    k = max(min(n_samples, 20), min(40, int(np.log(n_samples) * 2)))
    
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

    n_samples, n_features = df.shape

    if n_samples < 2:
        return np.zeros(n_samples, dtype=bool)
    
    if n_samples <= n_features:
        z_scores = np.abs(zscore(df, axis=0, nan_policy='omit'))
        max_z_scores = np.nanmax(z_scores, axis=1)
        threshold = np.percentile(max_z_scores, (1 - contamination) * 100)
        return max_z_scores > threshold
    
    data_array = df.values
    feature_variances = np.var(data_array, axis=0)
    
    if (feature_variances < 1e-10).any():
        valid_features = feature_variances >= 1e-10
        if valid_features.sum() == 0:
            return np.zeros(n_samples, dtype=bool)
        
        df_filtered = df.iloc[:, valid_features]
        n_features_filtered = df_filtered.shape[1]
        
        if n_samples <= n_features_filtered:
            z_scores = np.abs(zscore(df_filtered, axis=0, nan_policy='omit'))
            max_z_scores = np.nanmax(z_scores, axis=1)
            threshold = np.percentile(max_z_scores, (1 - contamination) * 100)
            return max_z_scores > threshold
    else:
        df_filtered = df
        n_features_filtered = n_features
    
    min_support_samples = n_features_filtered + 1
    max_support_samples = n_samples
    support_samples = max(min_support_samples, int(n_samples * 0.5))
    support_samples = min(support_samples, max_support_samples)
    support_fraction = support_samples / n_samples
    support_fraction = max(0.5, min(0.9, support_fraction))
    
    try:
        mcd = MinCovDet(support_fraction=support_fraction).fit(df_filtered)
        mahalanobis_distances = mcd.mahalanobis(df_filtered)
    except (np.linalg.LinAlgError, ValueError):
        distances = pairwise_distances(df.values)
        mean_distances = np.mean(distances, axis=1)
        threshold = np.percentile(mean_distances, (1 - contamination) * 100)
        return mean_distances > threshold
    
    threshold_percentile = (1 - contamination) * 100
    threshold = np.percentile(mahalanobis_distances, threshold_percentile)
    
    return mahalanobis_distances > threshold

def copodOutliers(df: pd.DataFrame, contamination: float) -> np.ndarray:
    """
    Detect outliers using COPOD (Copula-Based Outlier Detection).
    
    COPOD uses copula functions to model the dependence structure between features
    and detects outliers based on their empirical copula values. It's parameter-free,
    fast, and works well with high-dimensional data.
    
    Best for: High-dimensional data, mixed distributions, very fast execution.
    Limitations: May struggle with highly correlated features.
    
    Args:
        df (pd.DataFrame): Input dataset with numerical features
        contamination (float): Expected proportion of outliers (0.0 to 1.0)
        
    Returns:
        np.ndarray: Boolean array where True indicates an outlier
    """
    
    X = df.values
    n, d = X.shape
    if n == 0:
        return np.array([], dtype=bool)

    ranks = df.rank(method='average').values
    F = ranks / (n + 1.0)

    eps = 1.0 / (n + 1.0)
    left_log = -np.log(F + eps)
    right_log = -np.log(1.0 - F + eps)
    tail_score = np.maximum(left_log, right_log)

    raw_scores = tail_score.sum(axis=1)
    med = np.median(raw_scores)
    mad = np.median(np.abs(raw_scores - med))
    if mad == 0:
        mad = 1.0
    scores = (raw_scores - med) / mad

    threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
    return scores > threshold

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
