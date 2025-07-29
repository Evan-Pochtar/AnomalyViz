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
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import zscore
import numpy as np

def zscoreOutliers(df: pd.DataFrame, threshold: int = 3) -> np.ndarray[bool]:
    return (np.abs(zscore(df)) > threshold).any(axis=1)

def dbscanAdaptiveOutliers(df: pd.DataFrame) -> np.ndarray[bool]:
    scaled = StandardScaler().fit_transform(df)
    neigh = NearestNeighbors(n_neighbors=5).fit(scaled)

    distances = np.sort(neigh.kneighbors(scaled)[0][:, -1])
    for pct in [90, 95]:
        eps = np.percentile(distances, pct)
        labels = DBSCAN(eps=eps, min_samples=5).fit(scaled).labels_
        outliers = labels == -1
        if np.mean(outliers) <= 0.8:
            return outliers
        
    return outliers

def isoforestOutliers(df: pd.DataFrame) -> np.ndarray[bool]:
    return IsolationForest(contamination='auto').fit_predict(df) == -1

def lofOutliers(df: pd.DataFrame) -> np.ndarray[bool]:
    return LocalOutlierFactor(n_neighbors=20).fit_predict(df) == -1

def svmOutliers(df: pd.DataFrame) -> np.ndarray[bool]:
    scaler = StandardScaler()
    dataScaled = scaler.fit_transform(df)
   
    paramGrid = {
        'kernel': ['rbf', 'poly', 'sigmoid', 'linear'],
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
        if outlier_ratio < 0.01 or outlier_ratio > 0.5:
            continue
        
        label = (predictions == -1).astype(int)
        silScore = silhouette_score(dataScaled, label)
        separation = np.mean(decisionScores[predictions == 1]) - np.mean(decisionScores[predictions == -1])
        
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
        
        sil_weight, sep_weight, comp_weight, iso_weight = 1.0, 0.5, 0.3, 0.2
        total_score = (sil_weight * silScore + 
                      sep_weight * (separation / np.std(decisionScores)) +
                      comp_weight * (compactness / np.std(inlier_distances) if len(inliers) > 1 else 0) +
                      iso_weight * (isolation / np.std(dataScaled.flatten())))
        
        if total_score > bestScore:
            bestScore = total_score
            bestPredictions = predictions
    
    return bestPredictions == -1

def ellipticOutliers(df: pd.DataFrame) -> np.ndarray[bool]:
    return EllipticEnvelope(contamination=0.05).fit_predict(df) == -1

def knnOutliers(df: pd.DataFrame) -> np.ndarray[bool]:
    distances = NearestNeighbors(n_neighbors=5).fit(df).kneighbors(df)[0][:, -1]
    return distances > np.percentile(distances, 95)

def mcdOutliers(df: pd.DataFrame) -> np.ndarray[bool]:
    mahal = MinCovDet().fit(df).mahalanobis(df)
    return mahal > np.percentile(mahal, 95)

def abodOutliers(df: pd.DataFrame) -> np.ndarray[bool]:
    X = df.values
    n = len(X)
    outlier_scores = np.zeros(n)
    dists = pairwise_distances(X)
    for i in range(n):
        angles = []
        for j in range(n):
            for k in range(j + 1, n):
                if i != j and i != k:
                    v1 = X[j] - X[i]
                    v2 = X[k] - X[i]
                    norm_v1 = np.linalg.norm(v1)
                    norm_v2 = np.linalg.norm(v2)
                    if norm_v1 > 1e-10 and norm_v2 > 1e-10:
                        cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
                        cos_angle = np.clip(cos_angle, -1, 1)
                        angle = np.arccos(cos_angle)
                        angles.append(angle)
        outlier_scores[i] = np.var(angles) if angles else 0
    threshold = np.percentile(outlier_scores, 5)
    return outlier_scores <= threshold


def hbosOutliers(df: pd.DataFrame) -> np.ndarray[bool]:
    n_bins = 10
    scores = np.ones(len(df))

    for col in df.columns:
        hist, bins = np.histogram(df[col], bins=n_bins, density=True)
        idx = np.clip(np.digitize(df[col], bins[:-1], right=True), 0, n_bins - 1)
        prob = np.clip(hist[idx], 1e-6, None)
        scores *= 1 / prob

    return scores > np.percentile(scores, 95)
