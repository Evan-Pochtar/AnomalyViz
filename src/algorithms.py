from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import zscore
import numpy as np

def zscoreOutliers(df, threshold=3):
    return (np.abs(zscore(df)) > threshold).any(axis=1)

def dbscanAdaptiveOutliers(df):
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

def isoforestOutliers(df):
    return IsolationForest(contamination='auto', random_state=42).fit_predict(df) == -1

def lofOutliers(df):
    return LocalOutlierFactor(n_neighbors=20).fit_predict(df) == -1

def svmOutliers(df):
    return OneClassSVM(gamma='auto', nu=0.05).fit_predict(df) == -1

def ellipticOutliers(df):
    return EllipticEnvelope(random_state=42).fit_predict(df) == -1

def knnOutliers(df):
    distances = NearestNeighbors(n_neighbors=5).fit(df).kneighbors(df)[0][:, -1]
    return distances > np.percentile(distances, 95)

def mcdOutliers(df):
    mahal = MinCovDet().fit(df).mahalanobis(df)
    return mahal > np.percentile(mahal, 97.5)

def abodOutliers(df):
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


def hbosOutliers(df):
    n_bins = 10
    scores = np.ones(len(df))

    for col in df.columns:
        hist, bins = np.histogram(df[col], bins=n_bins, density=True)
        idx = np.clip(np.digitize(df[col], bins[:-1], right=True), 0, n_bins - 1)
        prob = np.clip(hist[idx], 1e-6, None)
        scores *= 1 / prob

    return scores > np.percentile(scores, 95)
