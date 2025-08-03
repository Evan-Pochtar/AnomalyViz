import pandas as pd
from collections import defaultdict
from src.algorithms import (
    zscoreOutliers,
    dbscanAdaptiveOutliers,
    isoforestOutliers,
    lofOutliers,
    svmOutliers,
    ellipticOutliers,
    knnOutliers,
    mcdOutliers,
    abodOutliers,
    hbosOutliers
)

ALGORITHM_MAP = {
    'zscore': zscoreOutliers,
    'dbscan': dbscanAdaptiveOutliers,
    'isoforest': isoforestOutliers,
    'lof': lofOutliers,
    'svm': svmOutliers,
    'elliptic': ellipticOutliers,
    'knn': knnOutliers,
    'mcd': mcdOutliers,
    #'abod': abodOutliers,
    'hbos': hbosOutliers
}

def runAll(df: pd.DataFrame, algorithms: list[str] = None, contamination: float = 0.00) -> dict[str, pd.Series]:
    results = {}
    
    if not algorithms:
        algorithms = list(ALGORITHM_MAP.keys())
    
    invalid = [alg for alg in algorithms if alg not in ALGORITHM_MAP]
    if invalid:
        raise ValueError(f"Invalid algorithm(s): {invalid}. "
                        f"Available algorithms: {list(ALGORITHM_MAP.keys())}")
    
    for algorithm in algorithms:
        results[algorithm] = ALGORITHM_MAP[algorithm](df, contamination)

    return results

def aggregate(results) -> defaultdict[int, int]:
    agreement = defaultdict(int)
    
    for method, mask in results.items():
        for idx, outlier in enumerate(mask):
            if outlier:
                agreement[idx] += 1

    return agreement
