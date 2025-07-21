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

def runAll(df):
    results = {}
    results['zscore'] = zscoreOutliers(df)
    results['dbscan'] = dbscanAdaptiveOutliers(df)
    results['isoforest'] = isoforestOutliers(df)
    results['lof'] = lofOutliers(df)
    results['svm'] = svmOutliers(df)
    results['elliptic'] = ellipticOutliers(df)
    results['knn'] = knnOutliers(df)
    results['mcd'] = mcdOutliers(df)
    #results['abod'] = abodOutliers(df)
    results['hbos'] = hbosOutliers(df)
    return results

def aggregate(results):
    agreement = defaultdict(int)

    for method, mask in results.items():
        for idx, is_outlier in enumerate(mask):
            if is_outlier:
                agreement[idx] += 1

    return agreement