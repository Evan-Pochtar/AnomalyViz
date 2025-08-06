import pandas as pd
import numpy as np
import time
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

def runAll(df: pd.DataFrame, algorithms: list[str] = None, contamination: float = 0.00) -> dict:
    if not algorithms:
        algorithms = list(ALGORITHM_MAP.keys())
    
    # Validate algorithms
    invalid = [alg for alg in algorithms if alg not in ALGORITHM_MAP]
    if invalid:
        raise ValueError(f"Invalid algorithm(s): {invalid}. "
                        f"Available algorithms: {list(ALGORITHM_MAP.keys())}")
    
    # Validate contamination
    if contamination < 0 or contamination > 0.5:
        raise ValueError("Contamination must be between 0.0 and 0.5")

    results, timings, statistics, failed = {}, {}, {}, []
    totalStartTime = time.time()
    
    for i, algorithm in enumerate(algorithms, 1):
        try:
            startTime = time.time()
            outlier_mask = ALGORITHM_MAP[algorithm](df, contamination)
            endTime = time.time()
            
            execution_time = endTime - startTime
            timings[algorithm] = execution_time
            results[algorithm] = pd.Series(outlier_mask, index=df.index, name=f'{algorithm}_outliers')
            
            n_outliers = np.sum(outlier_mask)
            actual_contamination = n_outliers / len(df)
            
            statistics[algorithm] = {
                'n_outliers': int(n_outliers),
                'n_inliers': int(len(df) - n_outliers),
                'actual_contamination': actual_contamination,
                'expected_contamination': contamination,
                'contamination_difference': actual_contamination - contamination,
                'execution_time': execution_time
            }
            
        except Exception as e:
            failed.append(algorithm)
            timings[algorithm] = None
            statistics[algorithm] = {
                'error': str(e),
                'execution_time': None
            }
    
    total_endTime = time.time()
    total_execution_time = total_endTime - totalStartTime
    
    successful_algorithms = [alg for alg in algorithms if alg not in failed]
    if successful_algorithms:
        all_timings = [timings[alg] for alg in successful_algorithms if timings[alg] is not None]
        timing_stats = {
            'total_time': total_execution_time,
            'average_time': np.mean(all_timings) if all_timings else 0,
            'fastest_algorithm': min(successful_algorithms, key=lambda x: timings[x]) if all_timings else None,
            'slowest_algorithm': max(successful_algorithms, key=lambda x: timings[x]) if all_timings else None,
            'fastest_time': min(all_timings) if all_timings else 0,
            'slowest_time': max(all_timings) if all_timings else 0
        }
    else:
        timing_stats = {
            'total_time': total_execution_time,
            'average_time': 0,
            'fastest_algorithm': None,
            'slowest_algorithm': None,
            'fastest_time': 0,
            'slowest_time': 0
        }
    
    metadata = {
        'dataset_shape': df.shape,
        'contamination_used': contamination,
        'algorithms_requested': algorithms,
        'algorithms_successful': successful_algorithms,
        'algorithms_failed': failed,
        'success_rate': len(successful_algorithms) / len(algorithms) if algorithms else 0,
        'timing_stats': timing_stats
    }
    
    if failed:
        print(f"- Failed algorithms: {', '.join([alg.upper() for alg in failed])}")
    
    return {
        'results': results,
        'timings': timings,
        'statistics': statistics,
        'metadata': metadata
    }

def aggregate(results) -> defaultdict[int, int]:
    agreement = defaultdict(int)
    
    for method, mask in results.items():
        for idx, outlier in enumerate(mask):
            if outlier:
                agreement[idx] += 1

    return agreement
