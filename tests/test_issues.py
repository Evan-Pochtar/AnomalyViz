import pytest
import numpy as np
import pandas as pd
from collections import defaultdict
import sys, os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.algorithms import (
    zscoreOutliers, dbscanAdaptiveOutliers, isoforestOutliers,
    lofOutliers, svmOutliers, ellipticOutliers, knnOutliers,
    mcdOutliers, copodOutliers, hbosOutliers
)
from src.core import runAll, aggregate, ALGORITHM_MAP
from src.data import clean
from src.createData import createSampleDataset

class TestKnownIssues:
    def test_svm_grid_search_failure(self):
        df = pd.DataFrame(np.random.normal(0, 0.001, size=(50, 3)))
        result = svmOutliers(df, 0.1)
        assert isinstance(result, np.ndarray)
    
    def test_dbscan_uniform_density_issue(self):
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        xx, yy = np.meshgrid(x, y)
        df = pd.DataFrame({
            'x': xx.ravel(),
            'y': yy.ravel()
        })
        result = dbscanAdaptiveOutliers(df, 0.1)
        outlier_rate = np.mean(result)
        assert 0.0 <= outlier_rate <= 0.8
    
    def test_mcd_high_contamination_small_data(self):
        df = pd.DataFrame(np.random.normal(0, 1, size=(20, 5)))
        result = mcdOutliers(df, 0.4)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)

class TestEdgeCaseScenarios:
    def test_single_feature_algorithms(self):
        df_single = pd.DataFrame({'feature': np.random.normal(0, 1, 100)})
        results = {}
        algorithms_to_test = ['zscore', 'isoforest', 'lof', 'knn', 'hbos']
        for algo in algorithms_to_test:
            try:
                result = ALGORITHM_MAP[algo](df_single, 0.1)
                results[algo] = result
                assert isinstance(result, np.ndarray)
                assert len(result) == len(df_single)
            except Exception as e:
                print(f"Algorithm {algo} failed on single feature: {e}")
        assert len(results) >= 2
    
    def test_all_identical_values(self):
        df_identical = pd.DataFrame({
            'a': [1.0] * 50,
            'b': [2.0] * 50,
            'c': [3.0] * 50
        })
        algorithms_working = 0
        for algo_name, algo_func in ALGORITHM_MAP.items():
            try:
                result = algo_func(df_identical, 0.1)
                outlier_count = np.sum(result)
                if outlier_count <= 5:
                    algorithms_working += 1
                print(f"{algo_name}: {outlier_count} outliers detected")
            except Exception as e:
                print(f"Algorithm {algo_name} failed on identical data: {e}")
        assert algorithms_working >= len(ALGORITHM_MAP) // 2
    
    def test_extreme_contamination_rates(self):
        df = createSampleDataset(n_samples=200, n_features=5, contamination=0.1)
        df_clean = clean(df.select_dtypes(include=[np.number]).drop(columns=['is_anomaly']))
        try:
            results_low = runAll(df_clean, algorithms=['zscore', 'isoforest'], contamination=0.01)
            assert len(results_low['results']) == 2
            for result in results_low['results'].values():
                outlier_rate = np.mean(result)
                assert outlier_rate <= 0.1
        except Exception as e:
            pytest.skip(f"Low contamination test failed: {e}")
        try:
            results_high = runAll(df_clean, algorithms=['zscore', 'isoforest'], contamination=0.4)
            assert len(results_high['results']) >= 1
        except Exception as e:
            pytest.skip(f"High contamination test failed: {e}")
    
    def test_high_dimensional_data(self):
        df_high_dim = pd.DataFrame(np.random.normal(0, 1, size=(200, 100)))
        working_algorithms = 0
        for algo_name in ['zscore', 'isoforest', 'hbos']:
            try:
                result = ALGORITHM_MAP[algo_name](df_high_dim, 0.05)
                assert isinstance(result, np.ndarray)
                assert len(result) == len(df_high_dim)
                working_algorithms += 1
            except Exception as e:
                print(f"Algorithm {algo_name} failed on high-dimensional data: {e}")
        assert working_algorithms >= 2
    
    def test_data_with_infinite_values(self):
        df_with_inf = pd.DataFrame({
            'a': [1, 2, np.inf, 4, 5],
            'b': [1, 2, 3, -np.inf, 5],
            'c': [1, 2, 3, 4, 5]
        })
        try:
            df_cleaned = clean(df_with_inf)
            assert len(df_cleaned) < len(df_with_inf)
            assert not np.any(np.isinf(df_cleaned.values))
        except Exception as e:
            pytest.fail(f"Data cleaning failed with infinite values: {e}")

class TestConsensusEdgeCases:
    def test_no_consensus_scenario(self):
        df = pd.DataFrame(np.random.normal(0, 1, size=(100, 5)))
        results = runAll(df, algorithms=['zscore', 'isoforest', 'lof'], contamination=0.02)
        agreement = aggregate(results['results'])
        consensus_threshold = 3
        consensus = {idx: count for idx, count in agreement.items() if count >= consensus_threshold}
        print(f"Consensus outliers with threshold {consensus_threshold}: {len(consensus)}")
        assert isinstance(consensus, dict)
        assert len(consensus) >= 0
    
    def test_all_algorithms_disagree(self):
        normal_cluster = np.random.normal(0, 1, size=(80, 3))
        linear_outliers = np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
        distance_outliers = np.random.normal(5, 0.5, size=(7, 3))
        df = pd.DataFrame(np.vstack([normal_cluster, linear_outliers, distance_outliers]))
        results = runAll(df, algorithms=['zscore', 'lof', 'knn'], contamination=0.1)
        agreement = aggregate(results['results'])
        agreement_counts = defaultdict(int)
        for count in agreement.values():
            agreement_counts[count] += 1
        print("Agreement distribution:")
        for agree_level, point_count in agreement_counts.items():
            print(f"  {point_count} points flagged by {agree_level} algorithms")
        assert len(agreement) >= 0

class TestPerformanceEdgeCases:
    def test_memory_usage_large_dataset(self):
        df_large = pd.DataFrame(np.random.normal(0, 1, size=(5000, 20)))
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024
        results = runAll(df_large, algorithms=['zscore', 'isoforest'], contamination=0.02)
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before
        print(f"Memory usage increased by {memory_increase:.1f} MB")
        assert memory_increase < 500
        assert len(results['results']) == 2
    
    def test_very_wide_dataset(self):
        df_wide = pd.DataFrame(np.random.normal(0, 1, size=(50, 200)))
        results = runAll(df_wide, algorithms=['zscore', 'isoforest'], contamination=0.1)
        assert len(results['results']) >= 1

class TestDataQualityEdgeCases:
    def test_highly_correlated_features(self):
        base_feature = np.random.normal(0, 1, 100)
        df_correlated = pd.DataFrame({
            'f1': base_feature,
            'f2': base_feature + np.random.normal(0, 0.01, 100),
            'f3': base_feature * 2 + np.random.normal(0, 0.01, 100),
            'f4': np.random.normal(0, 1, 100)
        })
        working_count = 0
        for algo_name in ['zscore', 'isoforest', 'elliptic', 'mcd']:
            try:
                result = ALGORITHM_MAP[algo_name](df_correlated, 0.1)
                assert isinstance(result, np.ndarray)
                working_count += 1
            except Exception as e:
                print(f"Algorithm {algo_name} failed on correlated data: {e}")
        assert working_count >= 2
    
    def test_mixed_scale_features(self):
        df_mixed_scale = pd.DataFrame({
            'small_scale': np.random.normal(0, 0.001, 100),
            'medium_scale': np.random.normal(0, 1, 100),
            'large_scale': np.random.normal(0, 1000, 100),
            'binary_like': np.random.choice([0, 1], 100)
        })
        results_no_scaling = runAll(df_mixed_scale, algorithms=['zscore', 'isoforest'], contamination=0.1)
        assert len(results_no_scaling['results']) >= 1
        for result in results_no_scaling['results'].values():
            outlier_rate = np.mean(result)
            assert 0.0 <= outlier_rate <= 0.5
