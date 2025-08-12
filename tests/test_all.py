import pytest
import numpy as np
import pandas as pd
import time
import warnings
import tempfile
import os
import sys
from collections import defaultdict
from unittest.mock import patch, MagicMock
import matplotlib

matplotlib.use('Agg')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.algorithms import (
    zscoreOutliers, dbscanAdaptiveOutliers, isoforestOutliers,
    lofOutliers, svmOutliers, ellipticOutliers, knnOutliers,
    mcdOutliers, copodOutliers, hbosOutliers
)
from src.core import runAll, aggregate, ALGORITHM_MAP
from src.data import clean
from src.contamination import estimateOutlierContamination
from src.createData import createSampleDataset
from src.visualization import PCAvisualization, createSummaryViz
from src.html import generateHTML
from src.report import printReport

@pytest.fixture
def small_dataset():
    return createSampleDataset(n_samples=100, n_features=5, contamination=0.1)

@pytest.fixture
def medium_dataset():
    return createSampleDataset(n_samples=1000, n_features=10, contamination=0.05)

@pytest.fixture
def large_dataset():
    return createSampleDataset(n_samples=10000, n_features=20, contamination=0.02)

@pytest.fixture
def high_dim_dataset():
    return createSampleDataset(n_samples=500, n_features=50, contamination=0.1)

@pytest.fixture
def clean_small_data(small_dataset):
    df_numeric = small_dataset.select_dtypes(include=[np.number]).drop(columns=['is_anomaly'], errors='ignore')
    return clean(df_numeric)

@pytest.fixture
def clean_medium_data(medium_dataset):
    df_numeric = medium_dataset.select_dtypes(include=[np.number]).drop(columns=['is_anomaly'], errors='ignore')
    return clean(df_numeric)

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

class TestDataHandling:
    def test_createSampleDataset_basic(self):
        df = createSampleDataset(n_samples=100, n_features=5, contamination=0.1)
        assert len(df) == 100
        assert len([col for col in df.columns if col.startswith('feature_')]) == 5
        assert 'is_anomaly' in df.columns
        assert df['is_anomaly'].sum() == 10
        
    def test_createSampleDataset_edge_cases(self):
        df_small = createSampleDataset(n_samples=10, n_features=2, contamination=0.2)
        assert len(df_small) == 10
        assert df_small['is_anomaly'].sum() == 2
        
        df_high_contam = createSampleDataset(n_samples=100, n_features=3, contamination=0.4)
        assert df_high_contam['is_anomaly'].sum() == 40
        
    def test_clean_function_valid_data(self, small_dataset):
        df_numeric = small_dataset.select_dtypes(include=[np.number])
        cleaned = clean(df_numeric)
        assert not cleaned.empty
        assert cleaned.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all()
        
    def test_clean_function_with_missing_values(self):
        df_with_nan = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [1, np.nan, 3, 4],
            'c': [1, 2, 3, 4]
        })
        cleaned = clean(df_with_nan)
        assert len(cleaned) == 2
        
    def test_clean_function_error_cases(self):
        with pytest.raises(ValueError, match="Input DataFrame is empty or None"):
            clean(pd.DataFrame())
        df_text = pd.DataFrame({'text': ['a', 'b', 'c']})
        with pytest.raises(ValueError, match="No numeric columns with valid finite values found"):
            clean(df_text)

class TestContaminationEstimation:
    def test_estimateOutlierContamination_known_contamination(self, small_dataset):
        df_numeric = small_dataset.select_dtypes(include=[np.number]).drop(columns=['is_anomaly'])
        estimated = estimateOutlierContamination(df_numeric)
        assert 0.01 <= estimated <= 0.3
        
    def test_estimateOutlierContamination_large_dataset(self, large_dataset):
        df_numeric = large_dataset.select_dtypes(include=[np.number]).drop(columns=['is_anomaly'])
        estimated = estimateOutlierContamination(df_numeric)
        assert 0.001 <= estimated <= 0.3
        
    def test_estimateOutlierContamination_normal_data(self):
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (1000, 5))
        df_normal = pd.DataFrame(normal_data)
        estimated = estimateOutlierContamination(df_normal)
        assert estimated <= 0.2

class TestIndividualAlgorithms:
    @pytest.mark.parametrize("algorithm_func", [
        zscoreOutliers, isoforestOutliers, lofOutliers, ellipticOutliers,
        knnOutliers, mcdOutliers, hbosOutliers
    ])
    def test_algorithm_basic_functionality(self, algorithm_func, clean_small_data):
        contamination = 0.1
        result = algorithm_func(clean_small_data, contamination)
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool
        assert len(result) == len(clean_small_data)
        actual_contamination = np.mean(result)
        assert actual_contamination <= 0.5
        
    def test_zscore_algorithm_specific(self, clean_small_data):
        result = zscoreOutliers(clean_small_data, 0.1)
        outlier_rate = np.mean(result)
        assert 0.0 <= outlier_rate <= 0.3
        
    def test_dbscan_algorithm_specific(self, clean_small_data):
        result = dbscanAdaptiveOutliers(clean_small_data, 0.1)
        outlier_rate = np.mean(result)
        assert 0.0 <= outlier_rate <= 0.5
        
    def test_svm_algorithm_specific(self, clean_small_data):
        result = svmOutliers(clean_small_data, 0.1)
        outlier_rate = np.mean(result)
        assert 0.0 <= outlier_rate <= 0.5
        
    def test_abod_algorithm_specific(self, clean_small_data):
        result = copodOutliers(clean_small_data, 0.1)
        outlier_rate = np.mean(result)
        assert 0.0 <= outlier_rate <= 0.5
        
    def test_algorithm_with_different_contamination_rates(self, clean_small_data):
        contamination_rates = [0.01, 0.05, 0.1, 0.2]
        for contamination in contamination_rates:
            result = isoforestOutliers(clean_small_data, contamination)
            actual_rate = np.mean(result)
            assert actual_rate <= contamination * 5

class TestCoreFunctionality:
    def test_runAll_basic(self, clean_small_data):
        results = runAll(clean_small_data, algorithms=['zscore', 'isoforest'], contamination=0.1)
        assert isinstance(results, dict)
        assert 'results' in results
        assert 'timings' in results
        assert 'statistics' in results
        assert 'metadata' in results
        assert len(results['results']) == 2
        assert all(isinstance(mask, pd.Series) for mask in results['results'].values())
        
    def test_runAll_all_algorithms(self, clean_medium_data):
        results = runAll(clean_medium_data, contamination=0.05)
        assert len(results['results']) >= 5
        assert len(results['metadata']['algorithms_successful']) >= 5
        
    def test_runAll_invalid_algorithm(self, clean_small_data):
        with pytest.raises(ValueError, match="Invalid algorithm"):
            runAll(clean_small_data, algorithms=['invalid_algo'], contamination=0.1)
            
    def test_runAll_invalid_contamination(self, clean_small_data):
        with pytest.raises(ValueError, match="Contamination must be between"):
            runAll(clean_small_data, contamination=0.6)
        with pytest.raises(ValueError, match="Contamination must be between"):
            runAll(clean_small_data, contamination=-0.1)
            
    def test_aggregate_function(self, clean_small_data):
        results = runAll(clean_small_data, algorithms=['zscore', 'isoforest', 'lof'], contamination=0.1)
        agreement = aggregate(results['results'])
        assert isinstance(agreement, defaultdict)
        assert all(isinstance(k, (int, np.integer)) for k in agreement.keys())
        assert all(isinstance(v, (int, np.integer)) for v in agreement.values())
        assert all(1 <= v <= 3 for v in agreement.values())

class TestPerformance:
    def test_performance_medium_dataset(self, clean_medium_data):
        start_time = time.time()
        results = runAll(clean_medium_data, algorithms=['zscore', 'isoforest', 'lof'], contamination=0.05)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 30
        assert all(timing is not None for timing in results['timings'].values())
        
    def test_performance_large_dataset_subset(self, large_dataset):
        df_clean = clean(large_dataset.select_dtypes(include=[np.number]).drop(columns=['is_anomaly']))
        start_time = time.time()
        results = runAll(df_clean, algorithms=['zscore', 'isoforest'], contamination=0.02)
        end_time = time.time()
        execution_time = end_time - start_time
        assert execution_time < 60
        
    def test_algorithm_timing_comparison(self, clean_medium_data):
        results = runAll(clean_medium_data, algorithms=['mcd', 'isoforest'], contamination=0.05)
        timings = results['timings']
        for timing in timings.values():
            print(timing)
        assert all(timing > 0 for timing in timings.values() if timing is not None)

class TestQuality:
    def test_quality_with_known_outliers(self):
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (900, 5))
        outlier_data = np.random.normal(5, 1, (100, 5))
        all_data = np.vstack([normal_data, outlier_data])
        true_labels = np.hstack([np.zeros(900), np.ones(100)])
        df = pd.DataFrame(all_data)
        results = runAll(df, algorithms=['zscore', 'isoforest', 'lof'], contamination=0.1)
        agreement = aggregate(results['results'])
        consensus_threshold = 2
        consensus_outliers = {idx: count for idx, count in agreement.items() if count >= consensus_threshold}
        true_outlier_indices = set(range(900, 1000))
        consensus_indices = set(consensus_outliers.keys())
        if len(consensus_indices) > 0:
            precision = len(consensus_indices & true_outlier_indices) / len(consensus_indices)
            assert precision >= 0.3
            
    def test_consensus_threshold_behavior(self, clean_medium_data):
        results = runAll(clean_medium_data, algorithms=['zscore', 'isoforest', 'lof', 'knn'], contamination=0.05)
        agreement = aggregate(results['results'])
        consensus_counts = {}
        for threshold in range(1, 5):
            consensus = {idx: count for idx, count in agreement.items() if count >= threshold}
            consensus_counts[threshold] = len(consensus)
        if consensus_counts[1] > 0:
            assert consensus_counts[4] <= consensus_counts[1]

class TestVisualization:
    def test_pca_visualization_2d(self, clean_small_data, temp_dir):
        outlier_mask = np.random.choice([True, False], size=len(clean_small_data), p=[0.1, 0.9])
        save_path = os.path.join(temp_dir, 'test_pca_2d.png')
        PCAvisualization(clean_small_data, outlier_mask, "Test 2D", dim=2, savePath=save_path)
        assert os.path.exists(save_path)
        
    def test_pca_visualization_3d(self, clean_small_data, temp_dir):
        outlier_mask = np.random.choice([True, False], size=len(clean_small_data), p=[0.1, 0.9])
        save_path = os.path.join(temp_dir, 'test_pca_3d.png')
        if clean_small_data.shape[1] >= 3:
            PCAvisualization(clean_small_data, outlier_mask, "Test 3D", dim=3, savePath=save_path)
            assert os.path.exists(save_path)
            
    def test_pca_visualization_invalid_dim(self, clean_small_data):
        outlier_mask = np.random.choice([True, False], size=len(clean_small_data))
        with pytest.raises(ValueError, match="Dimension must be 2 or 3"):
            PCAvisualization(clean_small_data, outlier_mask, "Test", dim=4)
            
    def test_summary_visualization(self, temp_dir):
        results = {
            'zscore': np.array([True, False, True, False]),
            'isoforest': np.array([False, True, True, False]),
            'lof': np.array([True, True, False, False])
        }
        save_path = os.path.join(temp_dir, 'summary.png')
        createSummaryViz(results, (4, 3), save_path)
        assert os.path.exists(save_path)

class TestHTMLReport:
    def test_html_generation_basic(self, clean_small_data, temp_dir):
        results = runAll(clean_small_data, algorithms=['zscore', 'isoforest'], contamination=0.1)
        agreement = aggregate(results['results'])
        output_path = os.path.join(temp_dir, 'test_report.html')
        generateHTML(clean_small_data, results['results'], agreement, outputPath=output_path)
        assert os.path.exists(output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '<html' in content
            assert 'AnomalyViz' in content
            assert 'Consensus' in content

class TestIntegration:
    def test_full_pipeline_small_dataset(self, temp_dir):
        df = createSampleDataset(n_samples=200, n_features=5, contamination=0.1)
        df_clean = clean(df.select_dtypes(include=[np.number]).drop(columns=['is_anomaly']))
        results = runAll(df_clean, algorithms=['zscore', 'isoforest', 'lof'], contamination=0.1)
        agreement = aggregate(results['results'])
        output_path = os.path.join(temp_dir, 'integration_test.html')
        generateHTML(df_clean, results['results'], agreement, outputPath=output_path)
        assert len(results['results']) == 3
        assert os.path.exists(output_path)
        assert isinstance(agreement, defaultdict)
        
    def test_full_pipeline_medium_dataset(self, medium_dataset, temp_dir):
        df_clean = clean(medium_dataset.select_dtypes(include=[np.number]).drop(columns=['is_anomaly']))
        contamination = estimateOutlierContamination(df_clean)
        results = runAll(df_clean, contamination=contamination)
        agreement = aggregate(results['results'])
        assert len(results['results']) == 10
        assert len(agreement) >= 25
        output_path = os.path.join(temp_dir, 'medium_test.html')
        generateHTML(df_clean, results['results'], agreement, outputPath=output_path)
        assert os.path.exists(output_path)

class TestErrorHandling:
    def test_empty_dataframe(self):
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            clean(empty_df)
            
    def test_single_column_data(self):
        df_single = pd.DataFrame({'col1': [1, 2, 3, 4, 5]})
        cleaned = clean(df_single)
        results = runAll(cleaned, algorithms=['zscore'], contamination=0.2)
        assert len(results['results']) == 1
        
    def test_very_small_dataset(self):
        df_tiny = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6]
        })
        results = runAll(df_tiny, contamination=0.33)
        assert len(results['results']) >= 1
        
    def test_high_contamination_rate(self, clean_small_data):
        results = runAll(clean_small_data, contamination=0.4)
        assert len(results['results']) >= 1
        
    def test_algorithm_failure_handling(self, clean_small_data):
        results = runAll(clean_small_data, algorithms=['zscore', 'isoforest'], contamination=0.1)
        assert len(results['results']) >= 1
        assert len(results['metadata']['algorithms_successful']) >= 1
