# AnomalyViz

A comprehensive anomaly detection tool that applies multiple machine learning algorithms to identify outliers in CSV datasets through consensus-based analysis.

## Overview

AnomalyViz is a Python-based outlier detection system that runs multiple anomaly detection algorithms on numerical CSV data and provides consensus results. It generates both console reports and interactive HTML visualizations to help you identify and analyze anomalies in your datasets. This project is in a very early stage and still in concurrent development.

### Key Features

- **Multi-Algorithm Consensus**: Runs up to 10 different outlier detection algorithms
- **Flexible Algorithm Selection**: Choose specific algorithms or run all available methods
- **Dual Reporting**: Console output and interactive HTML reports with visualizations
- **Automatic Data Cleaning**: Handles missing values and non-numerical data
- **Visual Analysis**: Generates matplotlib plots for each algorithm's results
- **Configurable Consensus Threshold**: Adjustable agreement requirements for outlier classification

## Supported Algorithms

1. **Z-Score** - Statistical outlier detection using standard deviations
2. **DBSCAN** - Density-based clustering with adaptive parameters
3. **Isolation Forest** - Tree-based anomaly detection
4. **Local Outlier Factor (LOF)** - Local density-based outlier detection
5. **One-Class SVM** - Support Vector Machine with grid search optimization
6. **Elliptic Envelope** - Gaussian distribution-based detection
7. **K-Nearest Neighbors (KNN)** - Distance-based outlier detection
8. **Minimum Covariance Determinant (MCD)** - Robust covariance estimation
9. **Angle-Based Outlier Detection (ABOD)** - Variance of angles between data points
10. **Histogram-Based Outlier Score (HBOS)** - Histogram-based anomaly scoring

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Evan-Pochtar/AnomalyViz.git
cd AnomalyViz
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run all algorithms on your CSV file:
```bash
python main.py --file path/to/your/data.csv
```

### Advanced Options

**Select specific algorithms:**
```bash
python main.py --file data.csv --algorithms zscore isoforest lof
```

**Disable HTML report:**
```bash
python main.py --file data.csv --NoHtmlReport
```

**Disable console report:**
```bash
python main.py --file data.csv --NoConsoleReport
```

**Run only specific algorithms with custom reporting:**
```bash
python main.py --file data.csv --algorithms dbscan svm elliptic --NoConsoleReport
```

### Command Line Arguments

- `--file` (required): Path to your CSV file
- `--algorithms` (optional): Space-separated list of algorithms to run
  - Available: `zscore`, `dbscan`, `isoforest`, `lof`, `svm`, `elliptic`, `knn`, `mcd`, `abod`, `hbos`
- `--NoHtmlReport`: Disable HTML report generation
- `--NoConsoleReport`: Disable console text report

## Data Requirements

- **Format**: CSV files with numerical data
- **Missing Values**: Automatically handled during data cleaning
- **Non-numerical Columns**: Automatically filtered out
- **Minimum Rows**: At least 10 rows recommended for reliable results

## Output

### Sample Console Report
```
===== Outlier Detection Report =====
[Algorithm: ZSCORE] Detected 27 outliers.
[Algorithm: ISOFOREST] Detected 29 outliers.
[Algorithm: LOF] Detected 25 outliers.
...
[Consensus] Rows flagged as outliers by 5+ algorithms:
Row 475 flagged by 9/9 algorithms (100%)
Row 476 flagged by 9/9 algorithms (100%)
...
Total consensus outliers: 25
```

### HTML Report
- Consensus analysis with detailed breakdowns
- Data distribution plots
- Algorithm comparison charts

### Generated Files
- `plots/`: Individual matplotlib plots for each algorithm
- `report/AnomalyReport.html`: Comprehensive HTML report with interactive elements

## Consensus Mechanism

AnomalyViz uses a consensus-based approach to identify the most reliable outliers:

- **Default Threshold**: ≥50% of algorithms must agree (e.g., 5 out of 9 algorithms)
- **Confidence Levels**: Results show percentage agreement for each detected outlier
- **Robustness**: Reduces false positives by requiring multiple algorithm agreement

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

Or run specific tests:
```bash
python tests/test_all.py
```

## Sample Data Generation

Generate test data for experimentation:
```python
from src.createData import createSampleDataset
createSampleDataset(n_samples=500, contamination=0.05)
```

## Performance Considerations

- **Dataset Size**: Algorithms scale differently; ABOD may be slow on large datasets (>10k rows)
- **Memory Usage**: SVM grid search is memory-intensive for large datasets
- **Processing Time**: Running all algorithms may take several minutes on large datasets

## Dependencies

- pandas: Data manipulation and analysis
- scikit-learn: Machine learning algorithms
- numpy: Numerical computing
- scipy: Scientific computing
- matplotlib: Plotting and visualization
- argparse: Command-line interface

## License

This project is open source. Please check the LICENSE file for details.

## Future Enhancements

- Robust testing to verify correct functionality
- Automatic outlier contamination solver 
- Ability to find outliers in text, video, or picture data
- Additional visualization options
- Custom algorithm integration framework
- Automated parameter tuning for all algorithms
- Increased performance on large datasets
