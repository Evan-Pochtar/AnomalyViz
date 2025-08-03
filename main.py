import argparse
import pandas as pd
import warnings
import sys

from src.html import generateHTML
from src.report import printReport
from src.core import runAll, aggregate
from src.data import clean
from src.contamination import estimateOutlierContamination

warnings.filterwarnings("ignore")

def main(file, HTMLreport, ConsoleReport, algorithms=None, contamination=None):
    print(f"Loading dataset from {file}...")
    df = pd.read_csv(file)
    
    try:
        df_clean = clean(df)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if contamination is None:
        contamination = estimateOutlierContamination(df_clean)
        print(f"Estimated contamination: {contamination:.2f}")

    if contamination < 0 or contamination > 0.5:
        print("Contamination must be between 0 and 0.5.")
        sys.exit(1)

    print("Running outlier detection algorithms...")
    results = runAll(df_clean, algorithms, contamination)
    agreement = aggregate(results)

    if ConsoleReport:
        printReport(results, agreement)
    
    if HTMLreport:
        generateHTML(df_clean, results, agreement)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnomalyViz: Visual Outlier Detector for CSV data")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--NoHtmlReport", action='store_false', required=False, help="Flag to disable HTML report generation")
    parser.add_argument("--NoConsoleReport", action='store_false', required=False, help="Flag to disable console text report generation")
    parser.add_argument("--algorithms", type=str, nargs='+', required=False, 
                       choices=['zscore', 'dbscan', 'isoforest', 'lof', 'svm', 'elliptic', 'knn', 'mcd', 'abod', 'hbos'],
                       help="Select specific algorithms to run. If not specified, all algorithms will be run.")
    parser.add_argument("--contamination", type=float, required=False, help="Estimated contamination rate for outlier detection. If not specified, it will be estimated from the data. Must be between 0 and 0.5.")
    
    args = parser.parse_args()
    main(args.file, args.NoHtmlReport, args.NoConsoleReport, args.algorithms, args.contamination)
