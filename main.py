import argparse
import pandas as pd
import sys
import numpy as np
import os
from src.html import generateHTML
from src.report import printReport
from src.core import runAll, aggregate
from src.data import clean
from src.contamination import estimateOutlierContamination

def getUniquePath(path: str) -> str:
    base, ext = os.path.splitext(path)
    counter = 1
    candidate = path
    while os.path.exists(candidate):
        candidate = f"{base}({counter}){ext}"
        counter += 1
    if candidate != path:
        print(f"File exists, using new file name: {candidate}")
    return candidate

def PrintProgressBar(prefix: str, completed: int, total: int, width: int = 40, suffix: str = "") -> None:
    if not hasattr(PrintProgressBar, "_last_len"):
        PrintProgressBar._last_len = 0
    pct = 0 if total == 0 else int((completed / total) * 100)
    filled = int(width * completed // max(1, total))
    bar = "[" + ("#" * filled) + ("-" * (width - filled)) + f"] {pct:3d}%"
    line = f"{prefix} {bar} {suffix}"
    pad = " " * max(0, PrintProgressBar._last_len - len(line))
    sys.stdout.write("\r" + line + pad)
    sys.stdout.flush()
    PrintProgressBar._last_len = len(line)


def main(file, HTMLreport, ConsoleReport, algorithms=None, contamination=None, consensusThreshold=None, columns=None, removeOutliers=False, output=None):
    """
    Execute the complete outlier detection pipeline on a CSV dataset.
    
    This is the main orchestration function that coordinates the entire outlier detection
    workflow from data loading through report generation. It handles data preprocessing,
    contamination estimation, algorithm execution, consensus analysis, and report output.

    Args:
        file (str): Path to CSV file containing the dataset to analyze
        HTMLreport (bool): Whether to generate HTML report with visualizations
        ConsoleReport (bool): Whether to print console text report
        algorithms (list[str], optional): Specific algorithms to run. If None, runs all available
        contamination (float, optional): Expected contamination rate (0.0-0.5). 
                                       If None, automatically estimated from data
        consensusThreshold (int, optional): Minimum algorithms required for consensus.
                                          If None, defaults to 50% of algorithms (rounded up)
                                          
    Returns:
        None: Function performs side effects (file I/O, console output) rather than returning values
        
    Raises:
        SystemExit: On data validation errors, invalid parameters, or file loading issues
    """

    print(f"Loading dataset from {file}...")
    df = pd.read_csv(file)

    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            print(f"Error: Columns not found in dataset: {missing}")
            sys.exit(1)
        df = df[columns]
   
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

    if not algorithms:
        from src.core import ALGORITHM_MAP
        algorithms = list(ALGORITHM_MAP.keys())

    totalAlgos = len(algorithms)

    def LocalProgressCallback(kind, doneCount, totalCount, currentName="", extra=None):
        if kind == "algo_start":
            completed = doneCount - 1
            PrintProgressBar("Algorithms", completed, totalAlgos, suffix=f"Running: {currentName}")
        elif kind == "algo_done":
            completed = doneCount
            remaining = extra.get("remaining", []) if extra else []
            remText = f" | Remaining: {', '.join(remaining[:5])}" if remaining else ""
            PrintProgressBar("Algorithms", completed, totalAlgos, suffix=f"Completed: {currentName}{remText}")
        elif kind == "algorithms_complete":
            PrintProgressBar("Algorithms", totalAlgos, totalAlgos, suffix="All algorithms finished")

    results = runAll(df_clean, algorithms, contamination, progress_callback=LocalProgressCallback)
    print("")  # newline after progress bar
    agreement = aggregate(results['results'])
    
    numAlgos = len(results['results'])
    if consensusThreshold is None:
        consensusThreshold = max(1, int(np.ceil(numAlgos * 0.5)))
        print(f"Using default consensus threshold: {consensusThreshold}")
    else:
        if consensusThreshold < 1:
            print("Error: Consensus threshold must be at least 1.")
            sys.exit(1)
        if consensusThreshold > len(results):
            print(f"Error: Consensus threshold ({consensusThreshold}) cannot exceed the number of algorithms ({numAlgos}).")
            sys.exit(1)

    if ConsoleReport:
        printReport(results, agreement, consensusThreshold)
   
    if HTMLreport:
        generateHTML(df_clean, results['results'], agreement, consensusThreshold=consensusThreshold)

    if removeOutliers:
        originalIndexList = list(df_clean.index)
        keepIndexLabels = [originalIndexList[i] for i in range(len(originalIndexList)) if agreement.get(i, 0) < consensusThreshold]
        removedCount = len(originalIndexList) - len(keepIndexLabels)

        print(f"Removing {removedCount} rows flagged by consensus (threshold={consensusThreshold}).")
        df_cleaned = df_clean.loc[keepIndexLabels].copy()
        filteredResults = {}
        for alg, series in results['results'].items():
            filteredResults[alg] = series.loc[keepIndexLabels]

        results['results'] = filteredResults
        agreement = aggregate(results['results'])
        outPath = output if output else file
        outPath = getUniquePath(outPath)

        df_cleaned.to_csv(outPath, index=False)
        print(f"Wrote cleaned CSV to: {outPath}")

if __name__ == "__main__":
    """
    Command-line interface for AnomalyViz outlier detection tool.
    
    Provides a comprehensive command-line interface for running outlier detection
    on CSV datasets with configurable parameters. Supports algorithm selection,
    contamination estimation, consensus analysis, and multiple report formats.
    
    Command-line arguments:
    - file: Required CSV file path
    - NoHtmlReport: Flag to disable HTML report (default: enabled)
    - NoConsoleReport: Flag to disable console report (default: enabled)  
    - algorithms: Optional list of specific algorithms to run (runs all if not specified)
    - contamination: Optional contamination rate (auto-estimated if not provided)
    - consensusThreshold: Optional consensus threshold (defaults to 50% if not provided)
    
    Example usage:
        python main.py --file data.csv
        python main.py --file data.csv --algorithms zscore isoforest --contamination 0.1
        python main.py --file data.csv --consensusThreshold 3 --NoConsoleReport
    """

    parser = argparse.ArgumentParser(description="AnomalyViz: Visual Outlier Detector for CSV data")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--NoHtmlReport", action='store_false', required=False, help="Flag to disable HTML report generation")
    parser.add_argument("--NoConsoleReport", action='store_false', required=False, help="Flag to disable console text report generation")
    parser.add_argument("--algorithms", type=str, nargs='+', required=False,
                       choices=['zscore', 'dbscan', 'isoforest', 'lof', 'svm', 'elliptic', 'knn', 'mcd', 'copod', 'hbos'],
                       help="Select specific algorithms to run. If not specified, all algorithms will be run.")
    parser.add_argument("--contamination", type=float, required=False, help="Estimated contamination rate for outlier detection. If not specified, it will be estimated from the data. Must be between 0 and 0.5.")
    parser.add_argument("--consensusThreshold", type=int, required=False, help="Number of algorithms that must agree for a consensus outlier. Must be between 1 and the total number of algorithms. If not specified, defaults to 50%% of algorithms (rounded up).")
    parser.add_argument("--columns", type=str, nargs='+', required=False, help="Subset of columns to include in outlier detection. If not specified, all columns are used.")
    parser.add_argument("--removeOutliers", action='store_true', required=False, help="Automatically remove consensus outliers and write cleaned CSV.")
    parser.add_argument("--output", type=str, required=False, help="Path to write cleaned CSV when --removeOutliers is used.")
   
    args = parser.parse_args()
    main(args.file, args.NoHtmlReport, args.NoConsoleReport, args.algorithms, args.contamination, args.consensusThreshold, args.columns, args.removeOutliers, args.output)
