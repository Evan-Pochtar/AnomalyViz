import os
import numpy as np
from jinja2 import Template
import pandas as pd
from collections import defaultdict

from src.visualization import PCAvisualization, createSummaryViz

def generateHTML(df: pd.DataFrame, results: dict[str, pd.Series], agreement: defaultdict[int, int], outputPath: str = "report/AnomalyReport.html", consensusThreshold: int = None) -> None:
    """
    Generate a comprehensive HTML report with visualizations for outlier detection results.
    
    Creates an interactive HTML report that combines outlier detection results from multiple
    algorithms with visual representations. The report includes consensus analysis,
    individual algorithm results, and various plots to help interpret the findings.
    
    Report generation strategy:
    1. Calculate consensus outliers based on threshold
    2. Generate and saves PCA visualizations for consensus and individual algorithms
    3. Create and saves summary statistics visualization
    4. Render HTML template with all results and plots
    5. Save complete report with embedded visualizations
    
    Consensus logic:
    - Default threshold: 50% of algorithms must agree (rounded up)
    - Custom threshold can be specified
    
    Args:
        df (pd.DataFrame): Original dataset used for outlier detection
        results (dict[str, pd.Series]): Algorithm results mapping names to outlier masks
        agreement (defaultdict[int, int]): Agreement counts per data point index
        outputPath (str): File path for saving HTML report (default: "report/AnomalyReport.html")
        consensusThreshold (int, optional): Minimum algorithms required for consensus.
                                          If None, uses 50% of total algorithms (rounded up)
                                          
    Returns:
        None: Saves HTML report to specified path and prints summary statistics
        
    Raises:
        FileNotFoundError: If HTML template file cannot be found
    """

    numAlgos = len(results)
    if consensusThreshold is not None:
        conThreshold = consensusThreshold
    else:
        conThreshold = max(1, int(np.ceil(numAlgos * 0.5)))
    
    consensus = {idx: count for idx, count in agreement.items() if count >= conThreshold}

    os.makedirs('plots', exist_ok=True)
    print("Generating visualizations...")
    
    conMask = np.array([agreement.get(i, 0) >= conThreshold for i in range(len(df))])
    PCAvisualization(df, conMask, "Consensus Outliers (2D)", dim=2, 
                    savePath='plots/pca_consensus_2d.png', dpi=300)
    if df.shape[1] >= 3:
        PCAvisualization(df, conMask, "Consensus Outliers (3D)", dim=3, 
                        savePath='plots/pca_consensus_3d.png', dpi=300)
    
    outlierPlots = {}
    for algo, mask in results.items():
        plot_path = f'plots/pca_{algo}_2d.png'
        PCAvisualization(df, mask, f"{algo.upper()} Outliers Detection", 
                        dim=2, savePath=plot_path, dpi=300)
        outlierPlots[algo] = f'../plots/pca_{algo}_2d.png'

    createSummaryViz(results, df.shape, 'plots/algorithm_summary.png')

    template_path = os.path.join(os.path.dirname(__file__), 'html/reportTemplate.html')
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            templateCode = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"HTML template file not found at {template_path}. Please ensure 'report_template.html' exists in the same directory as this script.")

    maxAgreement = max(agreement.values()) if agreement else 0
    maxAgreeCount = len([idx for idx, count in agreement.items() if count == maxAgreement]) if agreement else 0

    template = Template(templateCode)
    htmlOutput = template.render(
        results=results, 
        consensus=consensus, 
        outlierPlots=outlierPlots, 
        df=df,
        numAlgos=numAlgos,
        conThreshold=conThreshold,
        maxAgreement=maxAgreement,
        maxAgreeCount=maxAgreeCount
    )
    
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    with open(outputPath, "w", encoding="utf-8") as f:
        f.write(htmlOutput)
    
    print(f"\nHTML report saved to {outputPath}")
    print(f"Consensus threshold: {conThreshold}/{numAlgos} algorithms (>={(conThreshold/numAlgos)*100:.0f}%)")
    print(f"Consensus outliers found: {len(consensus)}")
