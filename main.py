import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope, MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import zscore
from collections import defaultdict
import warnings
import sys
import os
from jinja2 import Template

warnings.filterwarnings("ignore")

def clean_data(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("No numeric columns found in the dataset for outlier detection.")
    return numeric_df.dropna()

def clean_data(df):
    df = df.select_dtypes(include=[np.number]).dropna()
    if df.empty:
        raise ValueError("No numeric columns found in the dataset for outlier detection.")
    return df

def zscore_outliers(df, threshold=3):
    return (np.abs(zscore(df)) > threshold).any(axis=1)

def dbscan_outliers_adaptive(df):
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

def isoforest_outliers(df):
    return IsolationForest(contamination='auto', random_state=42).fit_predict(df) == -1

def lof_outliers(df):
    return LocalOutlierFactor(n_neighbors=20).fit_predict(df) == -1

def svm_outliers_tuned(df):
    return OneClassSVM(gamma='auto', nu=0.05).fit_predict(df) == -1

def elliptic_outliers(df):
    return EllipticEnvelope(random_state=42).fit_predict(df) == -1

def knn_outliers(df):
    distances = NearestNeighbors(n_neighbors=5).fit(df).kneighbors(df)[0][:, -1]
    return distances > np.percentile(distances, 95)

def mcd_outliers(df):
    mahal = MinCovDet().fit(df).mahalanobis(df)
    return mahal > np.percentile(mahal, 97.5)

def abod_outliers(df):
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


def hbos_outliers(df):
    n_bins = 10
    scores = np.ones(len(df))

    for col in df.columns:
        hist, bins = np.histogram(df[col], bins=n_bins, density=True)
        idx = np.clip(np.digitize(df[col], bins[:-1], right=True), 0, n_bins - 1)
        prob = np.clip(hist[idx], 1e-6, None)
        scores *= 1 / prob

    return scores > np.percentile(scores, 95)

def run_all_algorithms(df):
    results = {}
    results['zscore'] = zscore_outliers(df)
    results['dbscan'] = dbscan_outliers_adaptive(df)
    results['isoforest'] = isoforest_outliers(df)
    results['lof'] = lof_outliers(df)
    results['svm'] = svm_outliers_tuned(df)
    results['elliptic'] = elliptic_outliers(df)
    results['knn'] = knn_outliers(df)
    results['mcd'] = mcd_outliers(df)
    #results['abod'] = abod_outliers(df)
    results['hbos'] = hbos_outliers(df)
    return results

def aggregate_outliers(results):
    agreement = defaultdict(int)
    for method, mask in results.items():
        for idx, is_outlier in enumerate(mask):
            if is_outlier:
                agreement[idx] += 1
    return agreement

def visualize_pca(df, outlier_mask, title, dim=2, save_path=None):
    pca = PCA(n_components=dim)
    reduced = pca.fit_transform(df)
    plt.figure(figsize=(10, 8))
    if dim == 2:
        sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=outlier_mask, palette={True: 'red', False: 'blue'}, legend=False)
    else:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=['red' if o else 'blue' for o in outlier_mask])
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def print_report(df, results, agreement):
    print("\n===== Outlier Detection Report =====")
    for algo, mask in results.items():
        print(f"\n[Algorithm: {algo.upper()}] Detected {np.sum(mask)} outliers.")

    print("\n[Consensus] Rows flagged as outliers by multiple algorithms:")
    consensus = {idx: count for idx, count in agreement.items() if count > 4}
    if consensus:
        for idx, count in sorted(consensus.items(), key=lambda x: -x[1]):
            print(f"Row {idx} flagged by {count} algorithms.")
    else:
        print("No strong consensus among algorithms.")

def generate_html_report(df, results, agreement, output_path="report/anomaly_report.html"):
    consensus = {idx: count for idx, count in agreement.items() if count > 4}
    os.makedirs('plots', exist_ok=True)
    consensus_mask = np.array([agreement.get(i, 0) > 4 for i in range(len(df))])

    visualize_pca(df, consensus_mask, "Consensus Outliers (2D)", dim=2, save_path='plots/pca_consensus_2d.png')
    if df.shape[1] >=3:
        visualize_pca(df, consensus_mask, "Consensus Outliers (3D)", dim=3, save_path='plots/pca_consensus_3d.png')

    algo_plots = {}
    for algo, mask in results.items():
        visualize_pca(df, mask, f"{algo.upper()} Outliers (2D)", dim=2, save_path=f'plots/pca_{algo}_2d.png')
        algo_plots[algo] = f'../plots/pca_{algo}_2d.png'

    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>AnomalyViz Report</title>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <!-- Google Font -->
      <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet" />
      <!-- DataTables CSS -->
      <link rel="stylesheet" href="https://cdn.datatables.net/1.13.5/css/jquery.dataTables.min.css" />
      <style>
        :root {
          --primary: #0056b3;
          --primary-light: #007bff;
          --bg: #f5f8fa;
          --card-bg: #ffffff;
          --text: #333333;
          --muted: #666666;
        }
        * { box-sizing: border-box; }
        body {
          margin: 0;
          font-family: 'Roboto', sans-serif;
          background: var(--bg);
          color: var(--text);
          line-height: 1.6;
        }
        .container {
          max-width: 80%;
          margin: 0 auto;
          padding: 20px;
        }
        h1 {
          text-align: center;
          margin-bottom: 10px;
          font-weight: 500;
          border-bottom: 3px solid var(--primary);
          padding-bottom: 5px;
        }
        h2, h3, h4 {
          text-align: center;
          margin-top: 1.5em;
          color: var(--primary);
          font-weight: 400;
        }
        table.dataTable {
          width: 100% !important;
          background: var(--card-bg);
          border-radius: 4px;
          overflow: hidden;
          box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }
        table.dataTable thead th {
          background: var(--primary-light);
          color: #fff;
          text-transform: uppercase;
          letter-spacing: 0.02em;
        }
        table.dataTable tbody tr:nth-child(odd) {
          background: #fafafa;
        }
        table.dataTable tbody tr:hover {
          background: #e9f2fd;
        }
        .collapsible {
          display: block;
          background: var(--card-bg);
          color: var(--primary);
          cursor: pointer;
          padding: 15px 20px;
          margin: 20px 0 0;
          border: 1px solid #ddd;
          border-radius: 4px;
          font-size: 18px;
          font-weight: 500;
          box-shadow: 0 1px 4px rgba(0,0,0,0.08);
          transition: background 0.2s, box-shadow 0.2s;
        }
        .collapsible:hover {
          background: #f0f4f8;
          box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        }
        .collapsible.active {
          border-color: var(--primary);
        }
        .content {
          overflow: hidden;
          max-height: 0;
          transition: max-height 0.3s ease;
          background: var(--card-bg);
          border: 1px solid #ddd;
          border-top: none;
          border-radius: 0 0 4px 4px;
          padding: 0 20px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }
        .content.open {
          padding: 20px;
        }
        img {
          max-width: 100%;
          display: block;
          margin: 20px auto;
          border-radius: 4px;
          box-shadow: 0 1px 6px rgba(0,0,0,0.1);
        }
        footer {
          text-align: center;
          font-size: 0.9em;
          color: var(--muted);
          margin: 40px 0 20px;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>AnomalyViz: Outlier Detection Report</h1>

        <button class="collapsible">Algorithm Summary</button>
        <div class="content">
          <table id="algoTable" class="display">
            <thead>
              <tr><th>Algorithm</th><th># Outliers</th></tr>
            </thead>
            <tbody>
              {% for algo, mask in results.items() %}
                <tr><td>{{ algo.upper() }}</td><td>{{ mask.sum() }}</td></tr>
              {% endfor %}
            </tbody>
          </table>
        </div>

        <button class="collapsible">Consensus Outliers (Detected by &gt;3 algorithms)</button>
        <div class="content">
          {% if consensus %}
          <table id="consensusTable" class="display">
            <thead>
              <tr><th>Row Index</th><th># Algorithms</th></tr>
            </thead>
            <tbody>
              {% for idx, count in consensus.items() %}
                <tr><td>{{ idx }}</td><td>{{ count }}</td></tr>
              {% endfor %}
            </tbody>
          </table>
          {% else %}
            <p>No strong consensus among algorithms.</p>
          {% endif %}
        </div>

        <button class="collapsible">Visualizations</button>
        <div class="content">
          <h3>Consensus Outliers</h3>
          <img src="{{ '../plots/pca_consensus_2d.png' }}" alt="Consensus PCA 2D" />
          {% if df.shape[1] >= 3 %}
          <img src="{{ '../plots/pca_consensus_3d.png' }}" alt="Consensus PCA 3D" />
          {% endif %}

          <h3>Individual Algorithm PCA Plots</h3>
          {% for algo, plot in algo_plots.items() %}
            <h4>{{ algo.upper() }}</h4>
            <img src="{{ plot }}" alt="{{ algo }} PCA Plot" />
          {% endfor %}
        </div>

        <footer>
          Generated by AnomalyViz
        </footer>
      </div>

      <script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
      <script src="https://cdn.datatables.net/1.13.5/js/jquery.dataTables.min.js"></script>
      <script>
        $(function() {
          $('#algoTable, #consensusTable').DataTable({
            paging: false,
            info: false,
            searching: false
          });

          $('.collapsible').on('click', function(){
            $(this).toggleClass('active');
            const $content = $(this).next('.content');
            if ($content.hasClass('open')) {
              $content.removeClass('open').css('max-height', 0);
            } else {
              $content.addClass('open').css('max-height', $content.prop('scrollHeight') + 'px');
            }
          });
        });
      </script>
    </body>
    </html>
    """

    template = Template(template_str)
    html_out = template.render(results=results, consensus=consensus, algo_plots=algo_plots, df=df)
    with open(output_path, "w") as f:
        f.write(html_out)
    
    print(f"\nHTML report saved to anomaly_report.html")

def main(file, HTMLreport, ConsoleReport):
    print(f"Loading dataset from {file}...")
    df = pd.read_csv(file)
    try:
        df_clean = clean_data(df)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Running outlier detection algorithms...")
    results = run_all_algorithms(df_clean)
    agreement = aggregate_outliers(results)

    if ConsoleReport: 
      print_report(df_clean, results, agreement)
    if HTMLreport:
      generate_html_report(df_clean, results, agreement)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnomalyViz: Visual Outlier Detector for CSV data")
    parser.add_argument("--file", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--NoHtmlReport", action='store_false', required=False, help="Flag to disable HTML report generation")
    parser.add_argument("--NoConsoleReport", action='store_false', required=False, help="Flag to disable console text report generation")
    args = parser.parse_args()
    main(args.file, args.NoHtmlReport, args.NoConsoleReport)
