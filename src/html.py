import os
import numpy as np
from jinja2 import Template
import pandas as pd
from collections import defaultdict

from src.visualization import PCAvisualization

def generateHTML(df: pd.DataFrame, results: dict[str, pd.Series], agreement: defaultdict[int, int], outputPath: str = "report/AnomalyReport.html", consensusThreshold: int = None) -> None:
    numAlgos = len(results)
    if consensusThreshold is not None:
        conThreshold = consensusThreshold
    else:
        conThreshold = max(1, int(np.ceil(numAlgos * 0.5)))
    
    consensus = {idx: count for idx, count in agreement.items() if count >= conThreshold}
    
    os.makedirs('plots', exist_ok=True)
    conMask = np.array([agreement.get(i, 0) >= conThreshold for i in range(len(df))])

    PCAvisualization(df, conMask, "Consensus Outliers (2D)", dim=2, savePath='plots/pca_consensus_2d.png')
    if df.shape[1] >=3:
        PCAvisualization(df, conMask, "Consensus Outliers (3D)", dim=3, savePath='plots/pca_consensus_3d.png')

    outlierPlots = {}
    for algo, mask in results.items():
        PCAvisualization(df, mask, f"{algo.upper()} Outliers (2D)", dim=2, savePath=f'plots/pca_{algo}_2d.png')
        outlierPlots[algo] = f'../plots/pca_{algo}_2d.png'

    templateCode = """
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
          --primary: #000000;
          --primary-light: #4C4E52;
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
        .consensus-info {
          text-align: center;
          background: #e3f2fd;
          padding: 15px;
          border-radius: 8px;
          margin: 20px 0;
          border-left: 4px solid var(--primary);
        }
        .consensus-info strong {
          color: var(--primary);
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
          margin-bottom: 10px;
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
        .plot-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: 20px;
          margin: 20px 0;
        }
        .plot-grid.single-plot {
          grid-template-columns: 1fr;
          max-width: 600px;
          margin: 20px auto;
        }
        .plot-item {
          text-align: center;
        }
        .plot-item h4 {
          margin: 0 0 15px 0;
          font-size: 1.1em;
        }
        img {
          max-width: 100%;
          height: auto;
          display: block;
          margin: 0 auto;
          border-radius: 4px;
          box-shadow: 0 1px 6px rgba(0,0,0,0.1);
          transition: transform 0.2s ease;
        }
        img:hover {
          transform: scale(1.02);
        }
        footer {
          text-align: center;
          font-size: 0.9em;
          color: var(--muted);
          margin: 40px 0 20px;
        }
        @media (max-width: 768px) {
          .plot-grid {
            grid-template-columns: 1fr;
          }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>AnomalyViz: Outlier Detection Report</h1>
        
        <div class="consensus-info">
          <strong>Consensus Settings:</strong> Using {{ numAlgos }} algorithms, requiring {{ conThreshold }}+ to agree 
          (>={{ "%.0f"|format((conThreshold/numAlgos)*100) }}% consensus)
        </div>

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

        <button class="collapsible">Consensus Outliers ({{ consensus|length }} found)</button>
        <div class="content">
          {% if consensus %}
          <table id="consensusTable" class="display">
            <thead>
              <tr><th>Row Index</th><th># Algorithms</th><th>Consensus %</th></tr>
            </thead>
            <tbody>
              {% for idx, count in consensus.items() %}
                <tr>
                  <td>{{ idx }}</td>
                  <td>{{ count }}/{{ numAlgos }}</td>
                  <td>{{ "%.0f"|format((count/numAlgos)*100) }}%</td>
                </tr>
              {% endfor %}
            </tbody>
          </table>
          {% else %}
            <p>No consensus outliers found with current threshold ({{ conThreshold }}+ algorithms).</p>
            {% if maxAgreement > 1 %}
            <p><strong>Highest agreement:</strong> {{ maxAgreement }} algorithms agreed on {{ maxAgreeCount }} point(s)</p>
            {% endif %}
          {% endif %}
        </div>

        <button class="collapsible">Visualizations</button>
        <div class="content">
          <h3>Consensus Outliers</h3>
          <div class="plot-grid{% if df.shape[1] < 3 %} single-plot{% endif %}">
            <div class="plot-item">
              <h4>2D Visualization</h4>
              <img src="{{ '../plots/pca_consensus_2d.png' }}" alt="Consensus PCA 2D" />
            </div>
            {% if df.shape[1] >= 3 %}
            <div class="plot-item">
              <h4>3D Visualization</h4>
              <img src="{{ '../plots/pca_consensus_3d.png' }}" alt="Consensus PCA 3D" />
            </div>
            {% endif %}
          </div>

          <h3>Individual Algorithm Results</h3>
          <div class="plot-grid{% if outlierPlots|length % 2 == 1 %} odd-count{% endif %}">
            {% for algo, plot in outlierPlots.items() %}
            <div class="plot-item">
              <h4>{{ algo.upper() }}</h4>
              <img src="{{ plot }}" alt="{{ algo }} PCA Plot" />
            </div>
            {% endfor %}
          </div>
          {% if outlierPlots|length % 2 == 1 %}
          <style>
            .plot-grid.odd-count > .plot-item:last-child {
              grid-column: 1 / -1;
              max-width: 600px;
              margin: 0 auto;
            }
          </style>
          {% endif %}
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
