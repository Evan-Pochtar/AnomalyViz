import os
import numpy as np
from jinja2 import Template

from src.visualization import PCAvisualization

def generateHTML(df, results, agreement, output_path="report/anomaly_report.html"):
    consensus = {idx: count for idx, count in agreement.items() if count > 4}
    os.makedirs('plots', exist_ok=True)
    consensus_mask = np.array([agreement.get(i, 0) > 4 for i in range(len(df))])

    PCAvisualization(df, consensus_mask, "Consensus Outliers (2D)", dim=2, save_path='plots/pca_consensus_2d.png')
    if df.shape[1] >=3:
        PCAvisualization(df, consensus_mask, "Consensus Outliers (3D)", dim=3, save_path='plots/pca_consensus_3d.png')

    algo_plots = {}
    for algo, mask in results.items():
        PCAvisualization(df, mask, f"{algo.upper()} Outliers (2D)", dim=2, save_path=f'plots/pca_{algo}_2d.png')
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
