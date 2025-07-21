import numpy as np

def printReport(results, agreement):
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
