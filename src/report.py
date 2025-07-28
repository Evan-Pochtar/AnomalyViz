import numpy as np

def printReport(results, agreement):
    print("\n===== Outlier Detection Report =====")
    
    for algo, mask in results.items():
        print(f"\n[Algorithm: {algo.upper()}] Detected {np.sum(mask)} outliers.")
    
    numAlgos = len(results)
    conThreshold = max(1, int(np.ceil(numAlgos * 0.5)))
    print(f"\n[Consensus Settings] Using {numAlgos} algorithms, requiring {conThreshold} or more to agree (≥{(conThreshold/numAlgos)*100:.0f}%)")
    print(f"\n[Consensus] Rows flagged as outliers by {conThreshold}+ algorithms:")
    consensus = {idx: count for idx, count in agreement.items() if count >= conThreshold}
    
    if consensus:
        for idx, count in sorted(consensus.items(), key=lambda x: -x[1]):
            percentage = (count / numAlgos) * 100
            print(f"Row {idx} flagged by {count}/{numAlgos} algorithms ({percentage:.0f}%)")
        print(f"\nTotal consensus outliers: {len(consensus)}")
    else:
        print("No consensus outliers found with current threshold.")
        
        if agreement:
            maxAgreement = max(agreement.values())
            maxAgreePoints = [idx for idx, count in agreement.items() if count == maxAgreement]
            print(f"Highest agreement: {maxAgreement} algorithms agreed on {len(maxAgreePoints)} point(s)")
            if maxAgreement > 1:
                for idx in sorted(maxAgreePoints):
                    percentage = (maxAgreement / numAlgos) * 100
                    print(f"  Row {idx} flagged by {maxAgreement}/{numAlgos} algorithms ({percentage:.0f}%)")
