import numpy as np
import pandas as pd
from collections import defaultdict

def printReport(results, agreement: defaultdict[int, int]) -> None:
    print("\n===== Outlier Detection Report =====")
    
    if isinstance(results, dict) and 'results' in results:
        algo_results = results['results']
        timings = results.get('timings', {})
        statistics = results.get('statistics', {})
        metadata = results.get('metadata', {})
        has_timing_info = True
    else:
        algo_results = results
        timings = {}
        statistics = {}
        metadata = {}
        has_timing_info = False
    
    print("\n[Algorithm Results]")
    total_outliers = 0
    
    for algo, mask in algo_results.items():
        outlier_count = np.sum(mask)
        total_outliers += outlier_count
        result_line = f"{algo.upper():<12} | {outlier_count:>4} outliers"
        
        if has_timing_info and algo in timings and timings[algo] is not None:
            result_line += f" | {timings[algo]:>6.3f}s"
            if algo in statistics and 'actual_contamination' in statistics[algo]:
                actual_cont = statistics[algo]['actual_contamination']
                result_line += f" | {actual_cont:>5.1%} actual"
        
        print(result_line)
    
    if has_timing_info and timings:
        print(f"\n[Performance Summary]")
        valid_timings = {k: v for k, v in timings.items() if v is not None}
        
        if valid_timings:
            total_time = metadata.get('timing_stats', {}).get('total_time')
            fastest_algo = min(valid_timings.keys(), key=lambda x: valid_timings[x])
            slowest_algo = max(valid_timings.keys(), key=lambda x: valid_timings[x])
            
            if total_time:
                print(f"Total execution time: {total_time:.3f}s")
            print(f"Fastest algorithm: {fastest_algo.upper()} ({valid_timings[fastest_algo]:.3f}s)")
            print(f"Slowest algorithm: {slowest_algo.upper()} ({valid_timings[slowest_algo]:.3f}s)")
            
            failed_algos = metadata.get('algorithms_failed', [])
            if failed_algos:
                print(f"Failed algorithms: {', '.join([alg.upper() for alg in failed_algos])}")
    
    numAlgos = len(algo_results)
    conThreshold = max(1, int(np.ceil(numAlgos * 0.5)))
    
    print(f"\n[Consensus Settings] Using {numAlgos} algorithms, requiring {conThreshold} or more to agree (≥{(conThreshold/numAlgos)*100:.0f}%)")
    
    consensus = {idx: count for idx, count in agreement.items() if count >= conThreshold}
   
    if consensus:
        print(f"\n[Consensus Results] {len(consensus)} rows flagged by {conThreshold}+ algorithms")
        agreement_dist = defaultdict(int)
        for count in consensus.values():
            agreement_dist[count] += 1
        
        print("Agreement distribution:")
        for agree_count in sorted(agreement_dist.keys(), reverse=True):
            row_count = agreement_dist[agree_count]
            percentage = (agree_count / numAlgos) * 100
            print(f"  {row_count} rows flagged by {agree_count}/{numAlgos} algorithms ({percentage:.0f}%)")
        
        if len(consensus) <= 20:
            print("\nSpecific consensus outliers:")
            for idx, count in sorted(consensus.items(), key=lambda x: -x[1])[:20]:
                percentage = (count / numAlgos) * 100
                print(f"  Row {idx} flagged by {count}/{numAlgos} algorithms ({percentage:.0f}%)")
        elif len(consensus) <= 100:
            print(f"\nTop 10 consensus outliers (showing 10 of {len(consensus)}):")
            for idx, count in sorted(consensus.items(), key=lambda x: -x[1])[:10]:
                percentage = (count / numAlgos) * 100
                print(f"  Row {idx} flagged by {count}/{numAlgos} algorithms ({percentage:.0f}%)")
        else:
            print(f"\nTop 5 consensus outliers (showing 5 of {len(consensus)}):")
            for idx, count in sorted(consensus.items(), key=lambda x: -x[1])[:5]:
                percentage = (count / numAlgos) * 100
                print(f"  Row {idx} flagged by {count}/{numAlgos} algorithms ({percentage:.0f}%)")
    else:
        print("\n[Consensus Results] No consensus outliers found with current threshold.")
       
        if agreement:
            maxAgreement = max(agreement.values())
            maxAgreePoints = [idx for idx, count in agreement.items() if count == maxAgreement]
            print(f"Highest agreement: {maxAgreement} algorithms agreed on {len(maxAgreePoints)} point(s)")
            
            if maxAgreement > 1:
                show_count = min(5, len(maxAgreePoints))
                if show_count < len(maxAgreePoints):
                    print(f"Sample of highest agreement points (showing {show_count} of {len(maxAgreePoints)}):")
                else:
                    print("Highest agreement points:")
                    
                for idx in sorted(maxAgreePoints)[:show_count]:
                    percentage = (maxAgreement / numAlgos) * 100
                    print(f"  Row {idx} flagged by {maxAgreement}/{numAlgos} algorithms ({percentage:.0f}%)")
    
    print("=" * 37)
