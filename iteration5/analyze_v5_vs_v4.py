# Quick comparison: which systems degraded in v5 relative to v4
import json
import pandas as pd
from pathlib import Path


def load_results():
    base = Path(__file__).resolve().parent
    v5_path = base / "results_v5" / "results.json"
    v4_path = base / "results_v4" / "results.json"

    with open(v5_path, 'r') as f:
        v5_data = json.load(f)

    with open(v4_path, 'r') as f:
        v4_data = json.load(f)

    return v5_data, v4_data


def analyze_degradation(v5_data, v4_data):
    """Compare v5 vs v4 system-by-system."""
    print("V5 vs V4 comparison\n")

    worse = []
    better = []
    same = []

    for system in v5_data.keys():
        v5 = v5_data[system]
        v4 = v4_data[system]
        sname = v5['system']

        improv_diff = v5['v5_improv_pct'] - v4['v4_improv_pct']
        median_diff = v5['v5_median'] - v4['v4_median']
        hits_diff = v5['v5_hits_optimal'] - v4['v4_hits_optimal']

        is_worse = False
        is_better = False
        reasons = []

        if improv_diff < -0.01:
            is_worse = True
            reasons.append(f"improvement dropped {improv_diff:.2f}%")
        elif improv_diff > 0.01:
            is_better = True
            reasons.append(f"improvement gained {improv_diff:.2f}%")

        if median_diff > 0.001:
            is_worse = True
            reasons.append(f"median worsened by {median_diff:.4f}")
        elif median_diff < -0.001:
            is_better = True
            reasons.append(f"median improved by {abs(median_diff):.4f}")

        if hits_diff < 0:
            is_worse = True
            reasons.append(f"optimal hits decreased by {abs(hits_diff)}")
        elif hits_diff > 0:
            is_better = True
            reasons.append(f"optimal hits increased by {hits_diff}")

        info = {
            'system': sname,
            'v4_improv_pct': v4['v4_improv_pct'],
            'v5_improv_pct': v5['v5_improv_pct'],
            'improv_diff': improv_diff,
            'v4_median': v4['v4_median'],
            'v5_median': v5['v5_median'],
            'median_diff': median_diff,
            'v4_hits': v4['v4_hits_optimal'],
            'v5_hits': v5['v5_hits_optimal'],
            'hits_diff': hits_diff,
            'reasons': reasons,
            'p_v4_vs_rs': v4['p_v4_vs_rs'],
            'p_v5_vs_rs': v5['p_v5_vs_rs'],
            'p_v5_vs_v4': v5['p_v5_vs_v4'],
        }

        if is_worse:
            worse.append(info)
            print(f"{sname}: DEGRADED")
            print(f"  v4 improv={v4['v4_improv_pct']:.2f}%  v5 improv={v5['v5_improv_pct']:.2f}%  (diff {improv_diff:+.2f}%)")
            print(f"  median: {v4['v4_median']:.4f} -> {v5['v5_median']:.4f}")
            print(f"  hits: {v4['v4_hits_optimal']} -> {v5['v5_hits_optimal']}")
            print(f"  reasons: {', '.join(reasons)}")
        elif is_better:
            better.append(info)
            print(f"{sname}: improved ({', '.join(reasons)})")
        else:
            same.append(info)

    if same:
        print(f"unchanged: {', '.join(c['system'] for c in same)}")

    return worse, better, same


def print_analysis(worse, better, same):
    """Print summary and detailed comparison table."""
    n_total = len(worse) + len(better) + len(same)
    print(f"\nDegraded: {len(worse)}  Improved: {len(better)}  "
          f"Neutral: {len(same)}  Total: {n_total}")

    all_cases = worse + better + same
    df = pd.DataFrame(all_cases)
    cols = ['system', 'v4_improv_pct', 'v5_improv_pct', 'v4_hits', 'v5_hits', 'p_v5_vs_v4']
    print()
    print(df[cols].to_string(index=False, float_format='%.4f'))


def main():
    v5_data, v4_data = load_results()

    worse, better, same = analyze_degradation(v5_data, v4_data)
    print_analysis(worse, better, same)

    print("\nSignificance (V5 vs V4):")
    for system in v5_data.keys():
        p = v5_data[system]['p_v5_vs_v4']
        sig = "sig" if p < 0.05 else "n.s."
        print(f"  {system}: p={p:.4f} ({sig})")

    if worse:
        print(f"\nDegraded systems:")
        for c in worse:
            print(f"  {c['system']}: {', '.join(c['reasons'])}")


if __name__ == "__main__":
    main()
