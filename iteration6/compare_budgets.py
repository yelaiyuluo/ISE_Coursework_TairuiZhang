# Compare v5 performance between R=30 and R=100
import json
import os
import time
import numpy as np

from flash_tuner import ConfigurationSpace, improv_pct, random_search
from flash_tuner_v5 import flash_rf_v5


def run_comparison(datasets_dir="datasets", n_repeats=30):
    """Compare v5 performance across different budgets."""
    budgets = [30, 100]
    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))

    all_results = {}
    seeds = list(range(n_repeats))

    for budget in budgets:
        print(f"\nRunning R={budget} evaluation")

        results = {}
        for fname in files:
            system = fname.replace(".csv", "")
            path = os.path.join(datasets_dir, fname)
            space = ConfigurationSpace(path)

            rs_vals = []
            v5_vals = []
            for i, seed in enumerate(seeds):
                rs_vals.append(random_search(space, budget, seed))
                v5_vals.append(flash_rf_v5(space, budget, seed))
                if (i+1) % 10 == 0:
                    print(f"  [{system}] {i+1}/{n_repeats} done")

            rs = np.array(rs_vals)
            v5 = np.array(v5_vals)

            rs_med = float(np.median(rs))
            v5_med = float(np.median(v5))
            improv = improv_pct(rs_med, v5_med)
            hits = int(np.sum(v5 <= space.best_possible + 1e-9))

            results[system] = {
                "rs_median": rs_med,
                "v5_median": v5_med,
                "improv_pct": improv,
                "hits_optimal": hits,
                "rs_iqr": float(np.percentile(rs, 75) - np.percentile(rs, 25)),
                "v5_iqr": float(np.percentile(v5, 75) - np.percentile(v5, 25)),
                "rs_raw": rs_vals,
                "v5_raw": v5_vals,
            }

            print(f"  [{system}] RS={rs_med:.4f} V5={v5_med:.4f} ({improv:+.2f}%) hits={hits}/30")

        all_results[budget] = results

    print(f"\n--- R=30 vs R=100 ---")
    for system in sorted(all_results[30].keys()):
        r30 = all_results[30][system]
        r100 = all_results[100][system]

        v5_diff = r30['v5_median'] - r100['v5_median']
        rs_diff = r30['rs_median'] - r100['rs_median']
        hits_diff = r30['hits_optimal'] - r100['hits_optimal']

        if v5_diff > 0.001:
            status = "DEGRADED"
        elif v5_diff < -0.001:
            status = "same/better"
        else:
            status = "stable"

        print(f"  {system}: "
              f"R30 V5={r30['v5_median']:.4f}({r30['improv_pct']:+.2f}%)  "
              f"R100 V5={r100['v5_median']:.4f}({r100['improv_pct']:+.2f}%)  "
              f"v5_diff={v5_diff:+.4f} rs_diff={rs_diff:+.4f} "
              f"hits={hits_diff:+d}  {status}")

    return all_results


if __name__ == "__main__":
    t0 = time.time()
    results = run_comparison()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
