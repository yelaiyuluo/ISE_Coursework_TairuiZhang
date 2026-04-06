# Full run takes approximately 10 minutes
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from flash_tuner import (
    ConfigurationSpace,
    bestconfig,
    hill_climbing,
    improv_pct,
    random_search,
    sig_stars,
    simulated_annealing,
)
from flash_tuner_v6 import flash_rf_v6

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

if HAS_OPTUNA:
    from flash_tuner_v2 import tpe_search


METHODS = {
    "RS":         random_search,
    "HC":         hill_climbing,
    "SA":         simulated_annealing,
    "BestConfig": bestconfig,
}
if HAS_OPTUNA:
    METHODS["TPE"] = tpe_search
METHODS["FLASH-RF_v6"] = flash_rf_v6

BUDGETS = [30, 100, 200]
N_REPEATS = 30
SEEDS = [i * 42 + 1 for i in range(N_REPEATS)]


def run_all(datasets_dir, output_dir):
    """Extended experiments: all methods × all systems × multiple budgets."""
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    spaces = {}
    for fname in files:
        name = fname.replace(".csv", "")
        spaces[name] = ConfigurationSpace(os.path.join(datasets_dir, fname))

    results = {}

    for budget in BUDGETS:
        budget_key = f"R={budget}"
        budget_dir = os.path.join(output_dir, f"budget_{budget}")
        os.makedirs(budget_dir, exist_ok=True)
        raw_dir = os.path.join(budget_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        print(f"\n--- Budget R={budget}  {len(METHODS)} methods x "
              f"{len(spaces)} systems x {N_REPEATS} repeats ---")

        res = {}

        for sys_name, space in spaces.items():
            print(f"\n  [{sys_name}] {space.n_configs} configs, "
                  f"{space.n_features} feats (opt={space.best_possible:.4f})")

            method_vals = {m: [] for m in METHODS}

            for i, seed in enumerate(SEEDS):
                for mname, mfunc in METHODS.items():
                    val = mfunc(space, budget, seed)
                    method_vals[mname].append(val)
                if (i + 1) % 10 == 0:
                    print(f"    {i+1}/{N_REPEATS}")

            entry = {
                "system": sys_name,
                "budget": budget,
                "n_configs": space.n_configs,
                "best_possible": space.best_possible,
            }

            raw_data = {"run": list(range(1, N_REPEATS + 1))}
            v6_arr = np.array(method_vals["FLASH-RF_v6"])

            for mname in METHODS:
                arr = np.array(method_vals[mname])
                med = float(np.median(arr))
                iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
                rs_med = float(np.median(np.array(method_vals["RS"])))
                improv = improv_pct(rs_med, med)
                hits = int(np.sum(arr <= space.best_possible + 1e-9))

                entry[f"{mname}_median"] = med
                entry[f"{mname}_iqr"] = iqr
                entry[f"{mname}_improv_pct"] = improv
                entry[f"{mname}_hits"] = hits

                if mname != "FLASH-RF_v6":
                    _, p = mannwhitneyu(v6_arr, arr, alternative="less")
                    entry[f"v6_vs_{mname}_p"] = float(p)

                raw_data[mname] = method_vals[mname]

            res[sys_name] = entry

            for mname in METHODS:
                m = entry[f"{mname}_median"]
                h = entry[f"{mname}_hits"]
                if mname == "FLASH-RF_v6":
                    print(f"    >> {mname}  med={m:.4f}  hits={h}/30")
                else:
                    p = entry[f"v6_vs_{mname}_p"]
                    print(f"    {mname}  med={m:.4f}  hits={h}/30  v6>{mname} {sig_stars(p)}")

            pd.DataFrame(raw_data).to_csv(
                os.path.join(raw_dir, f"{sys_name}.csv"), index=False)

        summary_rows = [
            {k: v for k, v in e.items() if not isinstance(v, list)}
            for sname, e in res.items()
        ]
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(budget_dir, "summary.csv"), index=False)

        results[budget_key] = res

    with open(os.path.join(output_dir, "results_extended.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _print_final_summary(results)

    return results


def _print_final_summary(results):
    print("\n\n=== FINAL SUMMARY ===")

    for budget_key, res in results.items():
        print(f"\n{budget_key}:")

        total_hits = {m: 0 for m in METHODS}
        v6_wins = {m: 0 for m in METHODS if m != "FLASH-RF_v6"}

        for sname in sorted(res.keys()):
            e = res[sname]
            line = f"  {sname:>12s}"
            for mname in METHODS:
                med = e[f"{mname}_median"]
                hits = e[f"{mname}_hits"]
                total_hits[mname] += hits
                if mname == "FLASH-RF_v6":
                    line += f"  {mname}={med:.2f}**"
                else:
                    p = e[f"v6_vs_{mname}_p"]
                    if p < 0.05:
                        v6_wins[mname] += 1
                    line += f"  {mname}={med:.2f}{sig_stars(p)}"
            print(line)

        n_sys = len(res)
        print(f"  hits: " + "  ".join(
            f"{m}={total_hits[m]}/{n_sys*N_REPEATS}" for m in METHODS))
        print(f"  v6 sig. wins: " + "  ".join(
            f"vs {m}={v6_wins[m]}/{n_sys}" for m in v6_wins))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--output", default="results_extended")
    args = ap.parse_args()

    if not os.path.isdir(args.datasets):
        print(f"Error: {args.datasets} not found", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    run_all(args.datasets, args.output)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
