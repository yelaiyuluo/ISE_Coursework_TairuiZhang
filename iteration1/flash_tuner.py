import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor


def sig_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def improv_pct(baseline_med, method_med):
    if baseline_med == 0:
        return 0.0
    return float((baseline_med - method_med) / baseline_med * 100)


class ConfigurationSpace:
    """Configuration space loaded from CSV. Last column is the performance value."""

    def __init__(self, csv_path, minimize=True):
        self.data = pd.read_csv(csv_path)
        self.name = Path(csv_path).stem

        self.config_cols = list(self.data.columns[:-1])
        self.perf_col = self.data.columns[-1]
        self.n_features = len(self.config_cols)
        self.n_configs = len(self.data)
        self._minimize = minimize
        self._sign = 1.0 if minimize else -1.0

        # Build lookup table for fast performance queries
        self._lookup = {}
        for _, row in self.data.iterrows():
            key = tuple(int(row[c]) for c in self.config_cols)
            self._lookup[key] = float(row[self.perf_col]) * self._sign

        self._domains = {
            col: sorted(self.data[col].unique().tolist())
            for col in self.config_cols
        }

        self.best_possible = float(min(self._lookup.values()))

    def measure(self, config):
        key = tuple(int(v) for v in config)
        if key in self._lookup:
            return self._lookup[key], True
        return float('inf'), False

    def random_config(self, rng):
        return [int(rng.choice(self._domains[col])) for col in self.config_cols]

    def get_all_configs(self):
        return self.data[self.config_cols].values.astype(int)

    def __repr__(self):
        direction = "min" if self._minimize else "max"
        return (f"ConfigSpace({self.name}: {self.n_configs} cfgs, "
                f"{self.n_features} feats, {direction}, opt={self.best_possible:.4f})")


def random_search(space, budget, seed=42):
    """Simplest baseline: uniform random sampling."""
    if budget > space.n_configs:
        warnings.warn(
            f"budget({budget}) > total configs ({space.n_configs}), equivalent to full enumeration",
            stacklevel=2,
        )

    rng = np.random.RandomState(seed)
    best = np.inf
    valid_count = 0
    max_iters = budget * 50

    iters = 0
    while valid_count < budget and iters < max_iters:
        config = space.random_config(rng)
        perf, valid = space.measure(config)
        if valid:
            valid_count += 1
        if perf < best:
            best = perf
        iters += 1

    return best


def hill_climbing(space, budget, seed=42):
    """Steepest-ascent hill climbing with random restarts."""
    rng = np.random.RandomState(seed)
    best = np.inf
    used = 0

    while used < budget:
        # restart from random point
        current = space.random_config(rng)
        perf, valid = space.measure(current)
        if valid:
            used += 1
        if perf < best:
            best = perf

        improved = True
        while improved and used < budget:
            improved = False
            best_neighbour = None
            best_neighbour_perf = perf

            for i, col in enumerate(space.config_cols):
                for val in space._domains[col]:
                    if val == current[i]:
                        continue
                    neighbour = list(current)
                    neighbour[i] = val
                    n_perf, n_valid = space.measure(neighbour)
                    if n_valid:
                        used += 1
                    if n_perf < best_neighbour_perf:
                        best_neighbour_perf = n_perf
                        best_neighbour = list(neighbour)
                    if used >= budget:
                        break
                if used >= budget:
                    break

            if best_neighbour is not None and best_neighbour_perf < perf:
                current = best_neighbour
                perf = best_neighbour_perf
                improved = True
                if perf < best:
                    best = perf

    return best


def simulated_annealing(space, budget, seed=42):
    """SA with exponential cooling"""
    rng = np.random.RandomState(seed)
    best = np.inf
    used = 0

    # Sample a few random points to estimate the temperature range
    warmup = min(10, budget // 3)
    perf_vals = []
    current = space.random_config(rng)
    cur_perf, valid = space.measure(current)
    if valid:
        used += 1
        perf_vals.append(cur_perf)
    if cur_perf < best:
        best = cur_perf

    for _ in range(warmup - 1):
        if used >= budget:
            break
        cfg = space.random_config(rng)
        p, v = space.measure(cfg)
        if v:
            used += 1
            perf_vals.append(p)
        if p < cur_perf:
            current, cur_perf = cfg, p
        if p < best:
            best = p

    t0 = (max(perf_vals) - min(perf_vals)) if len(perf_vals) > 1 else 1.0
    t0 = max(t0, 1e-6)
    remaining = budget - used
    # Alpha selection is intentionally coarse here
    alpha = 0.95 if remaining > 50 else 0.90
    temp = t0

    while used < budget:
        i = rng.randint(space.n_features)
        col = space.config_cols[i]
        vals = space._domains[col]
        new_val = int(rng.choice([v for v in vals if v != current[i]]))
        neighbour = list(current)
        neighbour[i] = new_val

        n_perf, n_valid = space.measure(neighbour)
        if n_valid:
            used += 1

        delta = n_perf - cur_perf
        if delta < 0 or (temp > 0 and rng.random() < np.exp(-delta / temp)):
            current, cur_perf = neighbour, n_perf
        if n_perf < best:
            best = n_perf
        temp *= alpha

    return best


# ref: Zhu et al., BestConfig, SoCC 2017
def bestconfig(space, budget, seed=42):
    """Divide-and-Diverge Sampling (DDS) + Recursive Bound Search (RBS)."""
    rng = np.random.RandomState(seed)
    best = np.inf
    best_config = []
    used = 0

    domains = {col: sorted(space._domains[col]) for col in space.config_cols}

    # DDS phase
    dds_budget = budget // 2
    n_rounds = max(1, dds_budget // space.n_features)

    for _ in range(n_rounds):
        if used >= dds_budget:
            break
        config = [int(rng.choice(domains[col])) for col in space.config_cols]

        perf, valid = space.measure(config)
        if valid:
            used += 1
        if perf < best:
            best = perf
            best_config = list(config)

    if not best_config:
        best_config = space.random_config(rng)
        perf, valid = space.measure(best_config)
        if valid:
            used += 1
        if perf < best:
            best = perf

    # RBS: per-dimension binary search
    bounds = {col: (min(space._domains[col]), max(space._domains[col]))
              for col in space.config_cols}
    rbs_rounds = 0

    while used < budget and rbs_rounds < 5:
        rbs_rounds += 1
        mid_config = list(best_config)

        for i, col in enumerate(space.config_cols):
            lo, hi = bounds[col]
            vals = [v for v in space._domains[col] if lo <= v <= hi]
            if len(vals) <= 1:
                continue

            mid = vals[len(vals) // 2]
            lower_vals = [v for v in vals if v <= mid]
            upper_vals = [v for v in vals if v > mid]

            best_half_perf = np.inf
            best_half = None
            for half in [lower_vals, upper_vals]:
                if not half or used >= budget:
                    continue
                candidate = list(mid_config)
                candidate[i] = int(rng.choice(half))
                perf, valid = space.measure(candidate)
                if valid:
                    used += 1
                if perf < best_half_perf:
                    best_half_perf = perf
                    best_half = half
                if perf < best:
                    best = perf
                    best_config = list(candidate)

            if best_half is not None:
                bounds[col] = (min(best_half), max(best_half))
                mid_config[i] = best_config[i]

        for _ in range(min(3, budget - used)):
            if used >= budget:
                break
            config = []
            for i, col in enumerate(space.config_cols):
                lo, hi = bounds[col]
                vals = [v for v in space._domains[col] if lo <= v <= hi]
                config.append(int(rng.choice(vals)) if vals else best_config[i])
            perf, valid = space.measure(config)
            if valid:
                used += 1
            if perf < best:
                best = perf
                best_config = list(config)

    return best


def flash_cart(space, budget, seed=42):
    """v1: RF surrogate model, selects next batch by predicted performance ranking."""
    if budget > space.n_configs:
        warnings.warn(
            f"budget({budget}) > total configs ({space.n_configs}), equivalent to full enumeration",
            stacklevel=2,
        )

    rng = np.random.RandomState(seed)
    all_X = space.get_all_configs()
    n_total = len(all_X)

    init_size = max(10, int(budget * 0.3))
    init_indices = rng.choice(n_total, size=min(init_size, n_total), replace=False)

    measured_set = set()
    X_list, y_list = [], []
    best = np.inf
    used = 0

    for idx in init_indices:
        if used >= budget:
            break
        config = all_X[idx].tolist()
        key = tuple(config)
        if key in measured_set:
            continue
        perf, _ = space.measure(config)
        used += 1
        measured_set.add(key)
        X_list.append(config)
        y_list.append(perf)
        if perf < best:
            best = perf

    while used < budget:
        X_train = np.array(X_list)
        y_train = np.array(y_list)

        rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=min(space.n_features, 10),
            min_samples_leaf=max(1, len(X_train) // 10),
            random_state=int(rng.randint(100_000)),
        )
        rf.fit(X_train, y_train)

        preds = rf.predict(all_X)

        candidates = [
            (preds[i], i) for i in range(n_total)
            if tuple(all_X[i].tolist()) not in measured_set
        ]
        if not candidates:
            break

        candidates.sort(key=lambda c: c[0])

        # 80/20 exploitation/exploration split
        remaining = budget - used
        batch = min(5, remaining, len(candidates))
        n_exploit = max(1, int(batch * 0.8))
        n_explore = batch - n_exploit

        selected = [candidates[i][1] for i in range(min(n_exploit, len(candidates)))]

        if n_explore > 0 and len(candidates) > n_exploit:
            pool = [c[1] for c in candidates[n_exploit:]]
            picks = rng.choice(pool, size=min(n_explore, len(pool)), replace=False)
            selected.extend(picks.tolist())

        for idx in selected:
            if used >= budget:
                break
            config = all_X[idx].tolist()
            key = tuple(config)
            if key in measured_set:
                continue
            perf, _ = space.measure(config)
            used += 1
            measured_set.add(key)
            X_list.append(config)
            y_list.append(perf)
            if perf < best:
                best = perf

    return best


def evaluate(datasets_dir, budget=100, n_repeats=30, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    results = {}

    print(f"\n=== FLASH-RF Configuration Tuning ===")
    print(f"budget={budget}, repeats={n_repeats}, datasets={len(files)}\n")

    for fname in files:
        system = fname.replace(".csv", "")
        path = os.path.join(datasets_dir, fname)
        space = ConfigurationSpace(path)
        print(f"[{system}] {space.n_configs} configs, {space.n_features} features "
              f"(optimal={space.best_possible:.4f})")

        rs_vals, fl_vals = [], []
        seeds = [i * 42 + 1 for i in range(n_repeats)]

        for seed in seeds:
            rs_vals.append(random_search(space, budget, seed))
            fl_vals.append(flash_cart(space, budget, seed))
            # print(f"  DEBUG {system}: rs={rs_vals[-1]:.4f}")

        rs = np.array(rs_vals)
        fl = np.array(fl_vals)

        stat, pval = mannwhitneyu(fl, rs, alternative="less")

        rec = {
            "system": system,
            "n_configs": space.n_configs,
            "n_features": space.n_features,
            "best_possible": space.best_possible,
            "rs_median": float(np.median(rs)),
            "rs_q1": float(np.percentile(rs, 25)),
            "rs_q3": float(np.percentile(rs, 75)),
            "rs_iqr": float(np.percentile(rs, 75) - np.percentile(rs, 25)),
            "flash_median": float(np.median(fl)),
            "flash_q1": float(np.percentile(fl, 25)),
            "flash_q3": float(np.percentile(fl, 75)),
            "flash_iqr": float(np.percentile(fl, 75) - np.percentile(fl, 25)),
            "u_statistic": float(stat),
            "p_value": float(pval),
            "significant": pval < 0.05,
            "improvement_pct": improv_pct(np.median(rs), np.median(fl)),
            "rs_raw": rs_vals,
            "flash_raw": fl_vals,
        }
        results[system] = rec

        print(f"  RS    median={rec['rs_median']:.4f}  (IQR {rec['rs_iqr']:.4f})")
        print(f"  FLASH median={rec['flash_median']:.4f}  (IQR {rec['flash_iqr']:.4f})")
        print(f"  improvement={rec['improvement_pct']:+.1f}%  "
              f"p={pval:.6f} {sig_stars(pval)}\n")

        raw_df = pd.DataFrame({
            "run": list(range(1, n_repeats + 1)),
            "random_search": rs_vals,
            "flash_cart": fl_vals,
        })
        raw_df.to_csv(os.path.join(raw_dir, f"{system}_results.csv"), index=False)

    # Build summary table
    rows = []
    for name, entry in results.items():
        rows.append({
            "System": name,
            "Configs": entry["n_configs"],
            "Features": entry["n_features"],
            "Optimal": entry["best_possible"],
            "RS Median": entry["rs_median"],
            "RS IQR": entry["rs_iqr"],
            "FLASH Median": entry["flash_median"],
            "FLASH IQR": entry["flash_iqr"],
            "Improv%": entry["improvement_pct"],
            "p-value": entry["p_value"],
            "Sig": "Yes" if entry["significant"] else "No",
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    serialisable = {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}
    with open(os.path.join(output_dir, "results.json"), "w") as fh:
        json.dump(serialisable, fh, indent=2)

    print("=== SUMMARY ===")
    print(f"{'System':>12} | {'RS Med':>10} | {'FLASH Med':>10} | {'Change':>7} | {'p-value':>10} | Sig")
    print("-" * 72)
    for name in sorted(results.keys()):
        e = results[name]
        print(f"{name:>12} | {e['rs_median']:10.4f} | {e['flash_median']:10.4f} | "
              f"{e['improvement_pct']:+6.1f}% | {e['p_value']:10.6f} | {sig_stars(e['p_value'])}")
    print(f"\nResults saved to {output_dir}/\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="FLASH-RF configuration tuning")
    parser.add_argument("--datasets", default="datasets",
                        help="directory containing CSV dataset files")
    parser.add_argument("--budget", type=int, default=100,
                        help="measurement budget per run (default: 100)")
    parser.add_argument("--repeats", type=int, default=30,
                        help="number of independent repetitions")
    parser.add_argument("--output", default="results",
                        help="output directory (default: results)")
    args = parser.parse_args()

    if not os.path.isdir(args.datasets):
        print(f"Error: datasets directory '{args.datasets}' not found.", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    evaluate(args.datasets, args.budget, args.repeats, args.output)
    print(f"Total elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
