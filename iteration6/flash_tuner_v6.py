import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor
# from sklearn.decomposition import PCA  # tried PCA dimensionality reduction — no improvement

from flash_tuner import ConfigurationSpace, improv_pct, random_search, sig_stars
from flash_tuner_v2 import FeaturePreprocessor
from flash_tuner_v5 import flash_rf_v5


def flash_rf_v6(space, budget, seed=42):
    """v5 + stratified init for large config spaces"""
    rng = np.random.RandomState(seed)
    all_X = space.get_all_configs()
    n_total = len(all_X)

    preprocessor = FeaturePreprocessor()
    preprocessor.fit(space._domains)
    all_X_enc = preprocessor.transform(all_X)
    n_dim = preprocessor.feature_count

    init_size = max(10, int(budget * 0.3))

    # For large spaces random sampling gives poor coverage; use KMeans stratification
    # Threshold 5000: below this, random sampling is sufficient
    if n_total > 5000:
        n_clusters = min(15, init_size)
        km = MiniBatchKMeans(
            n_clusters=n_clusters, random_state=seed,
            n_init=3, batch_size=min(1000, n_total),
        )
        labels = km.fit_predict(all_X_enc)

        init_indices = []
        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            n_pick = max(1, round(init_size * len(cluster_idx) / n_total))
            picks = rng.choice(
                cluster_idx,
                size=min(n_pick, len(cluster_idx)),
                replace=False,
            )
            init_indices.extend(picks.tolist())

        init_set = set(init_indices)
        while len(init_indices) < init_size:
            extra = int(rng.choice(n_total))
            if extra not in init_set:
                init_indices.append(extra)
                init_set.add(extra)
        init_indices_arr = np.array(init_indices[:init_size])
    else:
        # Small space: plain random sampling is sufficient
        init_indices_arr = rng.choice(
            n_total, size=min(init_size, n_total), replace=False
        )

    seen = np.zeros(n_total, dtype=bool)
    seen_idx = []
    y_seen = []
    best = np.inf
    best_idx = -1
    used = 0

    for idx in init_indices_arr:
        if used >= budget:
            break
        if seen[idx]:
            continue
        perf, _ = space.measure(all_X[idx].tolist())
        used += 1
        seen[idx] = True
        seen_idx.append(int(idx))
        y_seen.append(perf)
        if perf < best:
            best = perf
            best_idx = int(idx)

    actual_init = used

    y_phase1 = np.array(y_seen)
    y_min, y_max = y_phase1.min(), y_phase1.max()
    dynamic_range = (y_max - y_min) / (y_min + 1e-10)
    # Apply log transform when performance range is extreme (skewed distribution)
    use_log = dynamic_range > 10.0

    n_trees = 20

    while used < budget:
        X_train = all_X_enc[seen_idx]
        y_raw = np.array(y_seen)

        y_train = np.log1p(y_raw) if use_log else y_raw

        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=min(n_dim, 12),
            min_samples_leaf=max(2, len(y_train) // 20),
            random_state=int(rng.randint(100_000)),
        )
        rf.fit(X_train, y_train)

        tree_preds = np.array([t.predict(all_X_enc) for t in rf.estimators_])

        unmeasured = np.where(~seen)[0]
        if len(unmeasured) == 0:
            break

        remaining = budget - used
        progress = min(1.0, (used - actual_init) / max(1, budget - actual_init))

        batch = min(5, remaining, len(unmeasured))

        n_subsample = max(1, int(n_trees * (0.2 + 0.6 * progress)))

        selected = []
        selected_set = set()
        for _ in range(batch):
            tree_idx = rng.choice(n_trees, size=n_subsample, replace=False)
            sampled_pred = tree_preds[tree_idx].mean(axis=0)

            candidates = np.array(
                [i for i in unmeasured if i not in selected_set]
            )
            if len(candidates) == 0:
                break
            best_candidate = candidates[np.argmin(sampled_pred[candidates])]
            selected.append(int(best_candidate))
            selected_set.add(int(best_candidate))

        for idx in selected:
            if used >= budget:
                break
            if seen[idx]:
                continue
            perf, _ = space.measure(all_X[idx].tolist())
            used += 1
            seen[idx] = True
            seen_idx.append(idx)
            y_seen.append(perf)
            if perf < best:
                best = perf
                best_idx = idx

    return best


def evaluate_v6(datasets_dir, budget=100, n_repeats=30, output_dir="results_v6"):
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    results = {}

    print(f"\n=== v6 evaluation (budget={budget}, {n_repeats} repeats, {len(files)} systems) ===")

    seeds = [i * 42 + 1 for i in range(n_repeats)]

    for fname in files:
        system = fname.replace(".csv", "")
        path = os.path.join(datasets_dir, fname)
        space = ConfigurationSpace(path)
        strat = "kmeans" if space.n_configs > 5000 else "random"
        print(f"[{system}] n={space.n_configs} dim={space.n_features} "
              f"best={space.best_possible:.4f} init={strat}")

        rs_vals = []
        v5_vals = []
        v6_vals = []

        for i, s in enumerate(seeds):
            rs_vals.append(random_search(space, budget, s))
            v5_vals.append(flash_rf_v5(space, budget, s))
            v6_vals.append(flash_rf_v6(space, budget, s))
            if (i + 1) % 10 == 0:
                print(f"  ... {i + 1}/{n_repeats} runs done")

        rs = np.array(rs_vals)
        v5 = np.array(v5_vals)
        v6 = np.array(v6_vals)

        _, p_v5_rs = mannwhitneyu(v5, rs, alternative="less")
        _, p_v6_rs = mannwhitneyu(v6, rs, alternative="less")
        _, p_v6_v5 = mannwhitneyu(v6, v5, alternative="less")

        rs_med = float(np.median(rs))
        entry = {
            "system": system,
            "n_configs": space.n_configs,
            "n_features": space.n_features,
            "best_possible": space.best_possible,
            "rs_median": rs_med,
            "rs_iqr": float(np.percentile(rs, 75) - np.percentile(rs, 25)),
            "v5_median": float(np.median(v5)),
            "v5_iqr": float(np.percentile(v5, 75) - np.percentile(v5, 25)),
            "v6_median": float(np.median(v6)),
            "v6_iqr": float(np.percentile(v6, 75) - np.percentile(v6, 25)),
            "v5_improv_pct": improv_pct(rs_med, np.median(v5)),
            "v6_improv_pct": improv_pct(rs_med, np.median(v6)),
            "p_v5_vs_rs": float(p_v5_rs),
            "p_v6_vs_rs": float(p_v6_rs),
            "p_v6_vs_v5": float(p_v6_v5),
            "v5_hits_optimal": int(np.sum(v5 <= space.best_possible + 1e-9)),
            "v6_hits_optimal": int(np.sum(v6 <= space.best_possible + 1e-9)),
            "rs_raw": rs_vals,
            "v5_raw": v5_vals,
            "v6_raw": v6_vals,
        }

        results[system] = entry

        print(f"  RS    median={rs_med:.4f}  IQR={entry['rs_iqr']:.4f}")
        print(f"  v5    median={entry['v5_median']:.4f}  "
              f"improv={entry['v5_improv_pct']:+.2f}%  "
              f"opt={entry['v5_hits_optimal']}/30  v5>RS {sig_stars(p_v5_rs)}")
        print(f"  v6    median={entry['v6_median']:.4f}  "
              f"improv={entry['v6_improv_pct']:+.2f}%  "
              f"opt={entry['v6_hits_optimal']}/30  v6>v5 {sig_stars(p_v6_v5)}")
        print()

        raw_data = {
            "run": list(range(1, n_repeats + 1)),
            "random_search": rs_vals,
            "flash_v5": v5_vals,
            "flash_v6": v6_vals,
        }
        pd.DataFrame(raw_data).to_csv(
            os.path.join(raw_dir, f"{system}.csv"), index=False)

    _print_summary(results)
    _save_results(results, output_dir)
    return results


def _print_summary(results):
    print("\n--- v5 vs v6 comparison ---")
    print(f"{'System':>12s}  {'|X|':>6s}  {'v5 Med':>10s}  {'v6 Med':>10s}  "
          f"{'v5 %':>7s}  {'v6 %':>7s}  {'v6>v5':>5s}  "
          f"v5hit  v6hit")

    n_win_rs = 0
    n_win_v5 = 0
    tot_v5_opt = 0
    tot_v6_opt = 0

    for name in sorted(results):
        e = results[name]
        print(f"  {name:>12s}  {e['n_configs']:>6d}  {e['v5_median']:10.4f}  "
              f"{e['v6_median']:10.4f}  {e['v5_improv_pct']:+6.1f}%  "
              f"{e['v6_improv_pct']:+6.1f}%  {sig_stars(e['p_v6_vs_v5']):>5s}  "
              f"{e['v5_hits_optimal']:>3d}/30  {e['v6_hits_optimal']:>3d}/30")
        if e["p_v6_vs_rs"] < 0.05:
            n_win_rs += 1
        if e["p_v6_vs_v5"] < 0.05:
            n_win_v5 += 1
        tot_v5_opt += e["v5_hits_optimal"]
        tot_v6_opt += e["v6_hits_optimal"]

    n = len(results)
    print(f"\nv6 beat RS: {n_win_rs}/{n},  v6 beat v5: {n_win_v5}/{n}")
    print(f"optimal hits total:  v5={tot_v5_opt}/240  v6={tot_v6_opt}/240")


def _save_results(results, output_dir):
    with open(os.path.join(output_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    rows = [{k: v for k, v in e.items() if not k.endswith("_raw")}
            for e in results.values()]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)


def main():
    parser = argparse.ArgumentParser(description="FLASH-RF v6 evaluation")
    parser.add_argument("--datasets", default="datasets",
                        help="directory containing CSV dataset files")
    parser.add_argument("--budget", type=int, default=100, help="budget per system")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="results_v6")
    args = parser.parse_args()

    if not os.path.isdir(args.datasets):
        print(f"[ERROR] {args.datasets} not found", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    evaluate_v6(args.datasets, args.budget, args.repeats, args.output)
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
