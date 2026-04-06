# Only difference from v6: rank transform replaces log(1+y) as target encoding
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, rankdata
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestRegressor

from flash_tuner import ConfigurationSpace, improv_pct, random_search, sig_stars
from flash_tuner_v2 import FeaturePreprocessor
from flash_tuner_v6 import flash_rf_v6


def flash_rf_v7(space, budget, seed=42):
    """v7: v6 with rank(y)/n replacing log(1+y) as target transform. Rejected — causes degradation."""
    rng = np.random.RandomState(seed)
    all_X = space.get_all_configs()
    n_total = len(all_X)

    preprocessor = FeaturePreprocessor()
    preprocessor.fit(space._domains)
    all_X_enc = preprocessor.transform(all_X)
    n_dim = preprocessor.feature_count

    init_size = max(10, int(budget * 0.3))

    # Same stratified init as v6
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
        seen = set(init_indices)
        while len(init_indices) < init_size:
            extra = int(rng.choice(n_total))
            if extra not in seen:
                init_indices.append(extra)
                seen.add(extra)
        init_indices_arr = np.array(init_indices[:init_size])
    else:
        init_indices_arr = rng.choice(
            n_total, size=min(init_size, n_total), replace=False
        )

    measured = np.zeros(n_total, dtype=bool)
    m_idx = []
    y_measured = []
    best = np.inf
    used = 0

    for idx in init_indices_arr:
        if used >= budget:
            break
        if measured[idx]:
            continue
        perf, _ = space.measure(all_X[idx].tolist())
        used += 1
        measured[idx] = True
        m_idx.append(int(idx))
        y_measured.append(perf)
        if perf < best:
            best = perf

    actual_init = used

    y_phase1 = np.array(y_measured)
    y_min, y_max = y_phase1.min(), y_phase1.max()
    dynamic_range = (y_max - y_min) / (y_min + 1e-10)
    use_rank = dynamic_range > 10.0

    n_trees = 20

    while used < budget:
        X_train = all_X_enc[m_idx]
        y_raw = np.array(y_measured)

        # Rank transform: scale-invariant, but loses magnitude gap information
        y_train = rankdata(y_raw) / len(y_raw) if use_rank else y_raw

        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=min(n_dim, 12),
            min_samples_leaf=max(2, len(y_train) // 20),
            random_state=int(rng.randint(100_000)),
        )
        rf.fit(X_train, y_train)

        tree_preds = np.array([t.predict(all_X_enc) for t in rf.estimators_])

        unmeasured = np.where(~measured)[0]
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
            if measured[idx]:
                continue
            perf, _ = space.measure(all_X[idx].tolist())
            used += 1
            measured[idx] = True
            m_idx.append(idx)
            y_measured.append(perf)
            if perf < best:
                best = perf

    return best


def evaluate_v7(datasets_dir, budget=100, n_repeats=30, output_dir="results_v7"):
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    results = {}

    print(f"\nv7 eval: rank-transform, budget={budget}, {n_repeats}x{len(files)}")

    seeds = [i * 42 + 1 for i in range(n_repeats)]

    for fname in files:
        system = fname.replace(".csv", "")
        path = os.path.join(datasets_dir, fname)
        space = ConfigurationSpace(path)
        print(f"  {system}: {space.n_configs} configs, {space.n_features} feats, "
              f"optimal={space.best_possible:.4f}")

        rs_vals = []
        v6_vals = []
        v7_vals = []

        for i, s in enumerate(seeds):
            rs_vals.append(random_search(space, budget, s))
            v6_vals.append(flash_rf_v6(space, budget, s))
            v7_vals.append(flash_rf_v7(space, budget, s))
            if (i + 1) % 10 == 0:
                print(f"    {i + 1}/{n_repeats} done")

        rs = np.array(rs_vals)
        v6 = np.array(v6_vals)
        v7 = np.array(v7_vals)

        _, p_v6_rs = mannwhitneyu(v6, rs, alternative="less")
        _, p_v7_rs = mannwhitneyu(v7, rs, alternative="less")
        _, p_v7_v6 = mannwhitneyu(v7, v6, alternative="less")

        rs_med = float(np.median(rs))
        entry = {
            "system": system,
            "n_configs": space.n_configs,
            "n_features": space.n_features,
            "best_possible": space.best_possible,
            "rs_median": rs_med,
            "rs_iqr": float(np.percentile(rs, 75) - np.percentile(rs, 25)),
            "v6_median": float(np.median(v6)),
            "v6_iqr": float(np.percentile(v6, 75) - np.percentile(v6, 25)),
            "v7_median": float(np.median(v7)),
            "v7_iqr": float(np.percentile(v7, 75) - np.percentile(v7, 25)),
            "v6_improv_pct": improv_pct(rs_med, np.median(v6)),
            "v7_improv_pct": improv_pct(rs_med, np.median(v7)),
            "p_v6_vs_rs": float(p_v6_rs),
            "p_v7_vs_rs": float(p_v7_rs),
            "p_v7_vs_v6": float(p_v7_v6),
            "v6_hits_optimal": int(np.sum(v6 <= space.best_possible + 1e-9)),
            "v7_hits_optimal": int(np.sum(v7 <= space.best_possible + 1e-9)),
            "rs_raw": rs_vals,
            "v6_raw": v6_vals,
            "v7_raw": v7_vals,
        }

        results[system] = entry

        diff = entry["v7_hits_optimal"] - entry["v6_hits_optimal"]
        mark = f"({diff:+d})" if diff != 0 else ""

        print(f"  RS  med={rs_med:.4f}")
        print(f"  v6  med={entry['v6_median']:.4f}  {entry['v6_improv_pct']:+.1f}%  "
              f"hit={entry['v6_hits_optimal']}/30")
        print(f"  v7  med={entry['v7_median']:.4f}  {entry['v7_improv_pct']:+.1f}%  "
              f"hit={entry['v7_hits_optimal']}/30 {mark}  "
              f"v7>v6 {sig_stars(p_v7_v6)}")
        print()

        raw = {
            "run": list(range(1, n_repeats + 1)),
            "random_search": rs_vals,
            "flash_v6": v6_vals,
            "flash_v7": v7_vals,
        }
        pd.DataFrame(raw).to_csv(
            os.path.join(raw_dir, f"{system}.csv"), index=False)

    _print_summary(results)
    _save_results(results, output_dir)
    return results


def _print_summary(results):
    print(f"\n{'System':>12s}  {'v6 Med':>10s}  {'v7 Med':>10s}  "
          f"{'v6>RS':>5s}  {'v7>RS':>5s}  {'v7>v6':>5s}  "
          f"v6hit  v7hit  diff")

    tot_v6 = tot_v7 = 0
    degraded = []

    for name in sorted(results):
        e = results[name]
        v6h = e["v6_hits_optimal"]
        v7h = e["v7_hits_optimal"]
        d = v7h - v6h
        tag = f"{d:+d}" if d != 0 else " ="

        if d < 0:
            degraded.append(name)

        print(f"  {name:>12s}  {e['v6_median']:10.4f}  {e['v7_median']:10.4f}  "
              f"{sig_stars(e['p_v6_vs_rs']):>5s}  {sig_stars(e['p_v7_vs_rs']):>5s}  "
              f"{sig_stars(e['p_v7_vs_v6']):>5s}  "
              f"{v6h:>3d}/30  {v7h:>3d}/30  {tag}")
        tot_v6 += v6h
        tot_v7 += v7h

    print(f"\nhits optimal: v6={tot_v6}/240 v7={tot_v7}/240 (diff={tot_v7 - tot_v6:+d})")
    if degraded:
        print(f"Degraded: {', '.join(degraded)}")


def _save_results(results, output_dir):
    rows = [{k: v for k, v in e.items() if not k.endswith("_raw")}
            for e in results.values()]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    with open(os.path.join(output_dir, "results.json"), "w") as fout:
        json.dump(results, fout, indent=2)


def main():
    parser = argparse.ArgumentParser(description="FLASH-RF v7: rank target transform")
    parser.add_argument("--datasets", default="datasets", help="directory containing CSV datasets")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=30, help="number of repetitions")
    parser.add_argument("--output", default="results_v7")
    args = parser.parse_args()

    ds_path = Path(args.datasets)
    if not ds_path.is_dir():
        print(f"datasets directory not found: {ds_path}")
        return

    t0 = time.time()
    evaluate_v7(str(ds_path), args.budget, args.repeats, args.output)
    print(f"\ntotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
