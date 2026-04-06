# v4 is already strong; this adds a conditional log transform on the target
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor

from flash_tuner import ConfigurationSpace, improv_pct, random_search, sig_stars
from flash_tuner_v2 import FeaturePreprocessor
from flash_tuner_v4 import flash_rf_v4


def flash_rf_v5(space, budget, seed=42):
    """v5: Thompson Sampling + conditional log(1+y) target transform."""
    rng = np.random.RandomState(seed)
    all_X = space.get_all_configs()
    n_total = len(all_X)

    preprocessor = FeaturePreprocessor()
    preprocessor.fit(space._domains)
    all_X_enc = preprocessor.transform(all_X)
    n_dim = preprocessor.feature_count

    init_size = max(10, int(budget * 0.3))
    init_indices = rng.choice(n_total, size=min(init_size, n_total), replace=False)

    done = np.zeros(n_total, dtype=bool)
    done_idx = []
    y_seen = []
    best = np.inf
    best_idx = -1
    count = 0

    for idx in init_indices:
        if count >= budget:
            break
        if done[idx]:
            continue
        perf, _ = space.measure(all_X[idx].tolist())
        count += 1
        done[idx] = True
        done_idx.append(int(idx))
        y_seen.append(perf)
        if perf < best:
            best = perf
            best_idx = int(idx)

    actual_init = count

    y_phase1 = np.array(y_seen)
    y_min, y_max = y_phase1.min(), y_phase1.max()
    dynamic_range = (y_max - y_min) / (y_min + 1e-10)
    # log transform if range is large
    use_log = dynamic_range > 10.0

    n_trees = 20

    while count < budget:
        X_train = all_X_enc[done_idx]
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

        unmeasured = np.where(~done)[0]
        if len(unmeasured) == 0:
            break

        remaining = budget - count
        progress = min(1.0, (count - actual_init) / max(1, budget - actual_init))

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
            if count >= budget:
                break
            if done[idx]:
                continue
            perf, _ = space.measure(all_X[idx].tolist())
            count += 1
            done[idx] = True
            done_idx.append(idx)
            y_seen.append(perf)
            if perf < best:
                best = perf
                best_idx = idx

    return best


def evaluate_v5(datasets_dir, budget=100, n_repeats=30, output_dir="results_v5"):
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    results = {}

    print(f"\nv5 eval: v4 + conditional log transform")
    print(f"budget={budget}, repeats={n_repeats}, systems={len(files)}\n")

    seeds = [i * 42 + 1 for i in range(n_repeats)]

    for fname in files:
        system = fname.replace(".csv", "")
        path = os.path.join(datasets_dir, fname)
        space = ConfigurationSpace(path)
        print(f"[{system}] {space.n_configs} configs, {space.n_features} feats,"
              f" best={space.best_possible:.4f}")

        rs_vals, v4_vals, v5_vals = [], [], []

        for i, s in enumerate(seeds):
            rs_vals.append(random_search(space, budget, s))
            v4_vals.append(flash_rf_v4(space, budget, s))
            v5_vals.append(flash_rf_v5(space, budget, s))
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{n_repeats} done")

        rs = np.array(rs_vals)
        v4 = np.array(v4_vals)
        v5 = np.array(v5_vals)

        _, p_v4_rs = mannwhitneyu(v4, rs, alternative="less")
        _, p_v5_rs = mannwhitneyu(v5, rs, alternative="less")
        _, p_v5_v4 = mannwhitneyu(v5, v4, alternative="less")

        rs_med = float(np.median(rs))
        entry = {
            "system": system,
            "n_configs": space.n_configs,
            "n_features": space.n_features,
            "best_possible": space.best_possible,
            "rs_median": rs_med,
            "rs_iqr": float(np.percentile(rs, 75) - np.percentile(rs, 25)),
            "v4_median": float(np.median(v4)),
            "v4_iqr": float(np.percentile(v4, 75) - np.percentile(v4, 25)),
            "v5_median": float(np.median(v5)),
            "v5_iqr": float(np.percentile(v5, 75) - np.percentile(v5, 25)),
            "v4_improv_pct": improv_pct(rs_med, np.median(v4)),
            "v5_improv_pct": improv_pct(rs_med, np.median(v5)),
            "p_v4_vs_rs": float(p_v4_rs),
            "p_v5_vs_rs": float(p_v5_rs),
            "p_v5_vs_v4": float(p_v5_v4),
            "v4_hits_optimal": int(np.sum(v4 <= space.best_possible + 1e-9)),
            "v5_hits_optimal": int(np.sum(v5 <= space.best_possible + 1e-9)),
            "rs_raw": rs_vals,
            "v4_raw": v4_vals,
            "v5_raw": v5_vals,
        }
        results[system] = entry

        print(f"  RS  {rs_med:.4f}   v4  {entry['v4_median']:.4f} ({entry['v4_improv_pct']:+.1f}%)"
              f"   v5  {entry['v5_median']:.4f} ({entry['v5_improv_pct']:+.1f}%)"
              f"  v5>v4 {sig_stars(p_v5_v4)}")

        pd.DataFrame({
            "run": list(range(1, n_repeats + 1)),
            "random_search": rs_vals,
            "flash_v4": v4_vals,
            "flash_v5": v5_vals,
        }).to_csv(os.path.join(raw_dir, f"{system}.csv"), index=False)

    _print_summary(results)
    _save_results(results, output_dir)
    return results


def _print_summary(results):
    print(f"\n{'System':>12s}  {'RS':>9s}  {'v4':>9s}  {'v5':>9s}  v5>v4")
    print("-" * 55)
    wins = 0
    for name in sorted(results):
        e = results[name]
        print(f"{name:>12s}  {e['rs_median']:9.4f}  {e['v4_median']:9.4f}"
              f"  {e['v5_median']:9.4f}  {sig_stars(e['p_v5_vs_v4'])}")
        if e["p_v5_vs_v4"] < 0.05:
            wins += 1
    print(f"\nv5 sig. better than v4: {wins}/{len(results)}")


def _save_results(results, output_dir):
    rows = [{k: v for k, v in e.items() if not k.endswith("_raw")}
            for e in results.values()]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    json.dump(results, open(os.path.join(output_dir, "results.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser(description="FLASH v5")
    ap.add_argument("--datasets", default="datasets")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--output", default="results_v5")
    args = ap.parse_args()

    if not os.path.isdir(args.datasets):
        raise FileNotFoundError(args.datasets)

    t0 = time.time()
    evaluate_v5(args.datasets, args.budget, args.repeats, args.output)
    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
