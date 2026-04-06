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
from flash_tuner_v2 import flash_rf_v2, FeaturePreprocessor


def flash_rf_v4(space, budget, seed=42):
    """Thompson sampling acquisition with RF."""
    rng = np.random.RandomState(seed)
    all_X = space.get_all_configs()
    n_total = len(all_X)

    preprocessor = FeaturePreprocessor()
    preprocessor.fit(space._domains)
    all_X_enc = preprocessor.transform(all_X)
    n_dim = preprocessor.feature_count

    init_size = max(10, int(budget * 0.3))
    init_indices = rng.choice(n_total, size=min(init_size, n_total), replace=False)

    evaluated = np.zeros(n_total, dtype=bool)
    eval_idx = []
    perf_vals = []
    best = np.inf
    n_used = 0

    for idx in init_indices:
        if n_used >= budget:
            break
        if evaluated[idx]:
            continue
        perf, _ = space.measure(all_X[idx].tolist())
        n_used += 1
        evaluated[idx] = True
        eval_idx.append(int(idx))
        perf_vals.append(perf)
        if perf < best:
            best = perf

    actual_init = n_used

    n_trees = 20

    while n_used < budget:
        X_train = all_X_enc[eval_idx]
        y_train = np.array(perf_vals)

        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=min(n_dim, 12),
            min_samples_leaf=max(2, len(y_train) // 20),
            random_state=int(rng.randint(100_000)),
        )
        rf.fit(X_train, y_train)

        # get per-tree predictions for Thompson sampling
        tree_preds = np.array([t.predict(all_X_enc) for t in rf.estimators_])

        unmeasured = np.where(~evaluated)[0]
        if len(unmeasured) == 0:
            break

        remaining = budget - n_used
        batch = min(5, remaining, len(unmeasured))

        # subsample trees for natural exploration
        # fewer trees early → more variance → exploration
        # more trees later → less variance → exploitation
        progress = min(1.0, (n_used - actual_init) / max(1, budget - actual_init))
        n_subsample = max(1, int(n_trees * (0.2 + 0.6 * progress)))

        # Thompson Sampling: simpler than LCB — no kappa hyperparameter
        selected = []
        for _ in range(batch):
            tree_idx = rng.choice(n_trees, size=n_subsample, replace=False)
            sampled_pred = tree_preds[tree_idx].mean(axis=0)

            candidates = np.array([i for i in unmeasured if i not in set(selected)])
            if len(candidates) == 0:
                break
            sampled_vals = sampled_pred[candidates]
            best_candidate = candidates[np.argmin(sampled_vals)]
            selected.append(int(best_candidate))

        for idx in selected:
            if n_used >= budget:
                break
            if evaluated[idx]:
                continue
            perf, _ = space.measure(all_X[idx].tolist())
            n_used += 1
            evaluated[idx] = True
            eval_idx.append(idx)
            perf_vals.append(perf)
            if perf < best:
                best = perf

    return best


def evaluate_v4(datasets_dir, budget=100, n_repeats=30, output_dir="results_v4"):
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    results = {}

    print(f"\nFLASH-RF v4 Evaluation (v2=RF+LCB, v4=RF+ThompsonSampling)")
    print(f"Budget={budget}  Repeats={n_repeats}  Systems={len(files)}\n")

    seeds = [i * 42 + 1 for i in range(n_repeats)]

    for fname in files:
        system = fname.replace(".csv", "")
        path = os.path.join(datasets_dir, fname)
        space = ConfigurationSpace(path)
        print(f"[{system}] {space.n_configs} configs x {space.n_features} feats "
              f"(optimal={space.best_possible:.4f})")

        rs_vals = []
        v2_vals = []
        v4_vals = []

        for i, s in enumerate(seeds):
            rs_vals.append(random_search(space, budget, s))
            v2_vals.append(flash_rf_v2(space, budget, s))
            v4_vals.append(flash_rf_v4(space, budget, s))
            if (i + 1) % 10 == 0:
                print(f"  ... {i + 1}/{n_repeats} runs done")

        rs = np.array(rs_vals)
        v2 = np.array(v2_vals)
        v4 = np.array(v4_vals)

        _, p_v2_rs = mannwhitneyu(v2, rs, alternative="less")
        _, p_v4_rs = mannwhitneyu(v4, rs, alternative="less")
        _, p_v4_v2 = mannwhitneyu(v4, v2, alternative="less")

        rs_med = float(np.median(rs))
        entry = {
            "system": system,
            "n_configs": space.n_configs,
            "n_features": space.n_features,
            "best_possible": space.best_possible,
            "rs_median": rs_med,
            "rs_iqr": float(np.percentile(rs, 75) - np.percentile(rs, 25)),
            "v2_median": float(np.median(v2)),
            "v2_iqr": float(np.percentile(v2, 75) - np.percentile(v2, 25)),
            "v4_median": float(np.median(v4)),
            "v4_iqr": float(np.percentile(v4, 75) - np.percentile(v4, 25)),
            "v2_improv_pct": improv_pct(rs_med, np.median(v2)),
            "v4_improv_pct": improv_pct(rs_med, np.median(v4)),
            "p_v2_vs_rs": float(p_v2_rs),
            "p_v4_vs_rs": float(p_v4_rs),
            "p_v4_vs_v2": float(p_v4_v2),
            "v2_hits_optimal": int(np.sum(v2 <= space.best_possible + 1e-9)),
            "v4_hits_optimal": int(np.sum(v4 <= space.best_possible + 1e-9)),
            "rs_raw": rs_vals,
            "v2_raw": v2_vals,
            "v4_raw": v4_vals,
        }

        results[system] = entry

        print(f"  RS   med={rs_med:12.4f}  (IQR {entry['rs_iqr']:.4f})")
        print(f"  v2   med={entry['v2_median']:12.4f}  (IQR {entry['v2_iqr']:.4f})  "
              f"Improv={entry['v2_improv_pct']:+.2f}%  opt={entry['v2_hits_optimal']}/30"
              f"  v2>RS {sig_stars(p_v2_rs)}")
        print(f"  v4   med={entry['v4_median']:12.4f}  (IQR {entry['v4_iqr']:.4f})  "
              f"Improv={entry['v4_improv_pct']:+.2f}%  opt={entry['v4_hits_optimal']}/30"
              f"  v4>v2 {sig_stars(p_v4_v2)}")
        print()

        raw_data = {
            "run": list(range(1, n_repeats + 1)),
            "random_search": rs_vals,
            "flash_v2": v2_vals,
            "flash_v4": v4_vals,
        }
        pd.DataFrame(raw_data).to_csv(
            os.path.join(raw_dir, f"{system}.csv"), index=False)

    _print_summary(results)
    _save_results(results, output_dir)
    return results


def _print_summary(results):
    print("\n== Summary: v4 (Thompson Sampling) vs v2 (LCB) ==")
    print(f"{'System':>12s}  {'RS':>10s}  {'v2':>10s}  {'v4':>10s}"
          f"  {'v2%':>6s}  {'v4%':>6s}  {'v4>v2':>5s}")
    print("-" * 72)

    wins_rs, wins_v2 = 0, 0

    for name in sorted(results):
        e = results[name]
        print(f"{name:>12s}  {e['rs_median']:10.4f}  {e['v2_median']:10.4f}"
              f"  {e['v4_median']:10.4f}  {e['v2_improv_pct']:+5.1f}%"
              f"  {e['v4_improv_pct']:+5.1f}%"
              f"  {sig_stars(e['p_v4_vs_v2']):>5s}")
        if e["p_v4_vs_rs"] < 0.05:
            wins_rs += 1
        if e["p_v4_vs_v2"] < 0.05:
            wins_v2 += 1

    n = len(results)
    print(f"\nv4 sig. better than RS: {wins_rs}/{n}")
    print(f"v4 sig. better than v2: {wins_v2}/{n}")


def _save_results(results, output_dir):
    rows = []
    for e in results.values():
        rows.append({k: v for k, v in e.items() if not k.endswith("_raw")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

    with open(os.path.join(output_dir, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="FLASH-RF v4: Thompson Sampling replacing LCB")
    parser.add_argument("--datasets", default="datasets",
                        help="directory containing CSV dataset files")
    parser.add_argument("--budget", type=int, default=100,
                        help="measurement budget per system")
    parser.add_argument("--repeats", type=int, default=30,
                        help="number of independent repetitions")
    parser.add_argument("--output", default="results_v4",
                        help="output directory")
    args = parser.parse_args()

    if not os.path.isdir(args.datasets):
        print(f"Error: {args.datasets} not found")
        sys.exit(1)

    t0 = time.time()
    evaluate_v4(args.datasets, args.budget, args.repeats, args.output)
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
