# Variant of v5 for small budgets: higher init_ratio and smaller batch size
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor

from flash_tuner import ConfigurationSpace, improv_pct, random_search, sig_stars
from flash_tuner_v2 import FeaturePreprocessor
from flash_tuner_v5 import flash_rf_v5


def flash_rf_v5_r30(space, budget, seed=42):
    """Small-budget variant: higher initial ratio and smaller batch size."""
    rng = np.random.RandomState(seed)
    all_X = space.get_all_configs()
    n_total = len(all_X)

    preprocessor = FeaturePreprocessor()
    preprocessor.fit(space._domains)
    all_X_enc = preprocessor.transform(all_X)
    n_dim = preprocessor.feature_count

    # Allocate more initial samples when budget is tight
    if budget <= 30:
        init_ratio = 0.50
    else:
        init_ratio = 0.30

    init_size = max(10, int(budget * init_ratio))
    init_size = min(init_size, n_total)
    init_indices = rng.choice(n_total, size=init_size, replace=False)

    done = np.zeros(n_total, dtype=bool)
    idx_done = []
    ys = []
    best = np.inf
    used = 0

    for idx in init_indices:
        if used >= budget:
            break
        if done[idx]:
            continue
        perf, _ = space.measure(all_X[idx].tolist())
        used += 1
        done[idx] = True
        idx_done.append(int(idx))
        ys.append(perf)
        if perf < best:
            best = perf

    actual_init = used

    y_arr = np.array(ys)
    y_min, y_max = y_arr.min(), y_arr.max()
    dynamic_range = (y_max - y_min) / (y_min + 1e-10)
    use_log = dynamic_range > 10.0

    n_trees = 10

    while used < budget:
        X_train = all_X_enc[idx_done]
        y_raw = np.array(ys)
        y_train = np.log1p(y_raw) if use_log else y_raw

        rf = RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=min(n_dim, 10),
            min_samples_leaf=max(1, len(y_train) // 15),
            random_state=int(rng.randint(100_000)),
        )
        rf.fit(X_train, y_train)

        tree_preds = np.array([t.predict(all_X_enc) for t in rf.estimators_])

        unmeasured = np.where(~done)[0]
        if len(unmeasured) == 0:
            break

        remaining = budget - used
        progress = min(1.0, (used - actual_init) / max(1, budget - actual_init))

        # Smaller batch to preserve more RF update frequency
        batch = min(3, remaining, len(unmeasured))

        n_subsample = max(1, int(n_trees * (0.3 + 0.5 * progress)))

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
            if done[idx]:
                continue
            perf, _ = space.measure(all_X[idx].tolist())
            used += 1
            done[idx] = True
            idx_done.append(idx)
            ys.append(perf)
            if perf < best:
                best = perf

    return best


def evaluate(datasets_dir, budget=30, n_repeats=30, output_dir="results_v5_r30"):
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    results = {}

    print(f"\n=== V5-R30 (budget={budget}, repeats={n_repeats}) ===\n")

    seeds = list(range(n_repeats))

    for fname in files:
        system = fname.replace(".csv", "")
        path = os.path.join(datasets_dir, fname)
        space = ConfigurationSpace(path)
        print(f"[{system}] {space.n_configs} configs, optimal={space.best_possible:.4f}")

        rs_vals = []
        v5_base_vals = []
        v5_opt_vals = []

        for i, s in enumerate(seeds):
            rs_vals.append(random_search(space, budget, s))
            v5_base_vals.append(flash_rf_v5(space, budget, s))
            v5_opt_vals.append(flash_rf_v5_r30(space, budget, s))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_repeats}")

        rs = np.array(rs_vals)
        v5_base = np.array(v5_base_vals)
        v5_opt = np.array(v5_opt_vals)

        _, p_v5_rs = mannwhitneyu(v5_base, rs, alternative="less")
        _, p_opt_rs = mannwhitneyu(v5_opt, rs, alternative="less")
        _, p_opt_v5 = mannwhitneyu(v5_opt, v5_base, alternative="less")

        rs_med = float(np.median(rs))
        v5_base_med = float(np.median(v5_base))
        v5_opt_med = float(np.median(v5_opt))
        entry = {
            "system": system,
            "n_configs": space.n_configs,
            "n_features": space.n_features,
            "best_possible": space.best_possible,
            "rs_median": rs_med,
            "rs_iqr": float(np.percentile(rs, 75) - np.percentile(rs, 25)),
            "v5_base_median": v5_base_med,
            "v5_base_iqr": float(np.percentile(v5_base, 75) - np.percentile(v5_base, 25)),
            "v5_opt_median": v5_opt_med,
            "v5_opt_iqr": float(np.percentile(v5_opt, 75) - np.percentile(v5_opt, 25)),
            "v5_base_improv_pct": improv_pct(rs_med, v5_base_med),
            "v5_opt_improv_pct": improv_pct(rs_med, v5_opt_med),
            "p_v5_base_vs_rs": float(p_v5_rs),
            "p_v5_opt_vs_rs": float(p_opt_rs),
            "p_v5_opt_vs_v5_base": float(p_opt_v5),
            "v5_base_hits_optimal": int(np.sum(v5_base <= space.best_possible + 1e-9)),
            "v5_opt_hits_optimal": int(np.sum(v5_opt <= space.best_possible + 1e-9)),
            "rs_raw": rs_vals,
            "v5_base_raw": v5_base_vals,
            "v5_opt_raw": v5_opt_vals,
        }

        results[system] = entry

        print(f"  RS={rs_med:.4f}  base={v5_base_med:.4f}({entry['v5_base_improv_pct']:+.1f}%)  "
              f"opt={v5_opt_med:.4f}({entry['v5_opt_improv_pct']:+.1f}%) "
              f"opt>base {sig_stars(p_opt_v5)}")

        pd.DataFrame({
            "run": list(range(1, n_repeats + 1)),
            "random_search": rs_vals,
            "flash_v5_base": v5_base_vals,
            "flash_v5_opt": v5_opt_vals,
        }).to_csv(os.path.join(raw_dir, f"{system}.csv"), index=False)

    print(f"\n--- Summary ---")
    for name in sorted(results):
        e = results[name]
        print(f"  {name:>12s}  base {e['v5_base_improv_pct']:+.1f}%  "
              f"opt {e['v5_opt_improv_pct']:+.1f}%  "
              f"opt_hit={e['v5_opt_hits_optimal']}/30  "
              f"{sig_stars(e['p_v5_opt_vs_v5_base'])}")

    rows = [{k: v for k, v in e.items() if not k.endswith("_raw")}
            for e in results.values()]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--budget", type=int, default=30,
                        help="measurement budget (default 30)")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="results_v5_r30")
    args = parser.parse_args()

    assert os.path.isdir(args.datasets), f'{args.datasets} not found'

    t0 = time.time()
    evaluate(args.datasets, args.budget, args.repeats, args.output)
    print(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
