# Tried LightGBM as surrogate; found it less stable than RF under small samples
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from flash_tuner import ConfigurationSpace, improv_pct, random_search, sig_stars
from flash_tuner_v2 import flash_rf_v2


class LGBFeatureAnalyzer:
    """Feature analyzer for LightGBM: uses native categorical support instead of OHE."""

    def __init__(self):
        self._actions = []
        self._categorical_indices = []

    def fit(self, domains):
        self._actions = []
        self._categorical_indices = []
        out_idx = 0

        for col_idx, (col_name, values) in enumerate(domains.items()):
            vals = sorted(values)
            n_levels = len(vals)

            if n_levels <= 2:
                self._actions.append((col_idx, "keep"))
                out_idx += 1
            elif self._is_geometric(vals):
                self._actions.append((col_idx, "log2"))
                out_idx += 1
            elif n_levels <= 6:
                self._actions.append((col_idx, "categorical"))
                self._categorical_indices.append(out_idx)
                out_idx += 1
            else:
                self._actions.append((col_idx, "keep"))
                out_idx += 1

    def transform(self, X):
        parts = []
        for col_idx, action in self._actions:
            col = X[:, col_idx].astype(np.float64)
            if action == "log2":
                parts.append(np.log2(np.maximum(col, 1.0)).reshape(-1, 1))
            else:
                parts.append(col.reshape(-1, 1))
        return np.hstack(parts)

    @property
    def categorical_indices(self):
        return self._categorical_indices

    @staticmethod
    def _is_geometric(vals):
        positive = [v for v in vals if v > 0]
        if len(positive) < 3:
            return False
        ratios = [positive[i + 1] / positive[i] for i in range(len(positive) - 1)]
        if any(r <= 1.5 for r in ratios):
            return False
        mean_r = np.mean(ratios)
        std_r = np.std(ratios)
        return std_r / mean_r < 0.2 if mean_r > 0 else False


N_ENSEMBLE = 15


def _build_lgb_ensemble(X, y, cat_indices, rng, n_models=N_ENSEMBLE):
    """Train an ensemble of LightGBM models on bootstrap subsets; predict with mean and std."""
    n = len(y)
    n_rounds = min(30, max(10, n // 3))
    min_leaf = max(2, n // 20)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": min(8, max(4, n // 5)),
        "min_data_in_leaf": min_leaf,
        "learning_rate": 0.15,
        "lambda_l2": 1.5,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "num_threads": 1,
    }

    models = []
    for i in range(n_models):
        boot_idx = rng.choice(n, size=n, replace=True)
        X_boot, y_boot = X[boot_idx], y[boot_idx]

        params["seed"] = int(rng.randint(100_000))
        ds = lgb.Dataset(
            X_boot, label=y_boot,
            categorical_feature=cat_indices if cat_indices else "auto",
            free_raw_data=False,
        )
        model = lgb.train(params, ds, num_boost_round=n_rounds,
                          callbacks=[lgb.log_evaluation(period=0)])
        models.append(model)

    return models


def _ensemble_predict(models, X):
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0), preds.std(axis=0)


def flash_lgb_v3(space, budget, seed=42):
    """v3: LightGBM ensemble surrogate replacing RF. Rejected — less stable than RF at n≤100."""
    if not HAS_LGB:
        raise RuntimeError("lightgbm is required for v3; install via: pip install lightgbm")

    rng = np.random.RandomState(seed)
    all_X = space.get_all_configs()
    n_total = len(all_X)

    analyzer = LGBFeatureAnalyzer()
    analyzer.fit(space._domains)
    all_X_enc = analyzer.transform(all_X)
    cat_indices = analyzer.categorical_indices

    init_size = max(10, int(budget * 0.3))
    init_indices = rng.choice(n_total, size=min(init_size, n_total), replace=False)

    tried = np.zeros(n_total, dtype=bool)
    sampled = []
    y_sampled = []
    best = np.inf
    n_eval = 0

    for idx in init_indices:
        if n_eval >= budget:
            break
        if tried[idx]:
            continue
        perf, _ = space.measure(all_X[idx].tolist())
        n_eval += 1
        tried[idx] = True
        sampled.append(int(idx))
        y_sampled.append(perf)
        if perf < best:
            best = perf

    actual_init = n_eval

    while n_eval < budget:
        X_train = all_X_enc[sampled]
        y_train = np.array(y_sampled)

        ensemble = _build_lgb_ensemble(X_train, y_train, cat_indices, rng)
        mu, sigma = _ensemble_predict(ensemble, all_X_enc)

        progress = min(1.0, (n_eval - actual_init) / max(1, budget - actual_init))
        base_kappa = 1.5 - 1.2 * progress

        mu_train, _ = _ensemble_predict(ensemble, X_train)
        corr = np.corrcoef(y_train, mu_train)[0, 1]
        if not np.isfinite(corr) or corr < 0:
            corr = 0.0
        kappa = base_kappa * max(0.1, (1.0 - corr) ** 2)

        # Increase exploration when model correlation with training data is low
        explore_ratio = 0.4 if corr < 0.3 else 0.2

        lcb = mu - kappa * sigma

        unmeasured = np.where(~tried)[0]
        if len(unmeasured) == 0:
            break

        lcb_vals = lcb[unmeasured]
        order = np.argsort(lcb_vals)
        sorted_unmeasured = unmeasured[order]

        remaining = budget - n_eval
        batch = min(5, remaining, len(sorted_unmeasured))
        n_exploit = max(1, int(batch * (1.0 - explore_ratio)))
        n_explore = batch - n_exploit

        selected = sorted_unmeasured[:n_exploit].tolist()

        if n_explore > 0:
            exploit_set = set(selected)
            explore_pool = np.array([i for i in unmeasured if i not in exploit_set])
            if len(explore_pool) > 0:
                picks = rng.choice(
                    explore_pool,
                    size=min(n_explore, len(explore_pool)),
                    replace=False,
                )
                selected.extend(picks.tolist())

        for idx in selected:
            if n_eval >= budget:
                break
            if tried[idx]:
                continue
            perf, _ = space.measure(all_X[idx].tolist())
            n_eval += 1
            tried[idx] = True
            sampled.append(idx)
            y_sampled.append(perf)
            if perf < best:
                best = perf

    return best


def evaluate_v3(datasets_dir, budget=100, n_repeats=30, output_dir="results_v3"):
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    results = {}

    print(f"\nv3 Evaluation (v2=RF-LCB, v3=LGB-ensemble)")
    print(f"Budget={budget}  Repeats={n_repeats}  Systems={len(files)}\n")

    seeds = [i * 42 + 1 for i in range(n_repeats)]

    for fname in files:
        system = fname.replace(".csv", "")
        path = os.path.join(datasets_dir, fname)
        space = ConfigurationSpace(path)
        print(f"[{system}] {space.n_configs} configs x {space.n_features} feats "
              f"(optimal={space.best_possible:.4f})")

        rs_vals, v2_vals, v3_vals = [], [], []

        for i, s in enumerate(seeds):
            rs_vals.append(random_search(space, budget, s))
            v2_vals.append(flash_rf_v2(space, budget, s))
            v3_vals.append(flash_lgb_v3(space, budget, s))
            if (i + 1) % 10 == 0:
                print(f"  ... {i + 1}/{n_repeats} runs done")

        rs = np.array(rs_vals)
        v2 = np.array(v2_vals)
        v3 = np.array(v3_vals)

        _, p_v2_rs = mannwhitneyu(v2, rs, alternative="less")
        _, p_v3_rs = mannwhitneyu(v3, rs, alternative="less")
        _, p_v3_v2 = mannwhitneyu(v3, v2, alternative="less")

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
            "v3_median": float(np.median(v3)),
            "v3_iqr": float(np.percentile(v3, 75) - np.percentile(v3, 25)),
            "v2_improv_pct": improv_pct(rs_med, np.median(v2)),
            "v3_improv_pct": improv_pct(rs_med, np.median(v3)),
            "p_v2_vs_rs": float(p_v2_rs),
            "p_v3_vs_rs": float(p_v3_rs),
            "p_v3_vs_v2": float(p_v3_v2),
            "v2_hits_optimal": int(np.sum(v2 <= space.best_possible + 1e-9)),
            "v3_hits_optimal": int(np.sum(v3 <= space.best_possible + 1e-9)),
            "rs_raw": rs_vals,
            "v2_raw": v2_vals,
            "v3_raw": v3_vals,
        }
        results[system] = entry

        print(f"  RS  median={rs_med:.4f}  IQR={entry['rs_iqr']:.4f}")
        print(f"  v2  median={entry['v2_median']:.4f}  IQR={entry['v2_iqr']:.4f}"
              f"  improv={entry['v2_improv_pct']:+.2f}%"
              f"  opt={entry['v2_hits_optimal']}/{n_repeats}"
              f"  {sig_stars(p_v2_rs)}")
        print(f"  v3  median={entry['v3_median']:.4f}  IQR={entry['v3_iqr']:.4f}"
              f"  improv={entry['v3_improv_pct']:+.2f}%"
              f"  opt={entry['v3_hits_optimal']}/{n_repeats}"
              f"  vs_v2={sig_stars(p_v3_v2)}")
        print()

        raw_data = {
            "run": list(range(1, n_repeats + 1)),
            "random_search": rs_vals,
            "flash_v2": v2_vals,
            "flash_v3": v3_vals,
        }
        pd.DataFrame(raw_data).to_csv(
            os.path.join(raw_dir, f"{system}.csv"), index=False)

    _print_summary(results)
    _save_results(results, output_dir)
    return results


def _print_summary(results):
    """Print summary: v3 vs v2 vs RS."""
    print("\n--- Summary: v3 vs v2 vs RS ---")
    n_better_rs = sum(1 for e in results.values() if e["p_v3_vs_rs"] < 0.05)
    n_better_v2 = sum(1 for e in results.values() if e["p_v3_vs_v2"] < 0.05)
    print(f"v3 sig. better than RS: {n_better_rs}/{len(results)}")
    print(f"v3 sig. better than v2: {n_better_v2}/{len(results)}")
    for name in sorted(results):
        e = results[name]
        print(f"  {name:>12s}  RS={e['rs_median']:.4f}  v2={e['v2_median']:.4f}"
              f"  v3={e['v3_median']:.4f}  v3>v2 {sig_stars(e['p_v3_vs_v2'])}")


def _save_results(results, output_dir):
    summary_rows = []
    for e in results.values():
        row = {k: v for k, v in e.items() if not k.endswith("_raw")}
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "summary.csv"), index=False)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="FLASH v3: LightGBM ensemble variant")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=30,
                        help="number of repetitions (default: 30)")
    parser.add_argument("--output", default="results_v3")
    args = parser.parse_args()

    if not HAS_LGB:
        sys.exit("lightgbm is required: pip install lightgbm")

    if not os.path.isdir(args.datasets):
        sys.exit(f"datasets directory not found: {args.datasets}")

    t0 = time.time()
    evaluate_v3(args.datasets, args.budget, args.repeats, args.output)
    print(f"\nTotal elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
