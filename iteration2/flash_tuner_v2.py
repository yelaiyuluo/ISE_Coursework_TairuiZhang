import argparse
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor

from flash_tuner import ConfigurationSpace, improv_pct, random_search, flash_cart, sig_stars

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class FeaturePreprocessor:
    """Automatic feature encoder: detects binary, geometric, and categorical variables."""

    def __init__(self):
        self._plan = []
        self.feature_count = 0

    def fit(self, domains):
        self._plan = []
        dim = 0
        for col_idx, (col_name, values) in enumerate(domains.items()):
            vals = sorted(values)
            n_levels = len(vals)

            if n_levels <= 2:
                self._plan.append((col_idx, "keep", None))
                dim += 1
            elif self._is_geometric(vals):
                self._plan.append((col_idx, "log2", None))
                dim += 1
            elif self._is_categorical(vals):
                self._plan.append((col_idx, "ohe", vals))
                dim += n_levels
            else:
                self._plan.append((col_idx, "keep", None))
                dim += 1

        self.feature_count = dim

    def transform(self, X):
        parts = []
        for col_idx, action, info in self._plan:
            col = X[:, col_idx].astype(np.float64)
            if action == "keep":
                parts.append(col.reshape(-1, 1))
            elif action == "log2":
                parts.append(np.log2(np.maximum(col, 1.0)).reshape(-1, 1))
            elif action == "ohe":
                for level in info:
                    parts.append((col == level).astype(np.float64).reshape(-1, 1))
        return np.hstack(parts)

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

    @staticmethod
    def _is_categorical(vals):
        if len(vals) < 3 or len(vals) > 6:
            return False
        diffs = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        return all(d == 1 for d in diffs)


def flash_rf_v2(space, budget, seed=42):
    """v2: RF surrogate with LCB acquisition and feature preprocessing."""
    rng = np.random.RandomState(seed)
    all_X = space.get_all_configs()
    n_total = len(all_X)

    preprocessor = FeaturePreprocessor()
    preprocessor.fit(space._domains)
    all_X_enc = preprocessor.transform(all_X)
    n_dim = preprocessor.feature_count

    init_size = max(10, int(budget * 0.3))
    init_indices = rng.choice(n_total, size=min(init_size, n_total), replace=False)

    visited = np.zeros(n_total, dtype=bool)
    obs_idx = []
    y_obs = []
    best = np.inf
    used = 0

    for idx in init_indices:
        if used >= budget:
            break
        if visited[idx]:
            continue
        perf, _ = space.measure(all_X[idx].tolist())
        used += 1
        visited[idx] = True
        obs_idx.append(int(idx))
        y_obs.append(perf)
        if perf < best:
            best = perf

    actual_init = used

    while used < budget:
        X_train = all_X_enc[obs_idx]
        y_train = np.array(y_obs)

        rf = RandomForestRegressor(
            n_estimators=20,
            max_depth=min(n_dim, 12),
            min_samples_leaf=max(2, len(y_train) // 20),
            oob_score=True,
            random_state=int(rng.randint(100_000)),
        )
        rf.fit(X_train, y_train)

        oob_raw = 0.0
        try:
            oob_raw = rf.oob_score_
        except AttributeError:
            pass
        oob = max(0.0, oob_raw) if np.isfinite(oob_raw) else 0.0

        # Kappa decays as search progresses; scale down further when model is accurate
        progress = min(1.0, (used - actual_init) / max(1, budget - actual_init))
        base_kappa = 1.5 - 1.2 * progress
        kappa = base_kappa * max(0.1, (1.0 - oob) ** 2)

        # Increase exploration ratio when OOB score is negative (model unreliable)
        explore_ratio = 0.4 if oob_raw < 0 else 0.2

        # LCB = mu - kappa * sigma
        tree_preds = np.array([t.predict(all_X_enc) for t in rf.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
        lcb = mu - kappa * sigma

        unmeasured = np.where(~visited)[0]
        if len(unmeasured) == 0:
            break

        lcb_vals = lcb[unmeasured]
        order = np.argsort(lcb_vals)
        sorted_unmeasured = unmeasured[order]

        remaining = budget - used
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
            if used >= budget:
                break
            if visited[idx]:
                continue
            perf, _ = space.measure(all_X[idx].tolist())
            used += 1
            visited[idx] = True
            obs_idx.append(idx)
            y_obs.append(perf)
            if perf < best:
                best = perf

    return best


def tpe_search(space, budget, seed=42):
    """TPE via Optuna."""
    if not HAS_OPTUNA:
        raise RuntimeError("optuna is required for TPE; install via: pip install optuna")

    all_X = space.get_all_configs()
    n_total = len(all_X)
    rng = np.random.RandomState(seed)

    n_startup = max(10, int(budget * 0.3))
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    domains = {c: [int(v) for v in space._domains[c]] for c in space.config_cols}

    best = np.inf
    used = 0
    measured = set()
    miss_streak = 0

    for _ in range(budget * 100):
        if used >= budget:
            break

        # Fall back to random selection after 40 consecutive duplicate proposals
        if miss_streak >= 40:
            unmeasured_idx = [i for i in range(n_total)
                             if tuple(all_X[i].tolist()) not in measured]
            if not unmeasured_idx:
                break
            idx = rng.choice(unmeasured_idx)
            cfg = all_X[idx].tolist()
            perf, _ = space.measure(cfg)
            used += 1
            measured.add(tuple(cfg))
            miss_streak = 0
            if perf < best:
                best = perf
            trial = study.ask()
            for c in space.config_cols:
                trial.suggest_categorical(c, domains[c])
            study.tell(trial, perf)
            continue

        trial = study.ask()
        config = [int(trial.suggest_categorical(c, domains[c]))
                  for c in space.config_cols]
        key = tuple(config)
        perf, valid = space.measure(config)

        if valid and key not in measured:
            study.tell(trial, perf)
            used += 1
            measured.add(key)
            miss_streak = 0
            if perf < best:
                best = perf
        else:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            miss_streak += 1

    return best


def evaluate_v2(datasets_dir, budget=100, n_repeats=30,
                output_dir="results_v2", include_tpe=True):
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    results = {}
    run_tpe = include_tpe and HAS_OPTUNA

    methods = ["RS", "v1", "v2"]
    if run_tpe:
        methods.append("TPE")

    print(f"\n--- v2 evaluation ---")
    print(f"budget={budget} repeats={n_repeats} systems={len(files)} "
          f"methods={','.join(methods)}\n")

    seeds = [i * 42 + 1 for i in range(n_repeats)]

    for fname in files:
        system = fname.replace(".csv", "")
        path = os.path.join(datasets_dir, fname)
        space = ConfigurationSpace(path)
        print(f"[{system}] {space.n_configs}x{space.n_features} "
              f"(opt={space.best_possible:.4f})")

        rs_vals, v1_vals, v2_vals, tpe_vals = [], [], [], []

        for i, seed in enumerate(seeds):
            rs_vals.append(random_search(space, budget, seed))
            v1_vals.append(flash_cart(space, budget, seed))
            v2_vals.append(flash_rf_v2(space, budget, seed))
            if run_tpe:
                tpe_vals.append(tpe_search(space, budget, seed))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{n_repeats} done")

        rs = np.array(rs_vals)
        v1 = np.array(v1_vals)
        v2 = np.array(v2_vals)
        rs_med = float(np.median(rs))

        _, p_v2_rs = mannwhitneyu(v2, rs, alternative="less")
        _, p_v2_v1 = mannwhitneyu(v2, v1, alternative="less")

        entry = {
            "system": system,
            "n_configs": space.n_configs,
            "n_features": space.n_features,
            "best_possible": space.best_possible,
            "rs_median": rs_med,
            "rs_iqr": float(np.percentile(rs, 75) - np.percentile(rs, 25)),
            "v1_median": float(np.median(v1)),
            "v1_iqr": float(np.percentile(v1, 75) - np.percentile(v1, 25)),
            "v2_median": float(np.median(v2)),
            "v2_iqr": float(np.percentile(v2, 75) - np.percentile(v2, 25)),
            "p_v2_vs_rs": float(p_v2_rs),
            "p_v2_vs_v1": float(p_v2_v1),
            "v1_improv_pct": improv_pct(rs_med, np.median(v1)),
            "v2_improv_pct": improv_pct(rs_med, np.median(v2)),
            "v2_hits_optimal": int(np.sum(v2 <= space.best_possible + 1e-9)),
            "v1_hits_optimal": int(np.sum(v1 <= space.best_possible + 1e-9)),
            "rs_raw": rs_vals,
            "v1_raw": v1_vals,
            "v2_raw": v2_vals,
        }

        if run_tpe:
            tpe = np.array(tpe_vals)
            _, p_v2_tpe = mannwhitneyu(v2, tpe, alternative="less")
            entry["tpe_median"] = float(np.median(tpe))
            entry["tpe_iqr"] = float(np.percentile(tpe, 75) - np.percentile(tpe, 25))
            entry["p_v2_vs_tpe"] = float(p_v2_tpe)
            entry["tpe_improv_pct"] = improv_pct(rs_med, np.median(tpe))
            entry["tpe_hits_optimal"] = int(np.sum(tpe <= space.best_possible + 1e-9))
            entry["tpe_raw"] = tpe_vals

        results[system] = entry

        print(f"  RS  med={entry['rs_median']:.4f}  IQR={entry['rs_iqr']:.4f}")
        print(f"  v1  med={entry['v1_median']:.4f}  imp={entry['v1_improv_pct']:+.2f}%"
              f"  opt={entry['v1_hits_optimal']}/30")
        print(f"  v2  med={entry['v2_median']:.4f}  imp={entry['v2_improv_pct']:+.2f}%"
              f"  opt={entry['v2_hits_optimal']}/30"
              f"  v2>RS {sig_stars(p_v2_rs)} v2>v1 {sig_stars(p_v2_v1)}")
        if run_tpe:
            print(f"  TPE med={entry['tpe_median']:.4f}"
                  f"  imp={entry['tpe_improv_pct']:+.2f}%"
                  f"  opt={entry['tpe_hits_optimal']}/30"
                  f"  v2>TPE {sig_stars(entry['p_v2_vs_tpe'])}")
        print()

        raw_data = {"run": list(range(1, n_repeats + 1)),
                    "random_search": rs_vals, "flash_v1": v1_vals, "flash_v2": v2_vals}
        if run_tpe:
            raw_data["tpe"] = tpe_vals
        pd.DataFrame(raw_data).to_csv(
            os.path.join(raw_dir, f"{system}.csv"), index=False)

    # Print summary
    print("\n--- summary ---")
    v2_better_rs = 0
    v2_better_v1 = 0
    for nm in sorted(results):
        e = results[nm]
        line = (f"  {nm:>10s}  RS={e['rs_median']:.4f}  v1={e['v1_median']:.4f}"
                f"  v2={e['v2_median']:.4f}  chg={e['v2_improv_pct']:+.2f}%"
                f"  opt={e['v2_hits_optimal']}/30")
        if run_tpe and "tpe_median" in e:
            line += f"  TPE={e['tpe_median']:.4f}"
        print(line)
        if e["p_v2_vs_rs"] < 0.05:
            v2_better_rs += 1
        if e["p_v2_vs_v1"] < 0.05:
            v2_better_v1 += 1
    print(f"\nv2 sig. better than RS: {v2_better_rs}/{len(results)}")
    print(f"v2 sig. better than v1: {v2_better_v1}/{len(results)}")

    _save_results(results, output_dir)
    return results


def _save_results(results, out_dir):
    clean = [{k: v for k, v in e.items() if not k.endswith("_raw")}
             for e in results.values()]
    pd.DataFrame(clean).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="v2 evaluation script")
    parser.add_argument("--datasets", default="datasets", help="directory containing CSV datasets")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="results_v2")
    parser.add_argument("--no-tpe", action="store_true", help="skip TPE (faster)")
    args = parser.parse_args()

    if not os.path.isdir(args.datasets):
        print(f"Error: datasets directory '{args.datasets}' not found", file=sys.stderr)
        sys.exit(1)

    t0 = time.time()
    evaluate_v2(args.datasets, args.budget, args.repeats,
                args.output, include_tpe=not args.no_tpe)
    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
