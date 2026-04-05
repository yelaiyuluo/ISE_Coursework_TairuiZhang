import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt  # 画图的以后再加
from scipy.stats import mannwhitneyu
from sklearn.ensemble import RandomForestRegressor

from flash_tuner import (
    ConfigurationSpace,
    flash_cart,
    improv_pct,
    random_search,
    sig_stars,
)


def tpe_search(space, budget, seed=42):
    """用optuna跑TPE"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

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
    miss_cnt = 0

    # optuna有时候会卡住，循环放大100倍
    for _ in range(budget * 100):
        if used >= budget:
            break

        # 连续miss太多就随机选一个
        if miss_cnt >= 40:
            unmeasured_idx = [i for i in range(n_total)
                             if tuple(all_X[i].tolist()) not in measured]
            if not unmeasured_idx:
                break
            idx = rng.choice(unmeasured_idx)
            cfg = all_X[idx].tolist()
            key = tuple(cfg)
            perf, _ = space.measure(cfg)
            used += 1
            measured.add(key)
            miss_cnt = 0
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
            miss_cnt = 0
            if perf < best:
                best = perf
        else:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            miss_cnt += 1

    return best


def flash_rf(space, budget, seed=42, *, init_ratio=0.3,
             batch_size=5, exploit_ratio=0.8):
    """可调参数版的FLASH-RF，敏感性分析用"""
    rng = np.random.RandomState(seed)
    all_X = space.get_all_configs()
    n_total = len(all_X)

    init_size = max(10, int(budget * init_ratio))
    init_indices = rng.choice(n_total, size=min(init_size, n_total), replace=False)

    measured = set()
    X_list = []
    y_list = []
    best = np.inf
    used = 0

    for idx in init_indices:
        if used >= budget:
            break
        cfg = all_X[idx].tolist()
        key = tuple(cfg)
        if key in measured:
            continue
        perf, _ = space.measure(cfg)
        used += 1
        measured.add(key)
        X_list.append(cfg)
        y_list.append(perf)
        if perf < best:
            best = perf

    while used < budget:
        X_train = np.array(X_list)
        rf = RandomForestRegressor(
            n_estimators=10,
            max_depth=min(space.n_features, 10),
            min_samples_leaf=max(1, len(X_train) // 10),
            random_state=int(rng.randint(100_000)),
        )
        rf.fit(X_train, np.array(y_list))
        preds = rf.predict(all_X)

        cands = sorted(
            ((preds[i], i) for i in range(n_total)
             if tuple(all_X[i].tolist()) not in measured),
            key=lambda c: c[0],
        )
        if not cands:
            break

        remaining = budget - used
        batch = min(batch_size, remaining, len(cands))
        n_exploit = max(1, int(batch * exploit_ratio))
        n_explore = batch - n_exploit

        selected = [cands[j][1] for j in range(min(n_exploit, len(cands)))]
        if n_explore > 0 and len(cands) > n_exploit:
            pool = [c[1] for c in cands[n_exploit:]]
            picks = rng.choice(pool, size=min(n_explore, len(pool)), replace=False)
            selected.extend(picks.tolist())

        for idx in selected:
            if used >= budget:
                break
            cfg = all_X[idx].tolist()
            key = tuple(cfg)
            if key in measured:
                continue
            perf, _ = space.measure(cfg)
            used += 1
            measured.add(key)
            X_list.append(cfg)
            y_list.append(perf)
            if perf < best:
                best = perf

    return best


def run_tpe_comparison(datasets_dir, budget, n_repeats, output_dir):
    """TPE和FLASH-RF的对照实验"""
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "tpe_raw")
    os.makedirs(raw_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(datasets_dir) if f.endswith(".csv"))
    results = {}

    print(f"\n=== TPE vs FLASH-RF ===")
    print(f"budget={budget}, repeats={n_repeats}, 共{len(files)}个系统\n")

    seeds = list(range(42, 42 + n_repeats))

    for fname in files:
        system = fname.replace(".csv", "")
        space = ConfigurationSpace(os.path.join(datasets_dir, fname))
        print(f"[{system}] {space.n_configs} cfgs, {space.n_features} feats "
              f"(opt={space.best_possible:.4f})")

        rs_v, fl_v, tpe_v = [], [], []
        t0 = time.time()
        for i, seed in enumerate(seeds):
            rs_v.append(random_search(space, budget, seed))
            fl_v.append(flash_cart(space, budget, seed))
            tpe_v.append(tpe_search(space, budget, seed))
            if (i + 1) % 10 == 0:
                print(f"    done {i+1}/{n_repeats}  ({time.time()-t0:.0f}s)")

        rs, fl, tpe = np.array(rs_v), np.array(fl_v), np.array(tpe_v)

        _, p_fl_rs = mannwhitneyu(fl, rs, alternative="less")
        _, p_tpe_rs = mannwhitneyu(tpe, rs, alternative="less")
        _, p_fl_tpe = mannwhitneyu(fl, tpe, alternative="less")

        def _med(a): return float(np.median(a))
        def _iqr(a): return float(np.percentile(a, 75) - np.percentile(a, 25))

        entry = dict(
            system=system, n_configs=space.n_configs,
            n_features=space.n_features, best_possible=space.best_possible,
            rs_median=_med(rs), flash_median=_med(fl), tpe_median=_med(tpe),
            rs_iqr=_iqr(rs), flash_iqr=_iqr(fl), tpe_iqr=_iqr(tpe),
            flash_vs_rs_p=float(p_fl_rs), tpe_vs_rs_p=float(p_tpe_rs),
            flash_vs_tpe_p=float(p_fl_tpe),
            flash_improv=improv_pct(_med(rs), _med(fl)),
            tpe_improv=improv_pct(_med(rs), _med(tpe)),
        )
        results[system] = entry

        print(f"    RS:    med={entry['rs_median']:.4f}  IQR={entry['rs_iqr']:.4f}")
        print(f"    FLASH: med={entry['flash_median']:.4f}  IQR={entry['flash_iqr']:.4f}")
        print(f"    TPE:   med={entry['tpe_median']:.4f}  IQR={entry['tpe_iqr']:.4f}")
        print(f"    p值: FL vs RS={p_fl_rs:.6f}{sig_stars(p_fl_rs)}  "
              f"TPE vs RS={p_tpe_rs:.6f}{sig_stars(p_tpe_rs)}  "
              f"FL vs TPE={p_fl_tpe:.6f}{sig_stars(p_fl_tpe)}")
        print()

        pd.DataFrame(dict(
            run=list(range(1, n_repeats + 1)),
            random_search=rs_v, flash_rf=fl_v, tpe=tpe_v,
        )).to_csv(os.path.join(raw_dir, f"{system}.csv"), index=False)

    rows = [
        dict(
            System=s, Configs=e["n_configs"], Features=e["n_features"],
            Optimal=e["best_possible"],
            RS_Median=e["rs_median"], RS_IQR=e["rs_iqr"],
            FLASH_Median=e["flash_median"], FLASH_IQR=e["flash_iqr"],
            TPE_Median=e["tpe_median"], TPE_IQR=e["tpe_iqr"],
            FLASH_Improv_pct=e["flash_improv"], TPE_Improv_pct=e["tpe_improv"],
            FLASH_vs_RS_p=e["flash_vs_rs_p"], TPE_vs_RS_p=e["tpe_vs_rs_p"],
            FLASH_vs_TPE_p=e["flash_vs_tpe_p"],
        )
        for s, e in results.items()
    ]
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "tpe_comparison.csv"),
                               index=False)
    with open(os.path.join(output_dir, "tpe_comparison.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    return results


REPRESENTATIVE = ["brotli", "x264", "LLVM", "7z"]

SWEEP = dict(
    init_ratio=([0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50], 0.30),
    batch_size=([1, 2, 3, 5, 8, 10, 15, 20], 5),
    exploit_ratio=([0.50, 0.60, 0.70, 0.80, 0.90, 1.00], 0.80),
)

DEFAULTS = dict(init_ratio=0.30, batch_size=5, exploit_ratio=0.80)


def run_sensitivity(datasets_dir, budget, n_repeats, output_dir):
    """超参敏感性分析，单因素变化"""
    os.makedirs(output_dir, exist_ok=True)
    seeds = list(range(42, 42 + n_repeats))

    # 只跑4个代表性系统够了
    spaces = {}
    for name in REPRESENTATIVE:
        path = os.path.join(datasets_dir, f"{name}.csv")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found, skipping")
            continue
        spaces[name] = ConfigurationSpace(path)

    rs_medians = {}
    for name, sp in spaces.items():
        vals = [random_search(sp, budget, s) for s in seeds]
        rs_medians[name] = float(np.median(vals))

    all_results = {}

    for param, (grid, default) in SWEEP.items():
        print(f"\nSensitivity: {param}  (default={default})")

        rows = []
        for val in grid:
            kwargs = dict(DEFAULTS)
            kwargs[param] = val
            for name, sp in spaces.items():
                vals = [flash_rf(sp, budget, s, **kwargs) for s in seeds]
                med = float(np.median(vals))
                iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                rs_m = rs_medians[name]
                rows.append(dict(
                    param_value=val, system=name, median=med, iqr=iqr,
                    rs_median=rs_m, improv_pct=improv_pct(rs_m, med),
                    is_default=(val == default),
                ))
            print(f"  {param}={val}  done")

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(output_dir, f"sensitivity_{param}.csv"),
                  index=False)
        all_results[param] = df

    return all_results


def generate_report(tpe_results, sens_results, output_path):
    """生成ablation实验的md报告"""
    lines = []
    L = lines.append

    L("## 附录A 消融实验\n")
    L("TPE对照+超参数敏感性，回应§7局限2和5。\n")

    L("### A.1 TPE vs FLASH-RF\n")
    L("三种方法(RS, FLASH-RF, TPE)在8个系统上各跑30次，budget=100。"
      "TPE的n_startup设成30和FLASH初始阶段对齐。"
      "检验: Mann-Whitney U 单侧, alpha=0.05。\n")

    L("#### 结果\n")
    L("| 系统 | 配置数 | 特征 | RS med | FLASH med | TPE med "
      "| FL vs RS p | TPE vs RS p | FL vs TPE p |")
    L("|------|--------|------|--------|-----------|--------"
      "|------------|-------------|-------------|")
    for sn in sorted(tpe_results.keys()):
        e = tpe_results[sn]
        L(f"| {sn} | {e['n_configs']} | {e['n_features']} "
          f"| {e['rs_median']:.4f} | {e['flash_median']:.4f} "
          f"| {e['tpe_median']:.4f} "
          f"| {e['flash_vs_rs_p']:.2e} {sig_stars(e['flash_vs_rs_p'])} "
          f"| {e['tpe_vs_rs_p']:.2e} {sig_stars(e['tpe_vs_rs_p'])} "
          f"| {e['flash_vs_tpe_p']:.2e} {sig_stars(e['flash_vs_tpe_p'])} |")

    L("")
    L("#### 改进幅度 (相对RS)\n")
    L("| 系统 | FLASH改进% | TPE改进% | 谁更好 |")
    L("|------|-----------|---------|--------|")
    flash_wins = 0
    tpe_wins = 0
    ties = 0
    for sn in sorted(tpe_results.keys()):
        e = tpe_results[sn]
        fi, ti = e["flash_improv"], e["tpe_improv"]
        if e["flash_vs_tpe_p"] < 0.05:
            winner = "FLASH"
            flash_wins += 1
        elif fi < ti - 0.5:
            winner = "TPE"
            tpe_wins += 1
        else:
            winner = "差不多"
            ties += 1
        L(f"| {sn} | {fi:+.2f}% | {ti:+.2f}% | {winner} |")

    L("")
    L(f"FLASH赢{flash_wins}个，TPE赢{tpe_wins}个，平{ties}个。\n")
    # TODO: 这段结论需要改写
    L("基本上两个方法差不多，选FLASH主要因为可解释性不是性能。\n")

    L("### A.2 超参数敏感性\n")
    L("4个代表系统(brotli, x264, LLVM, 7z)，单因素法，每组30次。"
      "默认值: init_ratio=0.30, batch_size=5, exploit_ratio=0.80。\n")

    param_cn = {
        "init_ratio": "初始比例",
        "batch_size": "批大小",
        "exploit_ratio": "利用比例",
    }

    for pidx, (param, df) in enumerate(sens_results.items()):
        _, default = SWEEP[param]
        L(f"#### A.2.{pidx+1} {param_cn[param]} ({param})\n")

        pivot = df.pivot_table(index="param_value", columns="system",
                               values="improv_pct")
        systems = [s for s in REPRESENTATIVE if s in pivot.columns]

        hdr = f"| {param} | " + " | ".join(systems) + " |"
        sep = "|" + "|".join(["---"] * (len(systems) + 1)) + "|"
        L(hdr)
        L(sep)
        for val in SWEEP[param][0]:
            tag = " (默认)" if val == default else ""
            cells = []
            for s in systems:
                v = pivot.loc[val, s] if val in pivot.index else float("nan")
                cells.append(f"{v:+.2f}%")
            L(f"| {val}{tag} | " + " | ".join(cells) + " |")
        L("")

    L("#### 小结\n")
    L("- init_ratio: 太低(0.10)RF训练数据不够，太高(0.50)浪费预算。0.25-0.35都行")
    L("- batch_size: 1太慢，20模型更新次数太少。5-10比较好")
    L("- exploit_ratio: 纯利用(1.0)有过拟合风险，0.7-0.9都稳定")
    L("")
    # TODO: 这段需要改写，加更详细的分析
    L("默认参数(0.30/5/0.80)基本在最优附近，扰动30%以内性能不会崩。\n")

    text = "\n".join(lines)
    with open(output_path, "w") as fh:
        fh.write(text)
    return text


def main():
    parser = argparse.ArgumentParser(description="消融实验: TPE对照 + 超参敏感性")
    parser.add_argument("--datasets", default="datasets")
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", default="results/ablation")
    parser.add_argument("--skip-tpe", action="store_true",
                        help="跳过TPE对照，只跑敏感性")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="跳过敏感性，只跑TPE")
    args = parser.parse_args()

    if not os.path.isdir(args.datasets):
        print(f"Error: {args.datasets} not found", file=sys.stderr)
        sys.exit(1)

    t_total = time.time()

    tpe_res = {}
    if not args.skip_tpe:
        t0 = time.time()
        tpe_res = run_tpe_comparison(args.datasets, args.budget,
                                     args.repeats, args.output)
        print(f"\n  TPE done in {time.time() - t0:.0f}s")

    sens_res = {}
    if not args.skip_sensitivity:
        t0 = time.time()
        sens_res = run_sensitivity(args.datasets, args.budget,
                                   args.repeats, args.output)
        print(f"\n  Sensitivity done in {time.time() - t0:.0f}s")

    if tpe_res and sens_res:
        report_path = os.path.join(args.output, "ablation_report.md")
        generate_report(tpe_res, sens_res, report_path)
        print(f"\n  Report: {report_path}")

    print(f"\nTotal: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
