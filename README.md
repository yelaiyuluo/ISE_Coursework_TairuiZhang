# FLASH-RF: Configuration Performance Tuning Tool (Seven-Iteration Study)

A Random Forest sequential surrogate optimisation tool for software configuration performance tuning, inspired by [FLASH (Nair et al., 2017)](#references). Evolved through seven iterations to the final solution **v6**: RF + Thompson Sampling (RL strategy) + conditional log transform + conditional stratified initial sampling.

## Problem Definition

Given a configurable software system with search space $X$ and black-box performance function $f$ (runtime / latency), find the optimal configuration under a fixed measurement budget:

$$x^* = \arg\min_{x \in X} f(x), \quad \text{s.t.} \; r \leq R$$

where $R = 100$ is the total number of allowed measurements. $f$ has no gradients, no analytical form, and each query consumes 1 unit of budget. Configuration options are discrete (binary, integer, enumerated).

## Datasets

8 real-world configurable software systems (CSV files provided as lookup tables):

| System | Domain | \|X\| | Features | Objective | Budget Coverage |
|--------|--------|------:|:--------:|-----------|:-----------:|
| 7z | Compression tool | 68,640 | 8 | Minimise runtime | 0.15% |
| Apache | Web server | 640 | 8 | Minimise latency | 15.6% |
| brotli | Compression tool | 180 | 2 | Minimise runtime | 55.6% |
| LLVM | Compiler | 65,536 | 16 | Minimise runtime | 0.15% |
| PostgreSQL | Database | 864 | 8 | Minimise runtime | 11.6% |
| spear | SAT solver | 16,384 | 14 | Minimise solve time | 0.61% |
| storm | Stream processing | 1,557 | 12 | Minimise latency | 6.4% |
| x264 | Video encoder | 4,608 | 10 | Minimise encoding time | 2.2% |

Each CSV: first $n{-}1$ columns are configuration parameters, last column is performance value. All systems are minimisation problems.

## Algorithm Overview (v6)

FLASH-RF v6 follows a two-phase sequential surrogate optimisation loop:

**Phase 1 — Initial Sampling** (30% of budget):
- If $|X| > 5000$: KMeans stratified sampling across the configuration space for uniform spatial coverage.
- Otherwise: plain random sampling.

**Phase 2 — Surrogate-Guided Sequential Optimisation** (remaining budget):
1. Preprocess features: OHE for categorical variables, log2 for geometric sequences.
2. Apply conditional log(1+y) transform on performance values when dynamic range > 10.
3. Train Random Forest (20 CARTs) on observed data.
4. Use Thompson Sampling to select next batch: subsample trees, take the mean prediction, pick the predicted-best unmeasured configuration.
5. Measure the selected configuration, update observations, repeat until budget exhausted.

## Core Technical Components

| Component | Technique | Role |
|-----------|-----------|------|
| Surrogate model | Random Forest (20 CARTs) | Predict unmeasured configuration performance from observations |
| Acquisition strategy | Thompson Sampling (RL/MAB) | Adaptive exploration-exploitation without manual $\kappa$ tuning |
| Target transform | Conditional log(1+y) (range > 10) | Compress extreme skew, improve RF resolution near optimum |
| Initial sampling | Conditional KMeans stratification (\|X\| > 5000) | Exploit unlabelled X structure for uniform spatial coverage |
| Feature preprocessing | OHE (categorical) + log2 (geometric) | Eliminate encoding defects and non-equidistant value ranges |

## Quick Start

**Requirements:** Python >= 3.8

```bash
pip install -r requirements.txt
python iteration6/flash_tuner_v6.py --datasets datasets/ --budget 100 --repeats 30
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | `datasets` | Directory containing CSV dataset files |
| `--budget` | `100` | Measurement budget per system |
| `--repeats` | `30` | Number of independent runs per system |
| `--output` | `results_v6` | Output directory for results |

## Project Structure

```
ISE_Coursework_TairuiZhang/
├── datasets/
│   ├── 7z.csv, Apache.csv, brotli.csv, LLVM.csv
│   └── PostgreSQL.csv, spear.csv, storm.csv, x264.csv
├── iteration1/
│   ├── flash_tuner.py
│   ├── run_ablation.py
│   └── results/
├── iteration2/
│   ├── flash_tuner_v2.py
│   └── results/
├── iteration3/
│   ├── flash_tuner_v3.py
│   └── results/
├── iteration4/
│   ├── flash_tuner_v4.py
│   └── results/
├── iteration5/
│   ├── flash_tuner_v5.py
│   ├── flash_tuner_v5_r30.py
│   ├── analyze_v5_vs_v4.py
│   └── results/ + results_r30/
├── iteration6/
│   ├── flash_tuner_v6.py
│   ├── run_extended_experiments.py
│   ├── compare_budgets.py
│   └── results/ + results_extended/
├── iteration7/
│   ├── flash_tuner_v7.py
│   └── results_v7/
├── report_long.md
├── report_long.pdf
├── requirements.pdf
├── manual.pdf
├── replication.pdf
└── requirements.txt
```

- **`datasets/`** — 8 real-world configurable software system CSV datasets.
- **`iteration1/`** — v1: FLASH-CART → RF Ensemble. Includes v1 implementation with RS/HC/SA/BestConfig baselines, TPE head-to-head + hyperparameter sensitivity ablation, and results.
- **`iteration2/`** — v2: RF-LCB + Feature Preprocessing.
- **`iteration3/`** — v3: LightGBM Attempt (rejected).
- **`iteration4/`** — v4: Thompson Sampling.
- **`iteration5/`** — v5: Conditional Log Transform. Includes R=30 budget variant and v5 vs v4 comparison analysis.
- **`iteration6/`** — v6: Conditional Stratified Sampling (**final solution**). Includes 6-method × 8-system × 3-budget extended experiments.
- **`iteration7/`** — v7: Rank Transform (rejected by experiment).
- **`report_long.md / .pdf`** — Full report (7-iteration complete record).
- **`requirements.pdf`** — Requirements document.
- **`manual.pdf`** — User manual.
- **`replication.pdf`** — Replication guide.

## Seven-Iteration Roadmap

| Iter. | Version | Core Change | Key Finding |
|-------|---------|-------------|-------------|
| 1 | v1 | RF(10) + mean ranking | 7z failure (−2.1%), exposing 5 compounding root causes |
| 2 | v2 | RF(20) + LCB + feature preprocessing | Fixed 7z, 4/8 significantly beat TPE |
| 3 | ~~v3~~ | LightGBM ensemble + LCB | **Rejected**: uniformly worse than v2 — variance reduction > bias reduction |
| 4 | v4 | RF(20) + Thompson Sampling | 3/8 significantly beat v2, LLVM 30/30 perfect hits |
| 5 | v5 | v4 + conditional log(1+y) | Zero degradation, 7z median +3.7% |
| 6 | **v6** | v5 + conditional KMeans stratified Phase-1 | **8/8 all significantly beat RS**, 7z first statistically significant |
| 7 | ~~v7~~ | v6 + conditional rank transform | **Rejected**: 3 systems degraded, total hits −6 |

## Experimental Protocol

All iterations use a unified evaluation framework:

- **Budget:** $R = 100$ per run.
- **Repetitions:** 30 independent runs per system, with deterministic seeds $s_i = 42i + 1$ ($i = 0, \ldots, 29$).
- **Metric:** Best performance value found within budget (lower is better).
- **Statistical test:** One-sided Mann-Whitney U test ($\alpha = 0.05$). $p < 0.05$: \*, $p < 0.01$: \*\*, $p < 0.001$: \*\*\*.
- **Hit-optimal rate:** Number of runs (out of 30) that find the globally optimal configuration.

## v6 Results Summary

FLASH-RF v6 statistically significantly outperforms the Random Search baseline on all **8/8** systems.

| System | RS Median | v6 Median | Improvement | Hit-Optimal | Significance |
|--------|----------|----------|-------------|------------|--------------|
| 7z | 4,576.3 | **4,305.3** | +5.9% | 5/30 | ** |
| Apache | 31.09 | **30.74** | +1.1% | 30/30 | *** |
| brotli | 1.472 | **1.46** | +0.8% | 30/30 | *** |
| LLVM | 59,141.5 | **52,285.4** | +11.6% | 30/30 | *** |
| PostgreSQL | 46,059.6 | **45,939.8** | +0.3% | 9/30 | *** |
| spear | 0.00099 | **0.000** | +100% | 27/30 | *** |
| storm | 0.000 | **0.000** | 0.0% | 30/30 | *** |
| x264 | 22.759 | **21.556** | +5.3% | 20/30 | *** |

**Total hit-optimal: 181/240**

## Multi-Baseline Comparison (R=100, total hits/240)

| RS | HC | SA | BestConfig | TPE | **v6** |
|----|----|----|-----------|-----|--------|
| 35 | 104 | 67 | 53 | 121 | **181** |

v6 significantly better than: RS (8/8), SA (8/8), BestConfig (8/8), HC (6/8), TPE (5/8).

## References

- [1] P. Jamshidi et al., "Transfer learning for performance modeling of configurable systems: An exploratory analysis," *Proc. ASE*, 2017.
- [2] V. Nair, T. Menzies, N. Siegmund, and S. Apel, "Using bad learners to find good configurations," *Proc. ESEC/FSE*, 2017, pp. 257–267. *(FLASH)*
- [5] N. Siegmund et al., "Performance-influence models for highly configurable systems," *Proc. ESEC/FSE*, 2015.
- [8] F. Hutter, H. H. Hoos, and K. Leyton-Brown, "Sequential model-based optimization for general algorithm configuration," *Proc. LION*, 2011. *(SMAC)*
- [9] J. Bergstra et al., "Algorithms for hyper-parameter optimization," *Proc. NIPS*, 2011. *(TPE)*
- [10] W. R. Thompson, "On the likelihood that one unknown probability exceeds another in view of the evidence of two samples," *Biometrika*, 1933.
- [12] L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

Full reference list available in [report_long.md](report_long.md).
