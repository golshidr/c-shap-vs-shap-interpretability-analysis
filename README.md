# C-SHAP vs SHAP: Interpretability Analysis

[![CI](https://github.com/golshidr/c-shap-vs-shap-interpretability-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/golshidr/c-shap-vs-shap-interpretability-analysis/actions/workflows/ci.yml)
[![Python 3.11 | 3.12 | 3.13](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Can Kernel SHAP be made faster by replacing its background data with K-means
centroids, as in C-SHAP, without changing the explanations? This repository
answers that question with `cshap`, a reusable Python package and command-line
tool for centroid backgrounds, and an evaluation of the method against exact
SHAP values on three public datasets and three model families, with
size-matched random controls, ten seeds and significance tests.

<p align="center">
  <img src="figures/fig04_sweep_error_vs_k.png" width="860" alt="Error against exact SHAP versus background size k for C-SHAP (equal-weight K-means centroids), size-weighted centroids and random subsamples, on three datasets and three models. Random subsamples improve steadily with k; equal-weight centroids barely improve on Breast Cancer and MAGIC; size-weighted centroids are best there and exact for logistic regression, but worst for the tree models on Spambase.">
</p>
<p align="center"><em>Error against exact SHAP as the background grows (mean over 100 test instances and 10 seeds, 95% CI).</em></p>

## Key findings

"Error" is the per-instance relative L1 distance to exact SHAP values computed
with the full reference background (lower is better).

- **The speed-up comes from background size, not clustering.** The notebook's comparison reproduces: on Breast Cancer, 20 K-means centroids explained 20 instances in 3.18 s, against 15.86 s for 100 random rows. A random background of 20 rows was just as fast (3.19 s), because Kernel SHAP cost is linear in background size.
- **At equal size, equal-weight centroids are usually less accurate than a random subsample.** C-SHAP had significantly lower error in 8 of 54 dataset × model × size settings, all at k ≤ 20, and significantly higher error in the other 46 (paired Wilcoxon, Holm-adjusted, 100 instances × 10 seeds). Giving every centroid the same weight shifts the background mean by 0.50–0.73 standard deviations at k = 20.
- **Size-weighted centroids fix this on two of the three datasets.** Weighting centroids by cluster size keeps the background mean, so it is exact for logistic regression. It had the lowest error on Breast Cancer and MAGIC and beat random in 39 of 54 settings, but on Spambase, with sparse and heavy-tailed word frequencies, it was the worst option for both tree models.
- **The background, not Kernel SHAP's sampling, sets the error.** Kernel SHAP's own Monte Carlo error was below 0.001 on MAGIC and for logistic regression and 0.02–0.08 elsewhere. At k = 20 the background contributed 0.08–1.13, except for size-weighted centroids on logistic regression, which are exact. A random 20-row background was as fast as C-SHAP and had lower error in all 9 dataset × model pairs.
- **Centroid backgrounds are stable, but C-SHAP is consistently shifted.** At k = 20 the seed-to-seed rank correlation was 0.90–0.99 for C-SHAP, 0.95–1.00 for weighted centroids and 0.63–0.93 for random subsamples (exact SHAP, 10 seeds).

Full tables, significance tests, faithfulness results and four shap library
pitfalls found along the way are in [docs/RESULTS.md](docs/RESULTS.md).

## What's inside

- **`CShapExplainer` API** with K-means (C-SHAP), size-weighted centroid and random backgrounds, and seeded, parallel Kernel SHAP.
- **`cshap` CLI** that explains your own joblib-saved scikit-learn model on CSV data, or compares C-SHAP with random backgrounds for it.
- **Exact SHAP references**: interventional Tree SHAP for tree ensembles and the closed form for logistic regression, used as ground truth.
- **Benchmarks** of accuracy, deletion/insertion faithfulness, seed stability and runtime, with Wilcoxon tests, bootstrap CIs and Holm correction.
- **Committed results** (CSV), figures and a static HTML report.
- **Tests and CI**: a pytest suite plus lint, tests and a smoke run on Python 3.11–3.13 in GitHub Actions.

## Quickstart

```bash
git clone https://github.com/golshidr/c-shap-vs-shap-interpretability-analysis.git
cd c-shap-vs-shap-interpretability-analysis
python3 -m venv .venv && source .venv/bin/activate        # Python 3.11-3.13
pip install -r requirements.txt && pip install -e . --no-deps
```

For development, also run `pip install -r requirements-dev.txt`, then `pytest`.

Explain a model from Python:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from cshap import CShapExplainer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)

f = lambda X: model.predict_proba(X)[:, 1]                 # any scalar-output function
explainer = CShapExplainer(f, n_clusters=20).fit(X_train)  # weighted=True: size-weighted centroids
result = explainer.explain(X_test[:5])                     # seeded Kernel SHAP, l1_reg=False
print(result.values.shape, result.base_value, result.seconds)
```

Or from the command line, with a joblib-saved model and CSV files:

```bash
cshap explain --model model.joblib --background train.csv --data new_rows.csv --k 20 --drop-columns target --output attributions.csv
```

`cshap compare` takes the same inputs and reports how C-SHAP and random
backgrounds of size k agree with a larger random reference.

## Reproducing the results

```bash
make install      # pinned dependencies and the package
make reproduce    # downloads the data, runs all experiments, rebuilds figures/ and reports/index.html
```

A full run takes about an hour on 20 CPU cores. `make smoke` runs a
Breast Cancer-only version in about a minute, the same one CI runs. Per-step
commands and timings are in [docs/RESULTS.md](docs/RESULTS.md#reproducing-the-results).

## Documentation

- [docs/RESULTS.md](docs/RESULTS.md): method, all results and tables, implementation notes, limitations.
- [reports/index.html](reports/index.html): HTML report with every figure (open it in a browser after cloning).
- [notebooks/02_cshap_package_walkthrough.ipynb](notebooks/02_cshap_package_walkthrough.ipynb): walkthrough of the package against exact SHAP.
- [notebooks/01_original_c_shap_vs_shap.ipynb](notebooks/01_original_c_shap_vs_shap.ipynb) and [docs/business_insights.md](docs/business_insights.md): the original analysis.
- [CHANGELOG.md](CHANGELOG.md): release notes.

## Project structure

```
├── src/cshap/           package: backgrounds, Kernel SHAP, exact references, metrics, CLI
│   └── experiments/     replicate, sweep, kernel, faithfulness, stability, runtime
├── configs/             default.yaml (full study), smoke.yaml (CI)
├── notebooks/           original notebook and package walkthrough
├── results/             CSV outputs of every experiment
├── figures/             generated figures; figures/original/ holds the notebook's figures
├── reports/index.html   HTML report built from results/
├── docs/                detailed results and the original notebook's insights
├── scripts/             dataset download
└── tests/               pytest suite
```

## Data

- Breast Cancer Wisconsin (Diagnostic): bundled with scikit-learn; [UCI 17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), doi:10.24432/C5DW2B, CC BY 4.0.
- MAGIC Gamma Telescope: [OpenML 1120](https://www.openml.org/d/1120), mirror of [UCI 159](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), doi:10.24432/C52C8B, CC BY 4.0 (UCI).
- Spambase: [OpenML 44](https://www.openml.org/d/44), mirror of [UCI 94](https://archive.ics.uci.edu/dataset/94/spambase), doi:10.24432/C53G6X, CC BY 4.0 (UCI).

`make data` caches MAGIC and Spambase as CSV in `data/raw/`, which is git-ignored.

## Citation

If you use this software, please cite it using the metadata in
[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button reads it).
The idea comes from the C-SHAP paper below. This repository evaluates the
centroid-background variant implemented in the original notebook and does not
claim to reproduce the paper's exact algorithm:

> Ranjbaran, G., Recupero, D. R., Roy, C. K., & Schneider, K. A. (2025).
> C-SHAP: A Hybrid Method for Fast and Efficient Interpretability.
> *Applied Sciences*, 15(2), 672. https://doi.org/10.3390/app15020672

## License

Released under the [MIT License](LICENSE). Copyright (c) 2026 Golshid Ranjbaran.
