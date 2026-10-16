# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-10-16

First release. The original single-notebook comparison of standard SHAP and a
C-SHAP-style K-means background is now a tested package with a full evaluation.

### Added

- `cshap` package: random, K-means (C-SHAP) and size-weighted centroid
  backgrounds; seeded, parallel Kernel SHAP; and `CShapExplainer`, a reusable
  explainer for any scalar-output model.
- `cshap` command-line tool: `explain` and `compare` for your own joblib-saved
  model and CSV data, `run` for the experiments, and `figures` and `report` to
  rebuild the outputs.
- Exact interventional SHAP references (Tree SHAP for random forests and
  gradient boosting, closed form for logistic regression), with brute-force
  Shapley tests.
- Evaluation on three public datasets (Breast Cancer Wisconsin, MAGIC Gamma
  Telescope, Spambase) and three model families: replication of the notebook,
  background-compression sweep with 10 seeds, end-to-end Kernel SHAP with an
  error decomposition, deletion and insertion faithfulness, stability across
  seeds and runtime scaling. Paired Wilcoxon tests with Holm correction,
  bootstrap confidence intervals and size-matched random controls.
- Committed results (CSV), figures, a static HTML report and an executed
  walkthrough notebook.
- Detailed results in `docs/RESULTS.md`, pinned `requirements.txt` and
  `requirements-dev.txt`, a Makefile (`make reproduce`, `make smoke`), a pytest
  suite, GitHub Actions CI on Python 3.11, 3.12 and 3.13, and `CITATION.cff`.

### Fixed

- Original notebook: SHAP outputs were indexed as `shap_values[1]`, which with
  shap >= 0.45 selects the second *instance* rather than the positive class.
  Class 1 is now selected on the last axis (`shap_values[:, :, 1]`).
- Original notebook: the cell that wrote `requirements.txt` no longer overwrites
  the pinned file at the repository root unless explicitly enabled.

[1.0.0]: https://github.com/golshidr/c-shap-vs-shap-interpretability-analysis/releases/tag/v1.0.0
