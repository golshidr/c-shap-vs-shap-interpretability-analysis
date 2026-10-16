"""Experiment configuration (YAML) with defaults."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "results_dir": "results",
    "figures_dir": "figures",
    "reports_dir": "reports",
    "data_dir": "data/raw",
    "n_jobs": 8,
    "datasets": ["breast_cancer", "magic", "spambase"],
    "models": ["rf", "gbm", "logreg"],
    "split": {"test_size": 0.2, "seed": 42},
    "replicate": {
        "dataset": "breast_cancer",
        "n_explain": 20,
        "standard_size": 100,
        "k": 20,
        "seed": 42,
        "l1_settings": ["num_features(10)", False],
    },
    "sweep": {
        "max_reference": 1000,
        "n_explain": 100,
        "sizes": [5, 10, 20, 50, 100, 200],
        "methods": ["random", "kmeans", "kmeans_weighted"],
        "seeds": 10,
    },
    "kernel": {
        "max_reference": 1000,
        "n_explain": 50,
        "sizes": [10, 20, 50, 100],
        "methods": ["random", "kmeans", "kmeans_weighted"],
        "seeds": 3,
        "nsamples": "auto",
        "l1_reg": False,
    },
    "faithfulness": {"size": 20, "standard_size": 100, "n_draws": 32},
    "runtime": {
        "max_reference": 2000,
        "model": "rf",
        "sizes": [5, 10, 20, 50, 100, 200, 500, 1000],
        "n_explain": 3,
        "kmeans_rows": [500, 1000, 2000, 5000, 10000],
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None, **overrides: Any) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path, encoding="utf-8") as fh:
            cfg = _merge(cfg, yaml.safe_load(fh) or {})
    return _merge(cfg, {k: v for k, v in overrides.items() if v is not None})
