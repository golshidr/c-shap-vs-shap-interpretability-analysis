"""Experiment 5 - runtime scaling, measured sequentially on one core.

* Kernel SHAP seconds per explained instance as a function of background size
  k (up to the full reference background), for every dataset.
* K-means fit time as a function of k and of the number of rows clustered.

Run this experiment on its own (``cshap run runtime``) so timings are not
distorted by other experiments sharing the CPU. BLAS threads are pinned to 1
by the CLI for this experiment.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from ..backgrounds import kmeans_background, random_background
from ..data import prepare_data
from ..explain import kernel_shap
from .common import build_setup, out_dir, save_table


def run(cfg: dict) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    rc = cfg["runtime"]
    rows = []
    for ds in cfg["datasets"]:
        s = build_setup(ds, rc["model"], cfg, "runtime", with_reference=False)
        X = s.X_explain[: rc["n_explain"]]
        n_ref = len(s.data.X_ref)
        sizes = sorted({k for k in rc["sizes"] if k < n_ref} | {n_ref})
        for k in sizes:
            bg = random_background(s.data.X_ref, k, seed=0)
            res = kernel_shap(s.f, bg, X, nsamples="auto", l1_reg=False, seed=0, n_jobs=1)
            km_fit = kmeans_background(s.data.X_ref, k, seed=0).fit_seconds if k < n_ref else 0.0
            rows.append({"kind": "kernel", "dataset": ds, "model": rc["model"], "k": k,
                         "n_features": s.data.n_features, "full_background": k == n_ref,
                         "explain_s_per_instance": float(np.mean(res.per_instance_seconds)),
                         "kmeans_fit_s": km_fit})
    # K-means fit time vs number of rows (MAGIC has the most rows)
    d = prepare_data("magic", max_reference=None, data_dir=cfg["data_dir"]) if "magic" in cfg["datasets"] \
        else prepare_data(cfg["datasets"][0], max_reference=None, data_dir=cfg["data_dir"])
    for n in rc["kmeans_rows"]:
        if n > len(d.X_train):
            continue
        Xn = d.X_train[np.random.default_rng(0).choice(len(d.X_train), n, replace=False)]
        for k in (20, 100):
            t0 = time.perf_counter()
            kmeans_background(Xn, k, seed=0)
            rows.append({"kind": "kmeans_rows", "dataset": d.name, "k": k, "n_rows": n,
                         "n_features": d.n_features, "kmeans_fit_s": time.perf_counter() - t0})
    df = pd.DataFrame(rows)
    save_table(df, out_dir(cfg, "runtime") / "runtime.csv")
    return df
