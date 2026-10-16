"""Experiment 3 - deletion/insertion faithfulness of the attributions.

Uses the Kernel SHAP attributions from experiment 2 (seed 0): C-SHAP
(K-means, k=20), size-weighted K-means (k=20), 20 random rows, 100 random
rows (the notebook's "standard SHAP"), the exact full-background SHAP values
and a random feature ordering. All methods share the same imputation draws.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..faithfulness import curve_auc, random_order_phi, removal_curves
from ..stats import holm_adjust, paired_test
from .common import build_setup, out_dir, save_table
from .kernel import load_attributions


def _task(dataset: str, family: str, cfg: dict):
    warnings.filterwarnings("ignore")
    fc = cfg["faithfulness"]
    s = build_setup(dataset, family, cfg, "kernel", with_reference=False)
    att = load_attributions(cfg, dataset, family)
    k, ks = fc["size"], fc["standard_size"]
    methods = {
        "exact": att["exact"],
        "kmeans": att[f"kmeans_k{k}_s0"],
        "kmeans_weighted": att[f"kmeans_weighted_k{k}_s0"],
        "random": att[f"random_k{k}_s0"],
        f"random_k{ks}": att[f"random_k{ks}_s0"],
        "random_order": random_order_phi(*att["exact"].shape, seed=123),
    }
    inst_rows, curve_rows = [], []
    for name, phi in methods.items():
        dele, ins = removal_curves(s.f, s.X_explain, phi, s.data.X_ref, s.output_type,
                                   n_draws=fc["n_draws"], seed=2024)
        inst_rows.append(pd.DataFrame({
            "dataset": dataset, "model": family, "method": name,
            "instance": np.arange(len(phi)), "deletion_auc": curve_auc(dele),
            "insertion_auc": curve_auc(ins),
        }))
        m = dele.shape[1] - 1
        curve_rows.append(pd.DataFrame({
            "dataset": dataset, "model": family, "method": name,
            "fraction": np.arange(m + 1) / m, "deletion": dele.mean(axis=0), "insertion": ins.mean(axis=0),
        }))
    return pd.concat(inst_rows), pd.concat(curve_rows)


def run(cfg: dict) -> pd.DataFrame:
    combos = [(d, m) for d in cfg["datasets"] for m in cfg["models"]]
    res = Parallel(n_jobs=cfg["n_jobs"])(delayed(_task)(d, m, cfg) for d, m in combos)
    per_inst = pd.concat([r[0] for r in res], ignore_index=True)
    curves = pd.concat([r[1] for r in res], ignore_index=True)
    per_inst["abpc"] = per_inst["insertion_auc"] - per_inst["deletion_auc"]
    summary = (per_inst.groupby(["dataset", "model", "method"])
               [["deletion_auc", "insertion_auc", "abpc"]].mean().reset_index())
    rows = []
    for (ds, mdl), g in per_inst.groupby(["dataset", "model"]):
        for metric in ("deletion_auc", "insertion_auc"):
            piv = g.pivot_table(index="instance", columns="method", values=metric)
            for a, b in [("kmeans", "random"), ("kmeans_weighted", "random"), ("kmeans", "exact")]:
                rows.append({"dataset": ds, "model": mdl, "metric": metric, "method_a": a,
                             "method_b": b, **paired_test(piv[a].to_numpy(), piv[b].to_numpy())})
    tests = pd.DataFrame(rows)
    tests["p_holm"] = holm_adjust(tests["p_value"].to_numpy())
    d = out_dir(cfg, "faithfulness")
    per_inst.to_csv(d / "per_instance.csv.gz", index=False, float_format="%.5g")
    save_table(summary, d / "summary.csv")
    save_table(curves, d / "curves.csv")
    save_table(tests, d / "tests.csv")
    return summary
