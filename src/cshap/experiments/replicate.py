"""Experiment 0 - re-run the original notebook comparison, then check it against exact SHAP.

Faithful to notebooks/01_original_c_shap_vs_shap.ipynb:
Breast Cancer, stratified 80/20 split (seed 42), StandardScaler, random forest
(200 trees, seed 42), first 20 test rows explained with
``shap.KernelExplainer(model.predict_proba, background)``;
"standard SHAP" = ``shap.sample(X_train, 100, random_state=42)``;
"C-SHAP" = ``KMeans(20, random_state=42, n_init=10)`` centroids.

Additions: a size-matched control (20 random rows), the exact interventional
SHAP values for the full training background as ground truth, explicit
seeding of Kernel SHAP's coalition sampling, and both the shap>=0.47 default
``l1_reg="num_features(10)"`` and ``l1_reg=False``.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import shap

from ..backgrounds import kmeans_background, random_background
from ..metrics import global_importance, global_metrics, instance_metrics
from .common import build_setup, out_dir, save_table, write_json

LABELS = {"standard": "Standard SHAP (100 random rows)", "cshap": "C-SHAP (20 K-means centroids)",
          "random20": "20 random rows (size-matched control)"}


def _l1_name(l1) -> str:
    return "shap-default" if l1 == "num_features(10)" else ("none" if l1 is False else str(l1))


def run(cfg: dict) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    rc = cfg["replicate"]
    cfg = {**cfg, "replicate": {**rc, "max_reference": None}}
    s = build_setup(rc["dataset"], "rf", cfg, "replicate")
    X_train = s.data.X_train
    backgrounds = {
        "standard": random_background(X_train, rc["standard_size"], seed=rc["seed"]),
        "cshap": kmeans_background(X_train, rc["k"], seed=rc["seed"], weighted=False),
        "random20": random_background(X_train, rc["k"], seed=rc["seed"]),
    }
    rows, phis = [], {}
    for l1 in rc["l1_settings"]:
        for name, bg in backgrounds.items():
            np.random.seed(0)
            t0 = time.perf_counter()
            explainer = shap.KernelExplainer(s.model.predict_proba, bg.data)
            vals = explainer.shap_values(s.X_explain, l1_reg=l1, silent=True)
            secs = time.perf_counter() - t0
            phi = np.asarray(vals)[:, :, 1]
            phis[(_l1_name(l1), name)] = phi
            im = instance_metrics(phi, s.ref_values, k=5)
            gm = global_metrics(phi, s.ref_values, k=10)
            rows.append({
                "l1_reg": _l1_name(l1), "method": name, "label": LABELS[name],
                "background_rows": bg.size, "background_fit_s": bg.fit_seconds,
                "explain_s": secs, "total_s": secs + bg.fit_seconds,
                "nonzero_per_instance": float(np.mean((np.abs(phi) > 0).sum(axis=1))),
                "vs_exact_inst_rel_l1": float(np.nanmean(im["rel_l1"])),
                "vs_exact_inst_spearman": float(np.nanmean(im["spearman"])),
                "vs_exact_inst_top5": float(np.mean(im["topk"])),
                "vs_exact_global_pearson": gm["pearson"],
                "vs_exact_global_spearman": gm["spearman"],
                "vs_exact_global_top10": gm["top10"],
            })
    df = pd.DataFrame(rows)
    # the two statistics the notebook reported: standard vs C-SHAP global importance
    for l1n in df["l1_reg"].unique():
        a, b = phis[(l1n, "standard")], phis[(l1n, "cshap")]
        ga, gb = global_importance(a), global_importance(b)
        df.loc[df["l1_reg"] == l1n, "nb_pearson_standard_vs_cshap"] = float(np.corrcoef(ga, gb)[0, 1])
        df.loc[df["l1_reg"] == l1n, "nb_mad_standard_vs_cshap"] = float(np.mean(np.abs(ga - gb)))
    d = out_dir(cfg, "replication")
    save_table(df, d / "summary.csv")
    for l1n in df["l1_reg"].unique():
        g = pd.DataFrame({"feature": s.data.feature_names, "exact_full_background": global_importance(s.ref_values)})
        loc = pd.DataFrame({"feature": s.data.feature_names, "exact_full_background": s.ref_values[0]})
        for name in backgrounds:
            g[name] = global_importance(phis[(l1n, name)])
            loc[name] = phis[(l1n, name)][0]
        save_table(g.sort_values("exact_full_background", ascending=False), d / f"global_importance_l1-{l1n}.csv")
        save_table(loc, d / f"local_instance0_l1-{l1n}.csv")
    write_json(d / "meta.json", {"accuracy": s.accuracy, "n_explain": len(s.X_explain),
                                 "n_train": len(X_train), "shap_version": shap.__version__,
                                 "exact_reference_s": s.ref_seconds})
    return df
