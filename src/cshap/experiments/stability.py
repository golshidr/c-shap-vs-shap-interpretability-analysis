"""Experiment 4 - run-to-run stability of end-to-end Kernel SHAP explanations.

Each seed changes both the background (K-means initialisation or random
subsample) and Kernel SHAP's coalition sampling. Agreement between seeds is
measured with per-instance Spearman correlation, top-5 overlap and the
relative standard deviation of attributions. (Background-only stability,
without sampling noise, is computed in experiment 1: results/sweep/stability.csv.)
"""

from __future__ import annotations

import pandas as pd

from ..metrics import pairwise_stability
from .common import out_dir, save_table
from .kernel import load_attributions


def run(cfg: dict) -> pd.DataFrame:
    kc = cfg["kernel"]
    rows = []
    for ds in cfg["datasets"]:
        for mdl in cfg["models"]:
            att = load_attributions(cfg, ds, mdl)
            for method in kc["methods"]:
                for k in kc["sizes"]:
                    runs = [att[f"{method}_k{k}_s{s}"] for s in range(kc["seeds"])]
                    rows.append({"dataset": ds, "model": mdl, "method": method, "k": k,
                                 **pairwise_stability(runs, k=5)})
    df = pd.DataFrame(rows)
    save_table(df, out_dir(cfg, "stability") / "kernel_stability.csv")
    return df
