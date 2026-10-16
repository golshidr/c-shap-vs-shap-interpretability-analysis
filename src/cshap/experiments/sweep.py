"""Experiment 1 - background-compression error measured against exact SHAP.

For every dataset x model x background size k x method x seed, compute the
*exact* interventional SHAP values with the compressed background and compare
them with the exact values for the full reference background. Because both
sides are exact, the difference is purely the effect of compressing the
background (no Kernel SHAP sampling noise). Seeds change the K-means
initialisation (``random_state``) and the random subsample.

Outputs (results/sweep/): per_instance.csv.gz (errors averaged over seeds),
per_seed.csv, summary.csv, stability.csv, tests.csv, meta.csv.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..backgrounds import make_background
from ..metrics import global_metrics, instance_metrics, pairwise_stability
from ..reference import exact_shap
from ..stats import holm_adjust, paired_test
from .common import build_setup, out_dir, save_table

INST_METRICS = ("rel_l1", "mae", "spearman", "topk", "sign_agree")


def _reference_task(dataset: str, family: str, cfg: dict):
    warnings.filterwarnings("ignore")
    s = build_setup(dataset, family, cfg, "sweep")
    eff = np.abs(s.ref_values.sum(axis=1) + s.ref_base - s.f(s.X_explain))
    meta = {
        "dataset": dataset, "model": family, "accuracy": s.accuracy,
        "n_features": s.data.n_features, "n_train": len(s.data.X_train),
        "n_reference": len(s.data.X_ref), "n_explain": len(s.X_explain),
        "reference_s": s.ref_seconds, "ref_efficiency_max": float(eff.max()),
        "ref_efficiency_median": float(np.median(eff)),
        "ref_mean_abs_attr": float(np.abs(s.ref_values).mean()),
    }
    return (dataset, family), s.ref_values, s.ref_base, meta


def _seed_task(dataset: str, family: str, seed: int, cfg: dict, ref: np.ndarray, ref_base: float):
    warnings.filterwarnings("ignore")
    sc = cfg["sweep"]
    s = build_setup(dataset, family, cfg, "sweep", with_reference=False)
    ref_mean = s.data.X_ref.mean(axis=0)
    inst_rows, seed_rows, phis = [], [], {}
    for k in sc["sizes"]:
        for method in sc["methods"]:
            bg = make_background(method, s.data.X_ref, k, seed)
            phi, base = exact_shap(s.model, family, s.X_explain, bg)
            im = instance_metrics(phi, ref, k=5)
            gm = global_metrics(phi, ref, k=10)
            phis[(method, k)] = phi.astype(np.float32)
            inst_rows.append(pd.DataFrame({
                "dataset": dataset, "model": family, "seed": seed, "method": method, "k": k,
                "instance": np.arange(len(phi)), **{m: im[m] for m in INST_METRICS},
            }))
            seed_rows.append({
                "dataset": dataset, "model": family, "seed": seed, "method": method, "k": k,
                "fit_s": bg.fit_seconds, "base_value_err": abs(base - ref_base),
                "mean_shift": float(np.linalg.norm(bg.mean() - ref_mean) / np.sqrt(len(ref_mean))),
                **{f"inst_{m}": float(np.nanmean(im[m])) for m in INST_METRICS},
                **{f"global_{key}": val for key, val in gm.items()},
            })
    return pd.concat(inst_rows), pd.DataFrame(seed_rows), (dataset, family, seed, phis)


def _summarize(per_inst: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in per_inst.groupby(["dataset", "model", "method", "k"]):
        row = dict(zip(["dataset", "model", "method", "k"], key, strict=True))
        for m in INST_METRICS:
            v = g[m].dropna().to_numpy()
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            row[f"{m}_mean"] = v.mean()
            row[f"{m}_ci95"] = 1.96 * se
        rows.append(row)
    return pd.DataFrame(rows)


def significance_tests(per_inst: pd.DataFrame, metric: str = "rel_l1") -> pd.DataFrame:
    """Paired Wilcoxon tests on per-instance errors (averaged over seeds), Holm-adjusted."""
    pairs = [("kmeans", "random"), ("kmeans_weighted", "random"), ("kmeans_weighted", "kmeans")]
    rows = []
    for (ds, mdl, k), g in per_inst.groupby(["dataset", "model", "k"]):
        piv = g.pivot_table(index="instance", columns="method", values=metric)
        for a, b in pairs:
            if a in piv and b in piv:
                res = paired_test(piv[a].to_numpy(), piv[b].to_numpy())
                rows.append({"dataset": ds, "model": mdl, "k": k, "metric": metric,
                             "method_a": a, "method_b": b, **res})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
        out["verdict"] = np.where(out["p_holm"] >= 0.05, "no significant difference",
                                  np.where(out["mean_diff"] < 0, "a lower error", "b lower error"))
    return out


def run(cfg: dict) -> pd.DataFrame:
    sc = cfg["sweep"]
    n_jobs = cfg["n_jobs"]
    combos = [(d, m) for d in cfg["datasets"] for m in cfg["models"]]
    refs = Parallel(n_jobs=n_jobs)(delayed(_reference_task)(d, m, cfg) for d, m in combos)
    ref_map = {key: (vals, base) for key, vals, base, _ in refs}
    meta = pd.DataFrame([r[3] for r in refs])
    tasks = [(d, m, s) for d, m in combos for s in range(sc["seeds"])]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_seed_task)(d, m, s, cfg, *ref_map[(d, m)]) for d, m, s in tasks
    )
    per_inst_all = pd.concat([r[0] for r in results], ignore_index=True)
    per_seed = pd.concat([r[1] for r in results], ignore_index=True)
    per_inst = (per_inst_all.groupby(["dataset", "model", "method", "k", "instance"], as_index=False)
                [list(INST_METRICS)].mean())
    # stability across seeds of the background-induced attributions
    by_cfg: dict = {}
    for d, m, _seed, phis in (r[2] for r in results):
        for (method, k), phi in phis.items():
            by_cfg.setdefault((d, m, method, k), []).append(phi)
    stab = pd.DataFrame([
        {"dataset": d, "model": m, "method": method, "k": k, **pairwise_stability(runs, k=5)}
        for (d, m, method, k), runs in by_cfg.items()
    ])
    d = out_dir(cfg, "sweep")
    save_table(meta, d / "meta.csv")
    per_inst.to_csv(d / "per_instance.csv.gz", index=False, float_format="%.5g")
    save_table(per_seed, d / "per_seed.csv")
    save_table(_summarize(per_inst), d / "summary.csv")
    save_table(stab, d / "stability.csv")
    save_table(significance_tests(per_inst, "rel_l1"), d / "tests.csv")
    return per_seed
