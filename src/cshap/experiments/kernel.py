"""Experiment 2 - end-to-end Kernel SHAP: runtime vs. accuracy, with error decomposition.

For each dataset x model x k x method x seed, run Kernel SHAP with the
compressed background and record per-instance runtime and three errors:

* ``total``      - Kernel SHAP(B) vs exact SHAP(full reference)   (what a user gets)
* ``background`` - exact SHAP(B)  vs exact SHAP(full reference)   (compression bias)
* ``sampling``   - Kernel SHAP(B) vs exact SHAP(B)                (Monte Carlo noise)

Kernel SHAP attributions are saved to results/kernel/attributions/ (git-ignored)
for the faithfulness and stability analyses.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..backgrounds import make_background
from ..explain import kernel_shap
from ..metrics import instance_metrics
from ..reference import exact_shap
from ..stats import holm_adjust, paired_test
from .common import build_setup, out_dir, save_table
from .sweep import _reference_task


def _task(dataset, family, seed, method, cfg, ref, ref_base):
    warnings.filterwarnings("ignore")
    kc = cfg["kernel"]
    s = build_setup(dataset, family, cfg, "kernel", with_reference=False)
    rows, phis = [], {}
    for k in kc["sizes"]:
        bg = make_background(method, s.data.X_ref, k, seed)
        res = kernel_shap(s.f, bg, s.X_explain, nsamples=kc["nsamples"], l1_reg=kc["l1_reg"],
                          seed=seed, n_jobs=1)
        exB, _ = exact_shap(s.model, family, s.X_explain, bg)
        tot = instance_metrics(res.values, ref, k=5)
        bgd = instance_metrics(exB, ref, k=5)
        smp = instance_metrics(res.values, exB, k=5)
        phis[f"{method}_k{k}_s{seed}"] = res.values.astype(np.float32)
        rows.append(pd.DataFrame({
            "dataset": dataset, "model": family, "seed": seed, "method": method, "k": k,
            "instance": np.arange(len(res.values)), "explain_s": res.per_instance_seconds,
            "fit_s": bg.fit_seconds,
            "total_rel_l1": tot["rel_l1"], "background_rel_l1": bgd["rel_l1"],
            "sampling_rel_l1": smp["rel_l1"], "total_spearman": tot["spearman"],
            "total_top5": tot["topk"], "total_mae": tot["mae"],
        }))
    return pd.concat(rows), (dataset, family, phis)


def run(cfg: dict) -> pd.DataFrame:
    kc = cfg["kernel"]
    n_jobs = cfg["n_jobs"]
    combos = [(d, m) for d in cfg["datasets"] for m in cfg["models"]]
    kcfg = {**cfg, "sweep": {**cfg["sweep"], "max_reference": kc["max_reference"],
                             "n_explain": kc["n_explain"]}}
    refs = Parallel(n_jobs=n_jobs)(delayed(_reference_task)(d, m, kcfg) for d, m in combos)
    ref_map = {key: (vals, base) for key, vals, base, _ in refs}
    tasks = [(d, m, s, meth) for d, m in combos for s in range(kc["seeds"]) for meth in kc["methods"]]
    # heaviest first for better load balancing
    weight = {"rf": 3, "gbm": 1, "logreg": 1}
    tasks.sort(key=lambda t: -weight.get(t[1], 1))
    results = Parallel(n_jobs=n_jobs)(
        delayed(_task)(d, m, s, meth, cfg, *ref_map[(d, m)]) for d, m, s, meth in tasks
    )
    per_inst = pd.concat([r[0] for r in results], ignore_index=True)
    d = out_dir(cfg, "kernel")
    att_dir = d / "attributions"
    att_dir.mkdir(exist_ok=True)
    store: dict = {}
    for ds, m, phis in (r[1] for r in results):
        store.setdefault((ds, m), {}).update(phis)
    for (ds, m), phis in store.items():
        np.savez_compressed(att_dir / f"{ds}_{m}.npz", exact=ref_map[(ds, m)][0], **phis)
    per_inst.to_csv(d / "per_instance.csv.gz", index=False, float_format="%.5g")
    summary = (per_inst.groupby(["dataset", "model", "method", "k"])
               .agg(explain_s=("explain_s", "mean"), fit_s=("fit_s", "mean"),
                    total_rel_l1=("total_rel_l1", "mean"), background_rel_l1=("background_rel_l1", "mean"),
                    sampling_rel_l1=("sampling_rel_l1", "mean"), total_spearman=("total_spearman", "mean"),
                    total_top5=("total_top5", "mean"))
               .reset_index())
    save_table(summary, d / "summary.csv")
    save_table(significance_tests(per_inst), d / "tests.csv")
    return summary


def significance_tests(per_inst: pd.DataFrame, metric: str = "total_rel_l1") -> pd.DataFrame:
    g0 = per_inst.groupby(["dataset", "model", "method", "k", "instance"], as_index=False)[metric].mean()
    rows = []
    for (ds, mdl, k), g in g0.groupby(["dataset", "model", "k"]):
        piv = g.pivot_table(index="instance", columns="method", values=metric)
        for a, b in [("kmeans", "random"), ("kmeans_weighted", "random"), ("kmeans_weighted", "kmeans")]:
            if a in piv and b in piv:
                rows.append({"dataset": ds, "model": mdl, "k": k, "metric": metric,
                             "method_a": a, "method_b": b, **paired_test(piv[a].to_numpy(), piv[b].to_numpy())})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm_adjust(out["p_value"].to_numpy())
    return out


def load_attributions(cfg: dict, dataset: str, family: str) -> dict[str, np.ndarray]:
    path = Path(cfg["results_dir"]) / "kernel" / "attributions" / f"{dataset}_{family}.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
