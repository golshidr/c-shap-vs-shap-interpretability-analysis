"""Static HTML report assembled from results/ and figures/ (no external assets)."""

from __future__ import annotations

import html
import os
from datetime import date
from pathlib import Path

import pandas as pd

from .data import DATASETS
from .models import MODEL_LABELS

CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --rule:#e1e0d9; --bg:#fcfcfb; --accent:#2a78d6; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --rule:#2c2c2a; --bg:#1a1a19; --accent:#3987e5; }
  :root:not([data-theme="light"]) img { background:#fff; }
}
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1080px; margin:0 auto; padding:24px 16px 64px; }
h1 { font-size:1.7rem; margin:0.2em 0; } h2 { font-size:1.25rem; margin-top:2.2em; border-bottom:1px solid var(--rule); padding-bottom:4px; }
h3 { font-size:1.05rem; margin-top:1.6em; }
p, li { color:var(--ink2); max-width:78ch; }
.meta { color:var(--muted); font-size:0.9rem; }
figure { margin:1.2em 0; } figure img { max-width:100%; height:auto; border:1px solid var(--rule); border-radius:4px; }
figcaption { color:var(--muted); font-size:0.88rem; margin-top:4px; }
.table-wrap { overflow-x:auto; }
table { border-collapse:collapse; font-size:0.86rem; font-variant-numeric:tabular-nums; margin:0.6em 0; }
th, td { padding:4px 10px; border-bottom:1px solid var(--rule); text-align:right; white-space:nowrap; }
th { color:var(--ink2); font-weight:600; } td:first-child, th:first-child { text-align:left; }
nav a { color:var(--accent); margin-right:14px; text-decoration:none; font-size:0.92rem; }
code { font-size:0.9em; }
"""


def _table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    return '<div class="table-wrap">' + df.to_html(index=False, escape=True, border=0) + "</div>"


def _fig(figdir_rel: str, name: str, caption: str, figdir: Path) -> str:
    if not (figdir / name).exists():
        return ""
    return (f'<figure><img src="{figdir_rel}/{name}" alt="{html.escape(caption)}" loading="lazy">'
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _nice(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "dataset" in out:
        out["dataset"] = out["dataset"].map(lambda d: DATASETS[d].label if d in DATASETS else d)
    if "model" in out:
        out["model"] = out["model"].map(lambda m: MODEL_LABELS.get(m, m))
    return out


def build_report(cfg: dict) -> Path:
    res = Path(cfg["results_dir"])
    figdir = Path(cfg["figures_dir"])
    out = Path(cfg.get("reports_dir", "reports")) / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(figdir, out.parent)
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>C-SHAP vs SHAP Report</title><style>{CSS}</style></head><body><main>",
             "<h1>C-SHAP vs SHAP: evaluation report</h1>",
             f"<p class='meta'>Generated {date.today().isoformat()} by <code>cshap report</code> from "
             f"<code>{html.escape(str(res))}/</code>. Every number below is read from the result CSVs.</p>",
             "<nav><a href='#rep'>Replication</a><a href='#sweep'>Compression error</a>"
             "<a href='#kernel'>Kernel SHAP</a><a href='#faith'>Faithfulness</a>"
             "<a href='#stab'>Stability</a><a href='#runtime'>Runtime</a></nav>"]

    p = res / "replication" / "summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        cols = ["l1_reg", "label", "background_rows", "total_s", "nonzero_per_instance",
                "vs_exact_inst_rel_l1", "vs_exact_inst_spearman", "vs_exact_global_pearson",
                "nb_pearson_standard_vs_cshap"]
        parts += ["<h2 id='rep'>1. Replication of the original notebook</h2>",
                  "<p>Breast Cancer, random forest, 20 test instances, Kernel SHAP on "
                  "<code>predict_proba</code>. <code>vs_exact</code> columns compare against exact "
                  "interventional SHAP with the full training background.</p>",
                  _table(s[cols]),
                  _fig(rel, "fig01_replication_runtime.png", "Runtime of the three backgrounds.", figdir),
                  _fig(rel, "fig02_replication_global_importance.png", "Global importance.", figdir),
                  _fig(rel, "fig03_replication_local_explanation.png", "Local explanation, test instance 0.", figdir)]

    p = res / "sweep" / "summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        piv = s[s["k"] == 20].pivot_table(index=["dataset", "model"], columns="method",
                                          values="rel_l1_mean").reset_index()
        tests = pd.read_csv(res / "sweep" / "tests.csv")
        tt = tests[tests["k"] == 20][["dataset", "model", "method_a", "method_b", "mean_diff",
                                      "frac_a_better", "rank_biserial", "p_holm"]]
        parts += ["<h2 id='sweep'>2. Background-compression error (exact SHAP)</h2>",
                  "<p>Mean per-instance relative L1 error at k = 20 (100 instances x 10 seeds):</p>",
                  _table(_nice(piv)),
                  "<h3>Paired Wilcoxon tests at k = 20 (Holm-adjusted over all k)</h3>",
                  _table(_nice(tt), "{:.3g}"),
                  _fig(rel, "fig04_sweep_error_vs_k.png", "Error vs background size.", figdir),
                  _fig(rel, "fig05_sweep_rank_vs_k.png", "Rank agreement vs background size.", figdir),
                  _fig(rel, "fig06_background_mean_shift.png", "Shift of the background mean.", figdir)]

    p = res / "kernel" / "summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        sub = s[s["k"] == 20][["dataset", "model", "method", "explain_s", "fit_s", "total_rel_l1",
                               "background_rel_l1", "sampling_rel_l1", "total_spearman"]]
        parts += ["<h2 id='kernel'>3. End-to-end Kernel SHAP</h2>",
                  "<p>k = 20, 50 instances x 3 seeds. <code>explain_s</code> is CPU seconds per instance "
                  "measured while other jobs shared the machine.</p>",
                  _table(_nice(sub)),
                  _fig(rel, "fig07_error_decomposition.png", "Error decomposition.", figdir),
                  _fig(rel, "fig08_runtime_vs_error.png", "Runtime vs error.", figdir)]

    p = res / "faithfulness" / "summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        piv = s.pivot_table(index=["dataset", "model"], columns="method", values="abpc").reset_index()
        parts += ["<h2 id='faith'>4. Deletion / insertion faithfulness</h2>",
                  "<p>Insertion AUC minus deletion AUC (higher is better):</p>", _table(_nice(piv)),
                  _fig(rel, "fig09_faithfulness_curves.png", "Faithfulness curves.", figdir),
                  _fig(rel, "fig10_faithfulness_summary.png", "Faithfulness summary.", figdir)]

    p1, p2 = res / "sweep" / "stability.csv", res / "stability" / "kernel_stability.csv"
    if p1.exists():
        parts += ["<h2 id='stab'>5. Stability across seeds</h2>",
                  _fig(rel, "fig11_stability_background.png", "Background-only stability.", figdir)]
        if p2.exists():
            s = pd.read_csv(p2)
            piv = s[s["k"] == 20].pivot_table(index=["dataset", "model"], columns="method",
                                              values="inst_spearman").reset_index()
            parts += ["<p>End-to-end Kernel SHAP, k = 20, mean pairwise per-instance Spearman:</p>",
                      _table(_nice(piv)),
                      _fig(rel, "fig12_stability_kernel.png", "End-to-end stability.", figdir)]

    p = res / "runtime" / "runtime.csv"
    if p.exists():
        r = pd.read_csv(p)
        kr = r[r["kind"] == "kernel"][["dataset", "k", "n_features", "full_background",
                                       "explain_s_per_instance", "kmeans_fit_s"]]
        parts += ["<h2 id='runtime'>6. Runtime scaling (single core)</h2>", _table(_nice(kr)),
                  _fig(rel, "fig13_runtime_scaling.png", "Runtime scaling.", figdir)]

    parts.append("</main></body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
