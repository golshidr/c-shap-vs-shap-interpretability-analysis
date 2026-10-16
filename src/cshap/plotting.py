"""Publication-style figures built from the CSVs in results/.

Style: one sans-serif typeface, thin marks, hairline solid grid, colourblind-
safe categorical colours (blue / orange / aqua, validated as a three-colour
set) with distinct markers as a second encoding, and a neutral grey for the
exact reference.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import DATASETS  # noqa: E402
from .models import MODEL_LABELS  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

COLORS = {
    "kmeans": "#2a78d6",           # C-SHAP as implemented in the notebook
    "random": "#eb6834",
    "kmeans_weighted": "#1baf7a",
    "exact": INK2,
    "random_k100": "#eb6834",
    "random_order": AXIS,
}
MARKERS = {"kmeans": "o", "random": "s", "kmeans_weighted": "^", "exact": "D"}
LABELS = {
    "kmeans": "C-SHAP: K-means centroids (notebook method)",
    "kmeans_weighted": "K-means centroids, size-weighted",
    "random": "Random subsample, same size",
    "exact": "Exact SHAP, full background",
    "random_k100": "100 random rows (notebook 'standard SHAP')",
    "random_order": "Random feature order",
}
SHORT = {"kmeans": "C-SHAP (K-means)", "kmeans_weighted": "K-means, weighted",
         "random": "Random", "exact": "Exact", "random_k100": "Random 100",
         "random_order": "Random order"}
METHOD_ORDER = ["kmeans", "kmeans_weighted", "random"]


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "regular",
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "text.color": INK,
    })


def _save(fig, cfg: dict, name: str) -> Path:
    out = Path(cfg["figures_dir"]) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def _ds_label(name: str) -> str:
    info = DATASETS.get(name)
    return info.label if info else name


def _grid(cfg, sharey=False, height=2.3, sharex=True):
    ds, ms = cfg["datasets"], cfg["models"]
    fig, axes = plt.subplots(len(ms), len(ds), figsize=(3.3 * len(ds), height * len(ms) + 0.6),
                             squeeze=False, sharex=sharex, sharey=sharey)
    for j, d in enumerate(ds):
        axes[0, j].set_title(_ds_label(d))
    for i, m in enumerate(ms):
        axes[i, 0].set_ylabel(MODEL_LABELS.get(m, m) + "\n", fontsize=9.5)
    return fig, axes


def _runtime_xaxis(ax) -> None:
    """Log x-axis labelled at 1-2-5 steps; linear with 4 ticks when the range is narrow."""
    from matplotlib import ticker

    lo, hi = ax.get_xlim()
    fmt = ticker.FuncFormatter(lambda v, _: f"{v:g}")
    if hi / lo < 4:
        ax.set_xscale("linear")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    else:
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.xaxis.set_major_formatter(fmt)


def _method_legend(fig, methods=METHOD_ORDER, extra=None, ncol=3, y=1.0):
    handles = [plt.Line2D([], [], color=COLORS[m], marker=MARKERS.get(m, "o"), label=LABELS[m])
               for m in methods]
    if extra:
        handles += extra
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol)


# --------------------------------------------------------------------------- replication

def fig_replication(cfg: dict) -> list[Path]:
    d = Path(cfg["results_dir"]) / "replication"
    if not (d / "summary.csv").exists():
        return []
    s = pd.read_csv(d / "summary.csv")
    paths = []

    # runtime (as in the notebook: shap default settings)
    sub = s[s["l1_reg"] == "shap-default"].set_index("method")
    order = ["standard", "random20", "cshap"]
    names = {"standard": "Standard SHAP\n100 random rows", "random20": "Size-matched control\n20 random rows",
             "cshap": "C-SHAP\n20 K-means centroids"}
    colors = {"standard": COLORS["random"], "random20": COLORS["random"], "cshap": COLORS["kmeans"]}
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order, strict=True):
        ax.barh(yi, sub.loc[m, "explain_s"], color=colors[m], height=0.55)
        if sub.loc[m, "background_fit_s"] > 0.01:
            ax.barh(yi, sub.loc[m, "background_fit_s"], left=sub.loc[m, "explain_s"] + 0.05,
                    color=colors[m], alpha=0.45, height=0.55, hatch="////", edgecolor="white", linewidth=0)
        ax.text(sub.loc[m, "total_s"] + 0.3, yi, f"{sub.loc[m, 'total_s']:.1f} s", va="center", color=INK2)
    ax.set_yticks(y, [names[m] for m in order])
    ax.set_xlabel("Wall-clock seconds to explain 20 test instances (one core)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, sub["total_s"].max() * 1.18)
    ax.set_title("Kernel SHAP runtime on Breast Cancer / random forest (hatched: K-means fit)", loc="left")
    paths.append(_save(fig, cfg, "fig01_replication_runtime.png"))

    # global importance: shap default vs no feature selection
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, l1, title in zip(axes, ["shap-default", "none"],
                             ["shap>=0.47 default (l1_reg='num_features(10)')", "Full Kernel SHAP (l1_reg=False)"],
                             strict=True):
        g = pd.read_csv(d / f"global_importance_l1-{l1}.csv").head(10).iloc[::-1]
        yy = np.arange(len(g))
        h = 0.26
        ax.barh(yy + h, g["exact_full_background"], h, color=COLORS["exact"], label=LABELS["exact"])
        ax.barh(yy, g["standard"], h, color=COLORS["random"], label="Standard SHAP (100 random rows)")
        ax.barh(yy - h, g["cshap"], h, color=COLORS["kmeans"], label="C-SHAP (20 K-means centroids)")
        ax.set_yticks(yy, g["feature"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Mean |SHAP value| over 20 test instances")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower right")
    fig.suptitle("Top-10 global feature importance (ordered by the exact values)", x=0.01, ha="left", y=1.0)
    fig.tight_layout()
    paths.append(_save(fig, cfg, "fig02_replication_global_importance.png"))

    # local explanation for test instance 0
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, l1, title in zip(axes, ["shap-default", "none"],
                             ["shap>=0.47 default (l1_reg='num_features(10)')", "Full Kernel SHAP (l1_reg=False)"],
                             strict=True):
        loc = pd.read_csv(d / f"local_instance0_l1-{l1}.csv")
        loc = loc.reindex(loc["exact_full_background"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
        yy = np.arange(len(loc))
        h = 0.26
        ax.barh(yy + h, loc["exact_full_background"], h, color=COLORS["exact"], label=LABELS["exact"])
        ax.barh(yy, loc["standard"], h, color=COLORS["random"], label="Standard SHAP (100 random rows)")
        ax.barh(yy - h, loc["cshap"], h, color=COLORS["kmeans"], label="C-SHAP (20 K-means centroids)")
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(yy, loc["feature"])
        ax.set_title(title, loc="left")
        ax.set_xlabel("SHAP value for P(benign), test instance 0")
        ax.grid(axis="y", visible=False)
    axes[0].legend(loc="lower left")
    fig.tight_layout()
    paths.append(_save(fig, cfg, "fig03_replication_local_explanation.png"))
    return paths


# --------------------------------------------------------------------------- sweep

def _line_grid(cfg, df, metric, ylabel, fname, title, ci=None, logy=False, ylim=None, floor=1e-3):
    fig, axes = _grid(cfg)
    for i, m in enumerate(cfg["models"]):
        for j, dname in enumerate(cfg["datasets"]):
            ax = axes[i, j]
            sub = df[(df["dataset"] == dname) & (df["model"] == m)]
            clipped = []
            for meth in METHOD_ORDER:
                ss = sub[sub["method"] == meth].sort_values("k")
                if ss.empty:
                    continue
                y = ss[metric].to_numpy()
                if logy:
                    if np.all(y < floor):
                        clipped.append(meth)
                    y = np.maximum(y, floor)
                ax.plot(ss["k"], y, color=COLORS[meth], marker=MARKERS[meth], label=LABELS[meth])
                if ci and ci in ss and not (logy and meth in clipped):
                    lo = ss[metric] - ss[ci]
                    lo = np.maximum(lo, floor) if logy else lo
                    ax.fill_between(ss["k"], lo, ss[metric] + ss[ci], color=COLORS[meth], alpha=0.15, lw=0)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
                if clipped:
                    ax.set_ylim(bottom=floor * 0.8)
                for meth in clipped:
                    ax.annotate(f"{SHORT[meth]}: error below {floor:g}\n(mean-preserving, near exact)", (0.97, 0.06),
                                xycoords="axes fraction", ha="right", va="bottom", fontsize=7.5,
                                color=COLORS[meth])
            if ylim:
                ax.set_ylim(*ylim)
    for ax in axes[-1]:
        ax.set_xlabel("Background size k (rows or centroids)")
    fig.supylabel(ylabel, x=0.0, fontsize=9.5, color=INK2)
    fig.suptitle(title, x=0.01, ha="left", y=1.035, fontsize=11)
    fig.tight_layout()
    _method_legend(fig, y=0.985)
    return _save(fig, cfg, fname)


def fig_sweep(cfg: dict) -> list[Path]:
    d = Path(cfg["results_dir"]) / "sweep"
    if not (d / "summary.csv").exists():
        return []
    s = pd.read_csv(d / "summary.csv")
    paths = [
        _line_grid(cfg, s, "rel_l1_mean", "Relative L1 error vs exact full-background SHAP (lower is better)",
                   "fig04_sweep_error_vs_k.png",
                   "Background-compression error, exact SHAP (mean over 100 test instances and 10 seeds, 95% CI)",
                   ci="rel_l1_ci95", logy=True),
        _line_grid(cfg, s, "spearman_mean", "Per-instance Spearman rank correlation with exact SHAP",
                   "fig05_sweep_rank_vs_k.png",
                   "Feature-ranking agreement with the full-background explanation (higher is better)",
                   ci="spearman_ci95"),
    ]
    # background mean shift (data property, independent of the model)
    ps = pd.read_csv(d / "per_seed.csv")
    ms = ps.groupby(["dataset", "method", "k"], as_index=False)["mean_shift"].mean()
    fig, axes = plt.subplots(1, len(cfg["datasets"]), figsize=(3.3 * len(cfg["datasets"]), 2.8), squeeze=False)
    for ax, dname in zip(axes[0], cfg["datasets"], strict=True):
        for meth in METHOD_ORDER:
            ss = ms[(ms["dataset"] == dname) & (ms["method"] == meth)].sort_values("k")
            ax.plot(ss["k"], np.maximum(ss["mean_shift"], 1e-3), color=COLORS[meth], marker=MARKERS[meth])
        ax.annotate("size-weighted: < 0.001", (0.97, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=7.5, color=COLORS["kmeans_weighted"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(8e-4, 2)
        ax.set_title(_ds_label(dname))
        ax.set_xlabel("Background size k")
    axes[0, 0].set_ylabel("RMS shift of background mean\n(standard deviations)")
    fig.suptitle("Equal-weight centroids move the background mean; size weights preserve it",
                 x=0.01, ha="left", y=1.08, fontsize=11)
    fig.tight_layout()
    _method_legend(fig, y=0.93)
    paths.append(_save(fig, cfg, "fig06_background_mean_shift.png"))

    # stability across seeds (background only)
    st = pd.read_csv(d / "stability.csv")
    paths.append(_line_grid(cfg, st, "inst_spearman", "Mean pairwise Spearman between seeds",
                            "fig11_stability_background.png",
                            "Seed-to-seed stability of explanations (exact SHAP, background effect only, 10 seeds)"))
    return paths


# --------------------------------------------------------------------------- kernel

def fig_kernel(cfg: dict) -> list[Path]:
    d = Path(cfg["results_dir"]) / "kernel"
    if not (d / "summary.csv").exists():
        return []
    s = pd.read_csv(d / "summary.csv")
    paths = []

    floor = 1e-3

    # error decomposition
    fig, axes = _grid(cfg)
    for i, m in enumerate(cfg["models"]):
        for j, dname in enumerate(cfg["datasets"]):
            ax = axes[i, j]
            sub = s[(s["dataset"] == dname) & (s["model"] == m)]
            notes = []
            for meth in METHOD_ORDER:
                ss = sub[sub["method"] == meth].sort_values("k")
                y = ss["background_rel_l1"].to_numpy()
                if np.all(y < floor):
                    notes.append((f"{SHORT[meth]}: < {floor:g}", COLORS[meth]))
                ax.plot(ss["k"], np.maximum(y, floor), color=COLORS[meth], marker=MARKERS[meth])
            samp = sub.groupby("k", as_index=False)["sampling_rel_l1"].mean()
            ys = samp["sampling_rel_l1"].to_numpy()
            if np.all(ys < floor):
                notes.append((f"sampling: < {floor:g}", MUTED))
            ax.plot(samp["k"], np.maximum(ys, floor), color=MUTED, lw=2.2, marker="x", zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(floor * 0.7, None)
            for n, (txt, col) in enumerate(notes):
                ax.annotate(txt, (0.97, 0.11 + 0.09 * n), xycoords="axes fraction", ha="right",
                            fontsize=7.5, color=col)
    for ax in axes[-1]:
        ax.set_xlabel("Background size k")
    fig.supylabel("Relative L1 error", x=0.0, fontsize=9.5, color=INK2)
    fig.suptitle("Where the error comes from: background compression (coloured) vs Kernel SHAP sampling (grey)",
                 x=0.01, ha="left", y=1.075, fontsize=11)
    fig.tight_layout()
    extra = [plt.Line2D([], [], color=MUTED, lw=2.2, marker="x",
                        label="Kernel SHAP sampling error (mean over methods)")]
    _method_legend(fig, extra=extra, ncol=2, y=0.985)
    paths.append(_save(fig, cfg, "fig07_error_decomposition.png"))

    # runtime vs accuracy
    fig, axes = _grid(cfg, sharex=False)
    for i, m in enumerate(cfg["models"]):
        for j, dname in enumerate(cfg["datasets"]):
            ax = axes[i, j]
            sub = s[(s["dataset"] == dname) & (s["model"] == m)]
            notes = []
            for meth in METHOD_ORDER:
                ss = sub[sub["method"] == meth].sort_values("k")
                x = ss["explain_s"] + ss["fit_s"] / cfg["kernel"]["n_explain"]
                y = ss["total_rel_l1"].to_numpy()
                if np.all(y < floor):
                    notes.append((f"{SHORT[meth]}: error < {floor:g}", COLORS[meth]))
                ax.plot(x, np.maximum(y, floor), color=COLORS[meth], marker=MARKERS[meth])
                if meth == "random":
                    for xi, yi, k in zip(x, y, ss["k"], strict=True):
                        ax.annotate(f"k={k}", (xi, yi), textcoords="offset points", xytext=(3, 4),
                                    fontsize=7, color=MUTED)
            ax.set_xscale("log")
            ax.set_yscale("log")
            _runtime_xaxis(ax)
            for n, (txt, col) in enumerate(notes):
                ax.annotate(txt, (0.97, 0.11 + 0.09 * n), xycoords="axes fraction", ha="right",
                            fontsize=7.5, color=col)
    for ax in axes[-1]:
        ax.set_xlabel("CPU seconds per explained instance")
    fig.supylabel("Relative L1 error vs exact full-background SHAP", x=0.0, fontsize=9.5, color=INK2)
    fig.suptitle("Runtime vs accuracy of end-to-end Kernel SHAP (lower-left is better; K-means fit amortised)",
                 x=0.01, ha="left", y=1.035, fontsize=11)
    fig.tight_layout()
    _method_legend(fig, y=0.985)
    paths.append(_save(fig, cfg, "fig08_runtime_vs_error.png"))

    stab_path = Path(cfg["results_dir"]) / "stability" / "kernel_stability.csv"
    if stab_path.exists():
        st = pd.read_csv(stab_path)
        paths.append(_line_grid(cfg, st, "inst_spearman", "Mean pairwise Spearman between seeds",
                                "fig12_stability_kernel.png",
                                "Seed-to-seed stability of end-to-end Kernel SHAP (background + sampling noise, 3 seeds)"))
    return paths


# --------------------------------------------------------------------------- faithfulness

def fig_faithfulness(cfg: dict) -> list[Path]:
    d = Path(cfg["results_dir"]) / "faithfulness"
    if not (d / "summary.csv").exists():
        return []
    curves = pd.read_csv(d / "curves.csv")
    summ = pd.read_csv(d / "summary.csv")
    k, ks = cfg["faithfulness"]["size"], cfg["faithfulness"]["standard_size"]
    methods = ["exact", "kmeans", "kmeans_weighted", "random", f"random_k{ks}", "random_order"]
    colors = {**COLORS, f"random_k{ks}": COLORS["random"]}
    styles = {f"random_k{ks}": (0, (4, 2))}
    names = {**LABELS, "kmeans": f"C-SHAP, k={k}", "kmeans_weighted": f"K-means weighted, k={k}",
             "random": f"Random, k={k}", f"random_k{ks}": f"Random, k={ks} (notebook 'standard')"}
    paths = []
    ds0, m0 = cfg["datasets"][0], cfg["models"][0]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, col, title in zip(axes, ["deletion", "insertion"],
                              ["Deletion (lower curve = more faithful)", "Insertion (higher curve = more faithful)"],
                              strict=True):
        sub = curves[(curves["dataset"] == ds0) & (curves["model"] == m0)]
        for meth in methods:
            ss = sub[sub["method"] == meth]
            ax.plot(ss["fraction"], ss[col], color=colors[meth], ls=styles.get(meth, "-"),
                    lw=1.4 if meth != "exact" else 2.2, label=names[meth])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Fraction of features removed / inserted")
    axes[0].set_ylabel("Predicted-class probability")
    axes[1].legend(loc="lower right", fontsize=7.5)
    fig.suptitle(f"Faithfulness curves, {_ds_label(ds0)} / {MODEL_LABELS[m0].lower()} "
                 f"(mean over {cfg['kernel']['n_explain']} instances)", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    paths.append(_save(fig, cfg, "fig09_faithfulness_curves.png"))

    # ABPC summary: insertion AUC - deletion AUC, normalised by the exact reference
    fig, axes = plt.subplots(1, len(cfg["models"]), figsize=(3.4 * len(cfg["models"]), 3.0), sharey=True,
                             squeeze=False)
    yl = list(cfg["datasets"])
    shown = [m for m in methods[1:] if m != "random_order"]
    ratios = []
    for ax, m in zip(axes[0], cfg["models"], strict=True):
        for idx, meth in enumerate(shown):
            vals = []
            for dname in yl:
                row = summ[(summ["dataset"] == dname) & (summ["model"] == m)].set_index("method")
                vals.append(row.loc[meth, "abpc"] / row.loc["exact", "abpc"])
            ratios += vals
            yy = np.arange(len(yl)) + (idx - 1.5) * 0.15
            mk = MARKERS.get(meth, "v")
            face = colors[meth] if not meth.startswith("random_k") else "white"
            ax.scatter(vals, yy, facecolors=face, edgecolors=colors[meth], marker=mk, s=30, zorder=3,
                       label=names[meth])
        ax.axvline(1.0, color=INK2, lw=0.9)
        ax.set_title(MODEL_LABELS[m])
        ax.set_yticks(np.arange(len(yl)), [_ds_label(x) for x in yl])
        ax.set_xlabel("Insertion - deletion AUC\n(relative to exact SHAP = 1)")
        ax.grid(axis="y", visible=False)
    lo = np.floor(min(ratios) * 20) / 20
    for ax in axes[0]:
        ax.set_xlim(lo, 1.015)
    if "random_order" in summ["method"].unique():
        rr = (summ.set_index(["dataset", "model", "method"])["abpc"].unstack("method"))
        rr = rr["random_order"] / rr["exact"]
        fig.text(0.01, -0.04, f"Random feature order (no-information baseline) scores between "
                 f"{rr.min():.2f} and {rr.max():.2f} on this scale, far off the left edge.",
                 fontsize=8, color=MUTED)
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7.5)
    fig.tight_layout()
    paths.append(_save(fig, cfg, "fig10_faithfulness_summary.png"))
    return paths


# --------------------------------------------------------------------------- runtime

def fig_runtime(cfg: dict) -> list[Path]:
    p = Path(cfg["results_dir"]) / "runtime" / "runtime.csv"
    if not p.exists():
        return []
    r = pd.read_csv(p)
    kr = r[r["kind"] == "kernel"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    marks = ["o", "s", "^", "D"]
    for (dname, g), mk in zip(kr.groupby("dataset", sort=False), marks, strict=False):
        g = g.sort_values("k")
        ax.plot(g["k"], g["explain_s_per_instance"], color=INK2, marker=mk, lw=1.2)
        last = g.iloc[-1]
        ax.annotate(f"{_ds_label(dname)} ({int(last['n_features'])} features)",
                    (last["k"], last["explain_s_per_instance"]), textcoords="offset points",
                    xytext=(-4, 6), ha="right", fontsize=8, color=INK)
        full = g[g["full_background"]]
        ax.scatter(full["k"], full["explain_s_per_instance"], s=70, facecolor="none", edgecolor=INK, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Background rows k (circled: full reference background)")
    ax.set_ylabel("Kernel SHAP seconds per instance")
    ax.set_title("Kernel SHAP cost grows linearly with background size", loc="left")
    ax = axes[1]
    kmr = r[r["kind"] == "kmeans_rows"]
    for kk, mk in zip(sorted(kmr["k"].unique()), ["o", "s"], strict=False):
        g = kmr[kmr["k"] == kk].sort_values("n_rows")
        ax.plot(g["n_rows"], g["kmeans_fit_s"], color=INK2, marker=mk, lw=1.2)
        last = g.iloc[-1]
        ax.annotate(f"k={kk}", (last["n_rows"], last["kmeans_fit_s"]), textcoords="offset points",
                    xytext=(-4, 6), ha="right", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rows clustered")
    ax.set_ylabel("K-means fit seconds (n_init=10)")
    ax.set_title("One-off K-means cost (MAGIC data)", loc="left")
    fig.tight_layout()
    return [_save(fig, cfg, "fig13_runtime_scaling.png")]


def make_all_figures(cfg: dict) -> list[Path]:
    set_style()
    paths: list[Path] = []
    for fn in (fig_replication, fig_sweep, fig_kernel, fig_faithfulness, fig_runtime):
        paths += fn(cfg)
    return paths
