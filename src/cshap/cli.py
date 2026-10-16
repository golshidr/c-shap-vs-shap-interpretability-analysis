"""Command-line interface.

Examples
--------
Reproduce everything (results/, figures/, reports/)::

    cshap run all --config configs/default.yaml

Explain your own scikit-learn compatible model with a C-SHAP background::

    cshap explain --model model.joblib --background train.csv --data new_rows.csv \\
        --k 20 --output attributions.csv

Compare C-SHAP against random backgrounds on your own model::

    cshap compare --model model.joblib --background train.csv --data new_rows.csv --k 20
"""

from __future__ import annotations

import argparse
import os
import sys
import time

EXPERIMENTS = ("replicate", "sweep", "kernel", "faithfulness", "stability", "runtime")


def _cmd_run(args) -> None:
    from .config import load_config

    cfg = load_config(args.config, n_jobs=args.n_jobs)
    names = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    for name in names:
        if name == "runtime":
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(var, "1")
        module = __import__(f"cshap.experiments.{name}", fromlist=["run"])
        t0 = time.perf_counter()
        print(f"[cshap] running {name} ...", flush=True)
        module.run(cfg)
        print(f"[cshap] {name} done in {time.perf_counter() - t0:.1f}s", flush=True)
    if args.experiment == "all":
        _cmd_figures(args)
        _cmd_report(args)


def _cmd_figures(args) -> None:
    from .config import load_config
    from .plotting import make_all_figures

    cfg = load_config(args.config)
    for path in make_all_figures(cfg):
        print(f"[cshap] wrote {path}")


def _cmd_report(args) -> None:
    from .config import load_config
    from .report import build_report

    cfg = load_config(args.config)
    print(f"[cshap] wrote {build_report(cfg)}")


def _cmd_download(args) -> None:
    from .data import DATASETS, download_dataset

    for name, info in DATASETS.items():
        if info.openml_id is None:
            print(f"{name}: bundled with scikit-learn")
        else:
            print(f"{name}: {download_dataset(name, args.data_dir, force=args.force)}")


def _load_user_inputs(args):
    import joblib
    import numpy as np
    import pandas as pd

    model = joblib.load(args.model)
    bg = pd.read_csv(args.background)
    X = pd.read_csv(args.data)
    for frame in (bg, X):
        if args.drop_columns:
            frame.drop(columns=[c for c in args.drop_columns if c in frame], inplace=True)
    if list(bg.columns) != list(X.columns):
        sys.exit("background and data must have identical columns (use --drop-columns for targets/ids)")
    method = getattr(model, args.method)

    def f(Z):
        out = np.asarray(method(pd.DataFrame(Z, columns=X.columns) if args.pass_dataframe else Z))
        return out[:, args.class_index] if out.ndim == 2 else out

    return f, bg.to_numpy(float), X.to_numpy(float), list(X.columns)


def _nsamples(value: str):
    return value if value == "auto" else int(value)


def _cmd_explain(args) -> None:
    import pandas as pd

    from .explain import CShapExplainer

    f, bg, X, cols = _load_user_inputs(args)
    exp = CShapExplainer(f, n_clusters=args.k, weighted=args.weighted, random_state=args.seed,
                         nsamples=_nsamples(args.nsamples), n_jobs=args.n_jobs).fit(bg)
    res = exp.explain(X, seed=args.seed)
    out = pd.DataFrame(res.values, columns=cols)
    out.insert(0, "base_value", res.base_value)
    out.to_csv(args.output, index=False)
    print(f"explained {len(X)} rows with k={args.k} centroids "
          f"(k-means {exp.background_.fit_seconds:.2f}s, Kernel SHAP {res.seconds:.2f}s CPU) -> {args.output}")


def _cmd_compare(args) -> None:
    import pandas as pd

    from .backgrounds import kmeans_background, random_background
    from .explain import kernel_shap
    from .metrics import global_metrics, instance_metrics

    f, bg, X, _ = _load_user_inputs(args)
    ref_size = min(args.reference_size, len(bg))
    runs = {
        f"random_{ref_size} (reference)": random_background(bg, ref_size, seed=args.seed),
        f"cshap_k{args.k}": kmeans_background(bg, args.k, seed=args.seed),
        f"cshap_weighted_k{args.k}": kmeans_background(bg, args.k, seed=args.seed, weighted=True),
        f"random_{args.k}": random_background(bg, args.k, seed=args.seed),
    }
    results = {name: kernel_shap(f, b, X, nsamples=_nsamples(args.nsamples), seed=args.seed,
                                 n_jobs=args.n_jobs) for name, b in runs.items()}
    ref_name = next(iter(results))
    ref = results[ref_name].values
    rows = []
    for name, r in results.items():
        im, gm = instance_metrics(r.values, ref), global_metrics(r.values, ref)
        rows.append({"background": name, "rows": runs[name].size,
                     "cpu_s_per_instance": r.seconds / len(X), "fit_s": runs[name].fit_seconds,
                     "rel_l1_vs_ref": im["rel_l1"].mean(), "spearman_vs_ref": im["spearman"].mean(),
                     "global_spearman_vs_ref": gm["spearman"]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.output:
        table.to_csv(args.output, index=False)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cshap", description="C-SHAP vs SHAP experiments and tools")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run experiments")
    r.add_argument("experiment", choices=(*EXPERIMENTS, "all"))
    r.add_argument("--config", default="configs/default.yaml")
    r.add_argument("--n-jobs", type=int, default=None)
    r.set_defaults(func=_cmd_run)

    fg = sub.add_parser("figures", help="regenerate figures from results/")
    fg.add_argument("--config", default="configs/default.yaml")
    fg.set_defaults(func=_cmd_figures)

    rp = sub.add_parser("report", help="build the HTML report")
    rp.add_argument("--config", default="configs/default.yaml")
    rp.set_defaults(func=_cmd_report)

    dl = sub.add_parser("download", help="download datasets into data/raw")
    dl.add_argument("--data-dir", default="data/raw")
    dl.add_argument("--force", action="store_true")
    dl.set_defaults(func=_cmd_download)

    def add_user_model_args(sp):
        sp.add_argument("--model", required=True, help="joblib-pickled fitted model")
        sp.add_argument("--background", required=True, help="CSV of background rows (e.g. training data)")
        sp.add_argument("--data", required=True, help="CSV of rows to explain")
        sp.add_argument("--method", default="predict_proba",
                        help="model method to explain (predict_proba, decision_function, predict)")
        sp.add_argument("--class-index", type=int, default=1, help="output column for 2-D outputs")
        sp.add_argument("--drop-columns", nargs="*", default=None, help="columns to drop, e.g. target")
        sp.add_argument("--pass-dataframe", action="store_true",
                        help="call the model with a DataFrame (for pipelines with column names)")
        sp.add_argument("--k", type=int, default=20, help="number of K-means centroids")
        sp.add_argument("--nsamples", default="auto")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--n-jobs", type=int, default=1)

    ex = sub.add_parser("explain", help="explain your own model with a C-SHAP background")
    add_user_model_args(ex)
    ex.add_argument("--weighted", action="store_true", help="weight centroids by cluster size")
    ex.add_argument("--output", default="cshap_attributions.csv")
    ex.set_defaults(func=_cmd_explain)

    cp = sub.add_parser("compare", help="compare C-SHAP with random backgrounds on your own model")
    add_user_model_args(cp)
    cp.add_argument("--reference-size", type=int, default=100)
    cp.add_argument("--output", default=None)
    cp.set_defaults(func=_cmd_compare)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
