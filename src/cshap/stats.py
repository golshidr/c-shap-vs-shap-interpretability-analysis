"""Paired significance tests used to compare background methods."""

from __future__ import annotations

import numpy as np
from scipy import stats


def paired_test(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Compare paired samples ``a`` and ``b`` (e.g. per-instance errors).

    Returns the Wilcoxon signed-rank p-value (two-sided), the mean and median
    of ``a - b``, a percentile bootstrap 95% CI of the mean difference, the
    fraction of pairs where ``a < b`` and the matched-pairs rank-biserial
    correlation (effect size in [-1, 1]; negative means ``a`` tends to be
    smaller).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = a - b
    n = len(d)
    if n == 0:
        raise ValueError("No valid pairs")
    nz = d[d != 0]
    if len(nz) == 0:
        p, rbc = 1.0, 0.0
    else:
        p = float(stats.wilcoxon(nz).pvalue)
        ranks = stats.rankdata(np.abs(nz))
        r_plus = ranks[nz > 0].sum()
        r_minus = ranks[nz < 0].sum()
        rbc = float((r_plus - r_minus) / ranks.sum())
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n": int(n),
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "frac_a_better": float(np.mean(a < b)),
        "rank_biserial": rbc,
        "p_value": p,
    }


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj
