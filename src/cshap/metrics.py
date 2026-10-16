"""Agreement and stability metrics between attribution matrices.

All functions take attribution matrices of shape ``(n_instances, n_features)``.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def global_importance(phi: np.ndarray) -> np.ndarray:
    """Mean absolute attribution per feature (the notebook's global importance)."""
    return np.abs(np.asarray(phi)).mean(axis=0)


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int) -> float:
    """|top-k(a) ∩ top-k(b)| / k, where top-k is by absolute value."""
    k = min(k, len(a))
    ta = set(np.argsort(-np.abs(a), kind="stable")[:k])
    tb = set(np.argsort(-np.abs(b), kind="stable")[:k])
    return len(ta & tb) / k


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(stats.spearmanr(a, b).statistic)


def instance_metrics(phi: np.ndarray, phi_ref: np.ndarray, k: int = 5) -> dict[str, np.ndarray]:
    """Per-instance agreement with a reference attribution matrix.

    Returns arrays of length ``n_instances``:

    * ``mae``      - mean absolute error over features.
    * ``rel_l1``   - ``||phi - ref||_1 / ||ref||_1`` (scale-free).
    * ``spearman`` - Spearman correlation of absolute attributions (rank agreement).
    * ``topk``     - top-k overlap of absolute attributions.
    * ``sign_agree`` - fraction of the reference top-k features whose sign matches.
    """
    phi = np.asarray(phi, dtype=float)
    ref = np.asarray(phi_ref, dtype=float)
    diff = np.abs(phi - ref)
    mae = diff.mean(axis=1)
    denom = np.abs(ref).sum(axis=1)
    rel = np.divide(diff.sum(axis=1), denom, out=np.full(len(ref), np.nan), where=denom > 0)
    sp = np.array([_safe_spearman(np.abs(p), np.abs(r)) for p, r in zip(phi, ref, strict=True)])
    tk = np.array([topk_overlap(p, r, k) for p, r in zip(phi, ref, strict=True)])
    kk = min(k, phi.shape[1])
    sign = np.empty(len(ref))
    for i, (p, r) in enumerate(zip(phi, ref, strict=True)):
        top = np.argsort(-np.abs(r), kind="stable")[:kk]
        sign[i] = np.mean(np.sign(p[top]) == np.sign(r[top]))
    return {"mae": mae, "rel_l1": rel, "spearman": sp, "topk": tk, "sign_agree": sign}


def global_metrics(phi: np.ndarray, phi_ref: np.ndarray, k: int = 10) -> dict[str, float]:
    """Agreement of global importance vectors.

    ``pearson`` and ``mad`` are the two statistics reported in the original
    notebook; Spearman/Kendall and top-k overlap are added because the
    question of interest is usually whether the *ranking* is preserved.
    """
    g = global_importance(phi)
    gr = global_importance(phi_ref)
    return {
        "pearson": float(np.corrcoef(g, gr)[0, 1]),
        "spearman": _safe_spearman(g, gr),
        "kendall": float(stats.kendalltau(g, gr).statistic),
        f"top{k}": topk_overlap(g, gr, k),
        "mad": float(np.mean(np.abs(g - gr))),
        "rel_l1": float(np.abs(g - gr).sum() / np.abs(gr).sum()),
    }


def pairwise_stability(runs: list[np.ndarray], k: int = 5) -> dict[str, float]:
    """Agreement between repeated runs (different seeds) of the same method.

    For every pair of runs, compute per-instance Spearman of absolute
    attributions, per-instance top-k overlap and the global-importance
    Spearman; return their means. Also returns the mean per-feature standard
    deviation across runs relative to the mean absolute attribution.
    """
    if len(runs) < 2:
        raise ValueError("Need at least two runs")
    sps, tks, gsp = [], [], []
    for a, b in itertools.combinations(runs, 2):
        m = instance_metrics(a, b, k=k)
        sps.append(np.nanmean(m["spearman"]))
        tks.append(np.mean(m["topk"]))
        gsp.append(_safe_spearman(global_importance(a), global_importance(b)))
    stack = np.stack(runs)  # (runs, n, M)
    cv = stack.std(axis=0, ddof=1).mean() / max(np.abs(stack).mean(), 1e-12)
    return {
        "inst_spearman": float(np.nanmean(sps)),
        f"inst_top{k}": float(np.mean(tks)),
        "global_spearman": float(np.nanmean(gsp)),
        "rel_std": float(cv),
    }
