"""Deletion / insertion faithfulness curves for tabular attributions.

For each instance the features are ranked by how strongly they support the
predicted class. *Deletion* removes features in that order and tracks the
predicted-class score; a faithful ranking makes the score drop fast (low
AUC). *Insertion* starts from a fully removed input and adds features in the
same order; a faithful ranking makes the score rise fast (high AUC).

"Removing" a feature means replacing it with values from random rows of a
reference pool and averaging the model output over those draws (marginal /
interventional imputation, consistent with the interventional SHAP game).
The same draws are used for every attribution method, so differences between
methods are paired.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _target_score(out: np.ndarray, positive: bool, output_type: str) -> np.ndarray:
    """Score of the predicted class given the raw explained output."""
    if output_type == "probability":
        return out if positive else 1.0 - out
    return out if positive else -out  # logit


def removal_curves(
    f: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    phi: np.ndarray,
    X_pool: np.ndarray,
    output_type: str = "probability",
    n_draws: int = 32,
    seed: int = 0,
    threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(deletion, insertion)`` curves, each of shape ``(n, M + 1)``.

    ``curve[i, t]`` is the mean predicted-class score after ``t`` features
    have been removed (deletion) or inserted (insertion).
    ``threshold`` decides the predicted class from the raw output (0.5 for
    probabilities, 0 for logits by default).
    """
    X = np.asarray(X, dtype=float)
    phi = np.asarray(phi, dtype=float)
    n, m = X.shape
    if threshold is None:
        threshold = 0.5 if output_type == "probability" else 0.0
    fx = f(X)
    rng = np.random.default_rng(seed)
    deletion = np.empty((n, m + 1))
    insertion = np.empty((n, m + 1))
    for i in range(n):
        positive = bool(fx[i] >= threshold)
        support = phi[i] if positive else -phi[i]
        order = np.argsort(-support, kind="stable")
        draws = X_pool[rng.choice(len(X_pool), n_draws, replace=len(X_pool) < n_draws)]
        # Build all (m + 1) masked inputs for both curves in one batch.
        # keep[t, j] is True if feature j is still taken from x after t deletions.
        ranks = np.empty(m, dtype=int)
        ranks[order] = np.arange(m)
        t = np.arange(m + 1)[:, None]
        keep_del = ranks[None, :] >= t  # first t features in `order` removed
        keep_ins = ranks[None, :] < t  # first t features in `order` inserted
        masks = np.concatenate([keep_del, keep_ins])  # (2(m+1), m)
        batch = np.where(masks[:, None, :], X[i][None, None, :], draws[None, :, :])
        out = f(batch.reshape(-1, m)).reshape(2 * (m + 1), n_draws).mean(axis=1)
        score = _target_score(out, positive, output_type)
        deletion[i] = score[: m + 1]
        insertion[i] = score[m + 1 :]
    return deletion, insertion


def curve_auc(curves: np.ndarray) -> np.ndarray:
    """Area under each curve with the x-axis normalized to [0, 1]."""
    m = curves.shape[1] - 1
    return np.trapezoid(curves, dx=1.0 / m, axis=1)


def random_order_phi(n: int, m: int, seed: int = 0) -> np.ndarray:
    """Random attributions, used as a no-information baseline ordering."""
    return np.random.default_rng(seed).standard_normal((n, m))
