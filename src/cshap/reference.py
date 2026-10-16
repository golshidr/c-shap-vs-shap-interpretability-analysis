"""Exact interventional SHAP values used as ground truth.

Kernel SHAP with background B estimates the Shapley values of the game
``v(S) = E_{b ~ B}[ f(x_S, b_{not S}) ]``. For two model families the exact
values of that same game are cheap to compute:

* tree ensembles: ``shap.TreeExplainer(..., feature_perturbation="interventional")``
  computes them exactly in polynomial time for the raw ensemble output (random
  forest probability, gradient-boosting log-odds). Note that shap's
  ``model_output="probability"`` for boosted trees is *not* exact for the
  probability game, so boosting is explained in log-odds space;
* linear models explained in log-odds space: ``phi_j = w_j (x_j - E_B[x_j])``.

This lets the experiments measure how far a compressed background moves the
explanation away from the full-background explanation, without Monte Carlo
noise, and separately how much noise Kernel SHAP's sampling adds.
"""

from __future__ import annotations

import warnings

import numpy as np
import shap

from .backgrounds import Background

EXACT_FAMILIES = ("rf", "gbm", "logreg")


def exact_shap(model, family: str, X: np.ndarray, background: Background) -> tuple[np.ndarray, float]:
    """Exact interventional SHAP values of the explained output for ``X``.

    Returns ``(values, base_value)`` with ``values`` of shape ``(n, M)``.
    Weighted backgrounds are handled exactly via ``Background.expanded``
    (tree models) or the weighted mean (linear models).
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if family == "logreg":
        w = np.asarray(model.coef_).reshape(-1)
        b = float(np.asarray(model.intercept_).reshape(-1)[0])
        mu = background.mean()
        return (X - mu) * w, float(w @ mu + b)
    if family not in ("rf", "gbm"):
        raise ValueError(f"No exact SHAP available for family {family!r}")
    data = background.expanded()
    # Passing a raw array as ``data`` makes shap wrap it in
    # ``maskers.Independent(data, max_samples=100)``, which silently subsamples
    # any background larger than 100 rows. Build the masker explicitly so the
    # whole background is used.
    masker = shap.maskers.Independent(data, max_samples=len(data))
    kwargs = {"feature_perturbation": "interventional", "model_output": "raw"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(model, data=masker, **kwargs)
        vals = np.asarray(explainer.shap_values(X, check_additivity=False))
    ev = np.asarray(explainer.expected_value).reshape(-1)
    if vals.ndim == 3:  # (n, M, n_classes) -> positive class
        vals = vals[..., 1]
        ev = ev[1:]
    return vals, float(ev[0])
