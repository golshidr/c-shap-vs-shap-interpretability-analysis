"""Kernel SHAP with a compressed background, and the reusable C-SHAP explainer."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shap
from joblib import Parallel, delayed

from .backgrounds import Background, kmeans_background


@dataclass
class ShapResult:
    """Attributions for a batch of instances.

    ``values`` has shape ``(n_instances, n_features)`` and, for Kernel SHAP,
    each row satisfies ``values.sum() + base_value == f(x)`` (efficiency).
    """

    values: np.ndarray
    base_value: float
    seconds: float
    per_instance_seconds: np.ndarray
    background: Background | None = None


def _explain_rows(f, bg_data, rows, idx, nsamples, l1_reg, seed):
    warnings.filterwarnings("ignore")
    explainer = shap.KernelExplainer(f, bg_data)
    out = np.empty_like(rows, dtype=float)
    secs = np.empty(len(rows))
    for j, (i, x) in enumerate(zip(idx, rows, strict=True)):
        # One RNG state per instance, so results do not depend on n_jobs/chunking.
        np.random.seed((seed * 1_000_003 + int(i)) % (2**32 - 1))
        t0 = time.perf_counter()
        phi = explainer.shap_values(x.reshape(1, -1), nsamples=nsamples, l1_reg=l1_reg, silent=True)
        secs[j] = time.perf_counter() - t0
        out[j] = np.asarray(phi).reshape(-1)
    return out, secs, float(np.asarray(explainer.expected_value).reshape(-1)[0])


def kernel_shap(
    f: Callable[[np.ndarray], np.ndarray],
    background: Background,
    X: np.ndarray,
    nsamples: int | str = "auto",
    l1_reg: str | bool = False,
    seed: int = 0,
    n_jobs: int = 1,
) -> ShapResult:
    """Run ``shap.KernelExplainer`` for a scalar-output function.

    Parameters
    ----------
    f:
        Function mapping ``(n, M)`` inputs to ``(n,)`` outputs.
    background:
        Background distribution (uniform or weighted).
    nsamples:
        Coalition samples per instance (``"auto"`` = ``2*M + 2048``, the shap default).
    l1_reg:
        Passed to ``shap_values``. Defaults to ``False`` (no feature selection),
        unlike shap>=0.47 whose default ``"num_features(10)"`` keeps only ten
        non-zero attributions per instance.
    seed:
        Seeds the coalition sampling per instance for reproducibility.
    n_jobs:
        Parallel workers over instances (joblib/loky). ``seconds`` is the
        summed per-instance compute time, i.e. the single-core cost.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    bg_data = background.to_shap_data()
    idx = np.arange(len(X))
    if n_jobs == 1 or len(X) == 1:
        chunks = [(X, idx)]
    else:
        n_chunks = min(len(X), n_jobs)
        chunks = [(X[c], c) for c in np.array_split(idx, n_chunks)]
    if len(chunks) == 1:
        results = [_explain_rows(f, bg_data, X, idx, nsamples, l1_reg, seed)]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_explain_rows)(f, bg_data, rows, ids, nsamples, l1_reg, seed) for rows, ids in chunks
        )
    values = np.vstack([r[0] for r in results])
    secs = np.concatenate([r[1] for r in results])
    return ShapResult(values, results[0][2], float(secs.sum()), secs, background)


class CShapExplainer:
    """Kernel SHAP with a K-means-compressed background (C-SHAP style).

    Example
    -------
    >>> explainer = CShapExplainer(lambda X: model.predict_proba(X)[:, 1], n_clusters=20)
    >>> explainer.fit(X_train)
    >>> result = explainer.explain(X_test[:10])
    >>> result.values.shape
    (10, n_features)

    Parameters
    ----------
    predict_fn:
        Scalar-output model function ``(n, M) -> (n,)``.
    n_clusters:
        Number of centroids ``k`` (the notebook uses 20).
    weighted:
        Weight centroids by cluster size (extension; the notebook uses equal weights).
    random_state, n_init:
        Passed to ``sklearn.cluster.KMeans``.
    nsamples, l1_reg, n_jobs:
        Passed to :func:`kernel_shap`.
    """

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], np.ndarray],
        n_clusters: int = 20,
        weighted: bool = False,
        random_state: int = 42,
        n_init: int = 10,
        nsamples: int | str = "auto",
        l1_reg: str | bool = False,
        n_jobs: int = 1,
    ) -> None:
        self.predict_fn = predict_fn
        self.n_clusters = n_clusters
        self.weighted = weighted
        self.random_state = random_state
        self.n_init = n_init
        self.nsamples = nsamples
        self.l1_reg = l1_reg
        self.n_jobs = n_jobs
        self.background_: Background | None = None

    def fit(self, X_background: np.ndarray) -> CShapExplainer:
        self.background_ = kmeans_background(
            np.asarray(X_background, dtype=float),
            self.n_clusters,
            seed=self.random_state,
            weighted=self.weighted,
            n_init=self.n_init,
        )
        return self

    @property
    def expected_value(self) -> float:
        self._check_fitted()
        return float(self.background_.weights @ self.predict_fn(self.background_.data))

    def explain(self, X: np.ndarray, seed: int = 0) -> ShapResult:
        self._check_fitted()
        return kernel_shap(self.predict_fn, self.background_, X, nsamples=self.nsamples,
                           l1_reg=self.l1_reg, seed=seed, n_jobs=self.n_jobs)

    def _check_fitted(self) -> None:
        if self.background_ is None:
            raise RuntimeError("Call fit(X_background) before explaining.")
