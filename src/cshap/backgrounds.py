"""Background (reference) datasets for interventional SHAP.

Kernel SHAP estimates the value of a feature coalition S as the average model
output when the features outside S are replaced by rows of a *background*
dataset. Its cost grows linearly with the number of background rows, which is
why this repository compresses the background.

Three constructions are provided:

* ``random_background``  - a uniform random subsample without replacement
  (what ``shap.sample`` does; the notebook's "standard SHAP" baseline).
* ``kmeans_background``  - K-means centroids with equal weights. This is the
  C-SHAP-style background as implemented in the original notebook:
  ``KMeans(n_clusters=k, random_state=42, n_init=10)`` fitted on the
  standardized training data, centroids passed to ``KernelExplainer``.
* ``kmeans_background(weighted=True)`` - the same centroids weighted by
  cluster size. This variant is an extension added here; it preserves the
  background mean exactly (see ``Background.mean``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import shap
from sklearn.cluster import KMeans

METHODS = ("random", "kmeans", "kmeans_weighted")


@dataclass(frozen=True)
class Background:
    """A (possibly weighted) background dataset.

    Attributes
    ----------
    data:
        Array of shape ``(k, n_features)``.
    weights:
        Non-negative weights of shape ``(k,)`` that sum to one.
    method:
        Name of the construction (``"random"``, ``"kmeans"``,
        ``"kmeans_weighted"`` or ``"full"``).
    fit_seconds:
        Wall-clock time spent building the background (K-means fit time for
        the clustered variants, ~0 for sampling).
    counts:
        For clustered backgrounds, the number of source rows per centroid.
    """

    data: np.ndarray
    weights: np.ndarray
    method: str
    fit_seconds: float = 0.0
    counts: np.ndarray | None = field(default=None, repr=False)

    @property
    def size(self) -> int:
        return int(self.data.shape[0])

    @property
    def is_uniform(self) -> bool:
        return bool(np.allclose(self.weights, 1.0 / self.size))

    def mean(self) -> np.ndarray:
        """Weighted mean of the background rows."""
        return self.weights @ self.data

    def expanded(self) -> np.ndarray:
        """Return an unweighted array representing the same distribution.

        For uniform weights this is ``data`` itself. For cluster-size weights
        each centroid is repeated ``counts[j]`` times, which represents the
        weighted distribution exactly (useful for explainers that do not
        accept weights, such as ``shap.TreeExplainer``).
        """
        if self.is_uniform:
            return self.data
        if self.counts is None:
            raise ValueError("Non-uniform background without integer counts cannot be expanded.")
        return np.repeat(self.data, self.counts.astype(int), axis=0)

    def to_shap_data(self, feature_names: list[str] | None = None):
        """Convert to the object ``shap.KernelExplainer`` expects."""
        if self.is_uniform:
            return self.data
        from shap.utils._legacy import DenseData

        names = feature_names or [f"x{i}" for i in range(self.data.shape[1])]
        return DenseData(self.data, names, None, self.weights.astype(float).copy())


def full_background(X: np.ndarray) -> Background:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    return Background(X, np.full(n, 1.0 / n), "full")


def random_background(X: np.ndarray, k: int, seed: int = 0) -> Background:
    """Uniform random subsample of ``k`` rows without replacement (``shap.sample``)."""
    X = np.asarray(X, dtype=float)
    t0 = time.perf_counter()
    data = X.copy() if k >= X.shape[0] else np.asarray(shap.sample(X, k, random_state=seed))
    elapsed = time.perf_counter() - t0
    return Background(data, np.full(len(data), 1.0 / len(data)), "random", elapsed)


def kmeans_background(
    X: np.ndarray,
    k: int,
    seed: int = 42,
    weighted: bool = False,
    n_init: int = 10,
) -> Background:
    """K-means centroids as background (the C-SHAP-style compression).

    Parameters mirror the notebook: ``KMeans(n_clusters=k, random_state=seed,
    n_init=10)``. With ``weighted=False`` (notebook behaviour) every centroid
    gets weight ``1/k`` regardless of cluster size. With ``weighted=True`` the
    weight of centroid ``j`` is ``n_j / n``.
    """
    X = np.asarray(X, dtype=float)
    if k >= X.shape[0]:
        bg = full_background(X)
        return Background(bg.data, bg.weights, "kmeans_weighted" if weighted else "kmeans", 0.0,
                          np.ones(len(X), dtype=int))
    t0 = time.perf_counter()
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(X)
    elapsed = time.perf_counter() - t0
    counts = np.bincount(km.labels_, minlength=k)
    if weighted:
        weights = counts / counts.sum()
        method = "kmeans_weighted"
    else:
        weights = np.full(k, 1.0 / k)
        method = "kmeans"
    return Background(km.cluster_centers_.copy(), weights, method, elapsed, counts)


def make_background(method: str, X: np.ndarray, k: int, seed: int = 0) -> Background:
    """Factory used by the experiments."""
    if method == "random":
        return random_background(X, k, seed)
    if method == "kmeans":
        return kmeans_background(X, k, seed, weighted=False)
    if method == "kmeans_weighted":
        return kmeans_background(X, k, seed, weighted=True)
    if method == "full":
        return full_background(X)
    raise ValueError(f"Unknown background method {method!r}; expected one of {METHODS + ('full',)}")
