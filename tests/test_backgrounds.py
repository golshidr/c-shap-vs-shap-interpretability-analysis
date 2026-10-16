import numpy as np
import pytest
import shap
from sklearn.cluster import KMeans

from cshap.backgrounds import full_background, kmeans_background, make_background, random_background


def test_random_background_matches_shap_sample(toy):
    X, _ = toy
    bg = random_background(X, 50, seed=42)
    np.testing.assert_array_equal(bg.data, shap.sample(X, 50, random_state=42))
    assert bg.size == 50 and bg.is_uniform


def test_kmeans_background_reproduces_notebook_construction(toy):
    X, _ = toy
    bg = kmeans_background(X, 20, seed=42)
    km = KMeans(n_clusters=20, random_state=42, n_init=10).fit(X)
    np.testing.assert_allclose(bg.data, km.cluster_centers_)
    np.testing.assert_allclose(bg.weights, np.full(20, 1 / 20))
    assert bg.counts.sum() == len(X)


def test_weighted_centroids_preserve_the_mean(toy):
    X, _ = toy
    bg = kmeans_background(X, 15, seed=0, weighted=True)
    # size-weighted centroids average back to the data mean (up to the last k-means update)
    np.testing.assert_allclose(bg.mean(), X.mean(axis=0), atol=1e-2)
    np.testing.assert_allclose(bg.weights.sum(), 1.0)


def test_unweighted_centroids_generally_shift_the_mean(toy):
    X, _ = toy
    Xs = np.vstack([X, X[:10] + 8.0])  # a small far-away cluster
    unweighted = kmeans_background(Xs, 10, seed=0)
    weighted = kmeans_background(Xs, 10, seed=0, weighted=True)
    shift_u = np.linalg.norm(unweighted.mean() - Xs.mean(axis=0))
    shift_w = np.linalg.norm(weighted.mean() - Xs.mean(axis=0))
    assert shift_u > 10 * shift_w


def test_expanded_represents_weighted_distribution(toy):
    X, _ = toy
    bg = kmeans_background(X, 8, seed=1, weighted=True)
    ex = bg.expanded()
    assert ex.shape == (len(X), X.shape[1])
    np.testing.assert_allclose(ex.mean(axis=0), bg.mean())


def test_k_larger_than_data_returns_full(toy):
    X, _ = toy
    assert random_background(X[:5], 10).size == 5
    assert kmeans_background(X[:5], 10).size == 5
    assert full_background(X).size == len(X)


def test_make_background_rejects_unknown(toy):
    X, _ = toy
    with pytest.raises(ValueError):
        make_background("nope", X, 5)
