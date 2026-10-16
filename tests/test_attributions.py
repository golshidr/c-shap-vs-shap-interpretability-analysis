"""Sanity properties of the attributions (Shapley axioms and agreement with exact values)."""

import itertools
import math

import numpy as np
import shap

from cshap.backgrounds import Background, full_background, kmeans_background, random_background
from cshap.explain import CShapExplainer, kernel_shap
from cshap.models import output_function
from cshap.reference import exact_shap

FULL_ENUM = 2**6  # 6 features -> Kernel SHAP enumerates every coalition


def test_kernel_shap_efficiency(toy, rf):
    X, _ = toy
    f = lambda Z: rf.predict_proba(Z)[:, 1]  # noqa: E731
    bg = kmeans_background(X, 10, seed=0)
    res = kernel_shap(f, bg, X[:5], nsamples=200, seed=0)
    np.testing.assert_allclose(res.values.sum(axis=1) + res.base_value, f(X[:5]), atol=1e-8)
    np.testing.assert_allclose(res.base_value, bg.weights @ f(bg.data))


def test_dummy_feature_gets_zero(toy):
    X, _ = toy
    f = lambda Z: Z[:, 0] + 2 * Z[:, 1] ** 2  # features 2..5 are unused  # noqa: E731
    res = kernel_shap(f, random_background(X, 20, seed=0), X[:4], nsamples=FULL_ENUM, seed=0)
    np.testing.assert_allclose(res.values[:, 2:], 0.0, atol=1e-10)


def _brute_force_shapley(f, x, background):
    """Exact interventional Shapley values by enumerating all 2^M coalitions."""
    m = len(x)
    phi = np.zeros(m)
    for j in range(m):
        others = [i for i in range(m) if i != j]
        for r in range(m):
            for S in itertools.combinations(others, r):
                w = math.factorial(r) * math.factorial(m - r - 1) / math.factorial(m)
                with_j, without = background.copy(), background.copy()
                with_j[:, list(S) + [j]] = x[list(S) + [j]]
                without[:, list(S)] = x[list(S)]
                phi[j] += w * (f(with_j).mean() - f(without).mean())
    return phi


def test_exact_reference_matches_brute_force_enumeration(toy, rf, gbm, logreg):
    X, _ = toy
    bg = random_background(X, 25, seed=5)
    for family, model in (("rf", rf), ("gbm", gbm), ("logreg", logreg)):
        f = output_function(model, family)
        e, _ = exact_shap(model, family, X[:2], bg)
        for i in range(2):
            np.testing.assert_allclose(e[i], _brute_force_shapley(f, X[i], bg.data), atol=1e-6)


def test_kernel_matches_exact_tree_shap_with_full_enumeration(toy, rf, gbm):
    X, _ = toy
    bg = random_background(X, 30, seed=3)
    for family, model in (("rf", rf), ("gbm", gbm)):
        f = output_function(model, family)
        k = kernel_shap(f, bg, X[:5], nsamples=FULL_ENUM, seed=0).values
        e, base = exact_shap(model, family, X[:5], bg)
        np.testing.assert_allclose(k, e, atol=1e-6)
        np.testing.assert_allclose(e.sum(axis=1) + base, f(X[:5]), atol=1e-6)


def test_linear_closed_form_and_kernel_agree(toy, logreg):
    X, _ = toy
    bg = random_background(X, 40, seed=0)
    e, base = exact_shap(logreg, "logreg", X[:5], bg)
    np.testing.assert_allclose(e, (X[:5] - bg.data.mean(axis=0)) * logreg.coef_[0])
    k = kernel_shap(logreg.decision_function, bg, X[:5], nsamples=FULL_ENUM, seed=0).values
    np.testing.assert_allclose(k, e, atol=1e-8)


def test_weighted_background_equals_expanded_rows(toy, rf):
    X, _ = toy
    f = lambda Z: rf.predict_proba(Z)[:, 1]  # noqa: E731
    bg = kmeans_background(X, 6, seed=0, weighted=True)
    expanded = Background(bg.expanded(), np.full(len(X), 1 / len(X)), "expanded")
    a = kernel_shap(f, bg, X[:3], nsamples=FULL_ENUM, seed=0).values
    b = kernel_shap(f, expanded, X[:3], nsamples=FULL_ENUM, seed=0).values
    np.testing.assert_allclose(a, b, atol=1e-8)
    e, _ = exact_shap(rf, "rf", X[:3], bg)
    np.testing.assert_allclose(a, e, atol=1e-6)


def test_exact_reference_uses_the_whole_background(toy, rf):
    """Guards against shap silently subsampling backgrounds larger than 100 rows."""
    X, _ = toy
    big = full_background(X[:300])
    halves = [full_background(X[:150]), full_background(X[150:300])]
    whole, _ = exact_shap(rf, "rf", X[300:305], big)
    parts = [exact_shap(rf, "rf", X[300:305], h)[0] for h in halves]
    np.testing.assert_allclose(whole, np.mean(parts, axis=0), atol=1e-10)


def test_seeded_kernel_shap_is_reproducible_and_parallel_safe(toy, rf):
    X, _ = toy
    f = lambda Z: rf.predict_proba(Z)[:, 1]  # noqa: E731
    bg = random_background(X, 10, seed=0)
    a = kernel_shap(f, bg, X[:4], nsamples=30, seed=7).values
    b = kernel_shap(f, bg, X[:4], nsamples=30, seed=7, n_jobs=2).values
    np.testing.assert_allclose(a, b)


def test_default_l1_reg_of_shap_truncates_explanations(toy):
    X, _ = toy
    Xw = np.hstack([X, X + 0.1])  # 12 features
    f = lambda Z: Z @ np.arange(1, 13, dtype=float)  # noqa: E731
    explainer = shap.KernelExplainer(f, Xw[:20])
    np.random.seed(0)
    phi = explainer.shap_values(Xw[:1], silent=True)
    assert (np.abs(phi) > 0).sum() <= 10  # shap>=0.47 default keeps at most 10 features
    ours = kernel_shap(f, random_background(Xw, 20, seed=0), Xw[:1], nsamples=500, seed=0)
    assert (np.abs(ours.values) > 0).sum() == 12


def test_cshap_explainer_api(toy, rf):
    X, _ = toy
    f = lambda Z: rf.predict_proba(Z)[:, 1]  # noqa: E731
    exp = CShapExplainer(f, n_clusters=8, nsamples=100).fit(X)
    res = exp.explain(X[:3])
    assert res.values.shape == (3, X.shape[1])
    np.testing.assert_allclose(res.base_value, exp.expected_value)
    np.testing.assert_allclose(res.values.sum(axis=1) + res.base_value, f(X[:3]), atol=1e-8)
