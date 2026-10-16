import numpy as np
import pytest

from cshap.faithfulness import curve_auc, random_order_phi, removal_curves
from cshap.metrics import global_metrics, instance_metrics, pairwise_stability, topk_overlap
from cshap.stats import holm_adjust, paired_test


def test_topk_overlap():
    a = np.array([5, 4, 3, 2, 1.0])
    assert topk_overlap(a, a, 3) == 1.0
    assert topk_overlap(a, a[::-1], 2) == 0.0
    assert topk_overlap(a, -a, 3) == 1.0  # by absolute value


def test_instance_metrics_identity_and_scale():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((5, 8))
    m = instance_metrics(ref, ref)
    np.testing.assert_allclose(m["rel_l1"], 0)
    np.testing.assert_allclose(m["spearman"], 1)
    np.testing.assert_allclose(m["topk"], 1)
    m2 = instance_metrics(2 * ref, ref)
    np.testing.assert_allclose(m2["rel_l1"], 1)
    np.testing.assert_allclose(m2["spearman"], 1)


def test_global_metrics_identity():
    ref = np.random.default_rng(1).standard_normal((10, 6))
    g = global_metrics(ref, ref, k=3)
    assert g["pearson"] == pytest.approx(1) and g["mad"] == 0 and g["top3"] == 1


def test_pairwise_stability():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((6, 10))
    runs = [base + 1e-6 * rng.standard_normal(base.shape) for _ in range(3)]
    s = pairwise_stability(runs)
    assert s["inst_spearman"] > 0.999 and s["rel_std"] < 1e-4
    with pytest.raises(ValueError):
        pairwise_stability(runs[:1])


def test_paired_test_detects_shift():
    rng = np.random.default_rng(3)
    b = rng.random(60)
    a = b - 0.1 + 0.01 * rng.standard_normal(60)
    r = paired_test(a, b)
    assert r["p_value"] < 1e-6 and r["mean_diff"] < 0 and r["frac_a_better"] > 0.9
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"] and r["rank_biserial"] < -0.9
    same = paired_test(b, b)
    assert same["p_value"] == 1.0


def test_holm_adjust():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(holm_adjust(p), [0.03, 0.06, 0.06, 0.02])


def test_removal_curves_endpoints_and_ordering(toy, logreg):
    X, _ = toy
    f = logreg.decision_function
    x = X[:20]
    exact = (x - X.mean(axis=0)) * logreg.coef_[0]
    dele, ins = removal_curves(f, x, exact, X, output_type="logit", n_draws=16, seed=0)
    z = f(x)
    np.testing.assert_allclose(dele[:, 0], np.where(z >= 0, z, -z))
    np.testing.assert_allclose(ins[:, -1], np.where(z >= 0, z, -z))
    rd, ri = removal_curves(f, x, random_order_phi(*x.shape, seed=0), X, output_type="logit",
                            n_draws=16, seed=0)
    # exact attributions should be more faithful than a random ordering
    assert curve_auc(dele).mean() < curve_auc(rd).mean()
    assert curve_auc(ins).mean() > curve_auc(ri).mean()
