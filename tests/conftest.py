import warnings

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def toy():
    """Small synthetic binary problem with 6 features (Kernel SHAP can enumerate all coalitions)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 6))
    logits = 1.5 * X[:, 0] - 2.0 * X[:, 1] + X[:, 2] * X[:, 3]
    y = (logits + 0.3 * rng.standard_normal(400) > 0).astype(int)
    return X, y


@pytest.fixture(scope="session")
def rf(toy):
    X, y = toy
    return RandomForestClassifier(n_estimators=30, max_depth=5, random_state=0).fit(X, y)


@pytest.fixture(scope="session")
def gbm(toy):
    X, y = toy
    return GradientBoostingClassifier(n_estimators=30, random_state=0).fit(X, y)


@pytest.fixture(scope="session")
def logreg(toy):
    X, y = toy
    return LogisticRegression(max_iter=1000).fit(X, y)
