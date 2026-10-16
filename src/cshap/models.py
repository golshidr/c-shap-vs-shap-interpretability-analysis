"""Model families used in the experiments and the scalar output each one explains."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

MODEL_FAMILIES = ("rf", "gbm", "logreg")

MODEL_LABELS = {
    "rf": "Random forest",
    "gbm": "Gradient boosting",
    "logreg": "Logistic regression",
}

# What scalar output is explained for each family.
#  - random forest: predicted probability of the positive class (as in the notebook);
#    exact interventional Tree SHAP is available because the forest's probability is
#    an average of tree outputs.
#  - gradient boosting and logistic regression: the log-odds (decision_function).
#    For boosting, the probability is a sigmoid of a sum of trees, and shap's
#    TreeExplainer(model_output="probability") does not return the exact Shapley
#    values of that game (checked against brute-force enumeration in
#    tests/test_attributions.py), whereas the log-odds output is exact.
OUTPUT_TYPE = {"rf": "probability", "gbm": "logit", "logreg": "logit"}


def make_model(family: str, seed: int = 42):
    """Instantiate an unfitted model. ``rf`` reproduces the notebook's settings."""
    if family == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    if family == "gbm":
        return GradientBoostingClassifier(random_state=seed)
    if family == "logreg":
        return LogisticRegression(max_iter=5000)
    raise ValueError(f"Unknown model family {family!r}; expected one of {MODEL_FAMILIES}")


def output_function(model, family: str) -> Callable[[np.ndarray], np.ndarray]:
    """Return ``f(X) -> (n,)``, the scalar output that is explained."""
    if OUTPUT_TYPE[family] == "logit":
        return model.decision_function

    def proba_pos(X: np.ndarray) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    return proba_pos


def fit_model(family: str, X_train: np.ndarray, y_train: np.ndarray, seed: int = 42):
    model = make_model(family, seed)
    model.fit(X_train, y_train)
    return model
