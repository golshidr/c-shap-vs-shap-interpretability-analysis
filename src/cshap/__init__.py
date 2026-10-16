"""C-SHAP-style clustered backgrounds for Kernel SHAP, plus an evaluation suite.

Public API::

    from cshap import CShapExplainer, kernel_shap, kmeans_background, random_background
"""

from .backgrounds import (
    Background,
    full_background,
    kmeans_background,
    make_background,
    random_background,
)
from .explain import CShapExplainer, ShapResult, kernel_shap
from .reference import exact_shap

__all__ = [
    "Background",
    "CShapExplainer",
    "ShapResult",
    "exact_shap",
    "full_background",
    "kernel_shap",
    "kmeans_background",
    "make_background",
    "random_background",
]

__version__ = "0.2.0"
