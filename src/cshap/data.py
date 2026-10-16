"""Public tabular datasets used in the experiments.

Only the Breast Cancer Wisconsin (Diagnostic) data ships with scikit-learn.
The other datasets are downloaded from OpenML (mirrors of UCI datasets) and
cached as CSV under ``data/raw/`` (git-ignored). Run
``python scripts/download_data.py`` or ``cshap download`` once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    label: str
    openml_id: int | None
    positive_class: str | None
    uci_url: str
    doi: str
    license: str


DATASETS: dict[str, DatasetInfo] = {
    "breast_cancer": DatasetInfo(
        name="breast_cancer",
        label="Breast Cancer Wisconsin",
        openml_id=None,  # bundled with scikit-learn
        positive_class=None,  # sklearn target: 1 = benign (as in the notebook)
        uci_url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
        doi="10.24432/C5DW2B",
        license="CC BY 4.0",
    ),
    "magic": DatasetInfo(
        name="magic",
        label="MAGIC Gamma Telescope",
        openml_id=1120,
        positive_class="g",  # gamma (signal) vs h (hadron background)
        uci_url="https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope",
        doi="10.24432/C52C8B",
        license="CC BY 4.0",
    ),
    "spambase": DatasetInfo(
        name="spambase",
        label="Spambase",
        openml_id=44,
        positive_class="1",  # spam
        uci_url="https://archive.ics.uci.edu/dataset/94/spambase",
        doi="10.24432/C53G6X",
        license="CC BY 4.0",
    ),
}

DEFAULT_DATA_DIR = Path("data/raw")


def _csv_path(name: str, data_dir: Path | str) -> Path:
    return Path(data_dir) / f"{name}.csv"


def download_dataset(name: str, data_dir: Path | str = DEFAULT_DATA_DIR, force: bool = False) -> Path:
    """Download an OpenML-hosted dataset and cache it as CSV with a binary ``target`` column."""
    info = DATASETS[name]
    path = _csv_path(name, data_dir)
    if info.openml_id is None:
        raise ValueError(f"{name} is bundled with scikit-learn and needs no download")
    if path.exists() and not force:
        return path
    from sklearn.datasets import fetch_openml

    path.parent.mkdir(parents=True, exist_ok=True)
    bunch = fetch_openml(data_id=info.openml_id, as_frame=True, parser="auto",
                         data_home=str(Path(data_dir) / "openml_cache"))
    X = bunch.data.apply(pd.to_numeric, errors="raise").astype(float)
    X.columns = [str(c).strip().rstrip(":") for c in X.columns]
    y = (bunch.target.astype(str).str.strip() == info.positive_class).astype(int)
    if y.nunique() != 2:
        raise RuntimeError(f"Unexpected target values for {name}: {bunch.target.unique()[:5]}")
    df = X.copy()
    df["target"] = y.to_numpy()
    df.to_csv(path, index=False)
    return path


def load_dataset(
    name: str, data_dir: Path | str = DEFAULT_DATA_DIR, download: bool = True
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return ``(X, y)`` with ``X`` a float DataFrame and ``y`` a 0/1 array."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {list(DATASETS)}")
    if name == "breast_cancer":
        from sklearn.datasets import load_breast_cancer

        d = load_breast_cancer()
        return pd.DataFrame(d.data, columns=d.feature_names), d.target.astype(int)
    path = _csv_path(name, data_dir)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py`.")
        download_dataset(name, data_dir)
    df = pd.read_csv(path)
    y = df.pop("target").to_numpy().astype(int)
    return df.astype(float), y


@dataclass
class PreparedData:
    """Standardized train/test split plus the reference background pool."""

    name: str
    feature_names: list[str]
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    X_ref: np.ndarray  # pool that all backgrounds are drawn from (= reference background)

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def prepare_data(
    name: str,
    test_size: float = 0.2,
    seed: int = 42,
    max_reference: int | None = 2000,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> PreparedData:
    """Stratified split + standardization, following the notebook.

    ``X_ref`` is the training set, randomly capped at ``max_reference`` rows.
    It is the "full" background: exact reference SHAP values are computed
    against it, and every compressed background (random or K-means) is built
    from it, so the comparison isolates the effect of compression.
    """
    X, y = load_dataset(name, data_dir)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)
    if max_reference is not None and len(X_tr_s) > max_reference:
        idx = np.random.default_rng(seed).choice(len(X_tr_s), max_reference, replace=False)
        X_ref = X_tr_s[np.sort(idx)]
    else:
        X_ref = X_tr_s
    return PreparedData(name, list(X.columns), X_tr_s, X_te_s, np.asarray(y_tr), np.asarray(y_te), X_ref)
