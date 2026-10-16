"""Shared setup for experiments: data, model and exact reference per (dataset, model)."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..backgrounds import full_background
from ..data import PreparedData, prepare_data
from ..models import OUTPUT_TYPE, fit_model, output_function
from ..reference import exact_shap


@dataclass
class Setup:
    data: PreparedData
    family: str
    model: object
    X_explain: np.ndarray
    ref_values: np.ndarray
    ref_base: float
    accuracy: float
    ref_seconds: float

    @property
    def f(self):
        return output_function(self.model, self.family)

    @property
    def output_type(self) -> str:
        return OUTPUT_TYPE[self.family]


def build_setup(dataset: str, family: str, cfg: dict, section: str, with_reference: bool = True) -> Setup:
    sec = cfg[section]
    data = prepare_data(
        dataset,
        test_size=cfg["split"]["test_size"],
        seed=cfg["split"]["seed"],
        max_reference=sec.get("max_reference"),
        data_dir=cfg["data_dir"],
    )
    model = fit_model(family, data.X_train, data.y_train, seed=cfg["split"]["seed"])
    f = output_function(model, family)
    thr = 0.5 if OUTPUT_TYPE[family] == "probability" else 0.0
    acc = float(np.mean((f(data.X_test) >= thr).astype(int) == data.y_test))
    X_explain = data.X_test[: sec.get("n_explain", len(data.X_test))]
    ref, base, secs = np.empty((0, 0)), float("nan"), 0.0
    if with_reference:
        t0 = time.perf_counter()
        ref, base = exact_shap(model, family, X_explain, full_background(data.X_ref))
        secs = time.perf_counter() - t0
    return Setup(data, family, model, X_explain, ref, base, acc, secs)


def out_dir(cfg: dict, name: str) -> Path:
    p = Path(cfg["results_dir"]) / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float) + "\n", encoding="utf-8")


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6g")
