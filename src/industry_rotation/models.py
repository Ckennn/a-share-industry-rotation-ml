"""Deterministic estimator factories for the public model comparison."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from .features import BASELINE_FEATURES

TARGET_COLUMN = "next_month_excess_return"


def build_estimator(
    model_name: str,
    params: Mapping[str, object],
    feature_columns: Sequence[str],
    random_state: int = 42,
    n_jobs: int = 1,
) -> Pipeline:
    """Build one audited estimator with training-only preprocessing."""

    parameters = dict(params)
    if model_name == "xgboost":
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
            **parameters,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    elif model_name == "lightgbm":
        parameters.setdefault("verbose", -1)
        model = LGBMRegressor(
            objective="regression",
            random_state=random_state,
            n_jobs=n_jobs,
            **parameters,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    elif model_name in {"random_forest", "randomforest"}:
        model = RandomForestRegressor(
            random_state=random_state,
            n_jobs=n_jobs,
            **parameters,
        )
        steps = [("imputer", SimpleImputer(strategy="median")), ("model", model)]
    elif model_name == "linear_lasso":
        model = Lasso(
            max_iter=20_000,
            tol=1e-4,
            selection="cyclic",
            random_state=random_state,
            **parameters,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    else:
        raise ValueError(f"unknown model: {model_name}")
    pipeline = Pipeline(steps=steps)
    pipeline.feature_columns = tuple(feature_columns)
    return pipeline


def fit_scores(
    model_name: str,
    params: Mapping[str, object],
    train: pd.DataFrame,
    predict: pd.DataFrame,
    feature_columns: Sequence[str] = BASELINE_FEATURES,
    random_state: int = 42,
    n_jobs: int = 1,
) -> np.ndarray:
    estimator = build_estimator(model_name, params, feature_columns, random_state, n_jobs)
    estimator.fit(train.loc[:, feature_columns], train[TARGET_COLUMN].astype(float))
    return np.asarray(estimator.predict(predict.loc[:, feature_columns]), dtype=float)
