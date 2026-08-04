import numpy as np
import pandas as pd

from industry_rotation.features import BASELINE_FEATURES
from industry_rotation.models import build_estimator, fit_scores


def test_tree_factory_is_deterministic(training_frame: pd.DataFrame) -> None:
    params = {
        "n_estimators": 10,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
    }
    first = fit_scores("xgboost", params, training_frame, training_frame)
    second = fit_scores("xgboost", params, training_frame, training_frame)
    np.testing.assert_allclose(first, second)


def test_all_public_estimators_use_registered_features() -> None:
    for model_name, params in (
        ("xgboost", {"n_estimators": 2, "max_depth": 1}),
        ("lightgbm", {"n_estimators": 2, "max_depth": 1, "num_leaves": 3, "verbose": -1}),
        ("random_forest", {"n_estimators": 2, "max_depth": 1}),
        ("linear_lasso", {"alpha": 0.001}),
    ):
        estimator = build_estimator(model_name, params, BASELINE_FEATURES)
        assert estimator.feature_columns == BASELINE_FEATURES
        assert estimator.steps[0][0] == "imputer"
        assert ("scaler" in estimator.named_steps) == (model_name == "linear_lasso")
