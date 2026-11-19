import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

DEFAULT_XGB = {
    "n_estimators": 700,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "n_jobs": 1,
    "early_stopping_rounds": 40,
    "random_state": 42,
}
DEFAULT_LGB = {
    "n_estimators": 300,
    "learning_rate": 0.06,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
    "verbose": -1,
    "random_state": 42,
}


def time_split(n_rows: int, valid_fraction: float = 0.1) -> int:
    return int(n_rows * (1 - valid_fraction))


def fit_xgb(X: pd.DataFrame, y: np.ndarray, params: dict | None = None, log_target: bool = True) -> "BoostModel":
    """Rows must be in time order so the validation block is the most recent data."""
    params = {**DEFAULT_XGB, **(params or {})}
    target = np.log1p(y) if log_target else y
    cut = time_split(len(X))
    model = xgb.XGBRegressor(**params)
    model.fit(X.iloc[:cut], target[:cut], eval_set=[(X.iloc[cut:], target[cut:])], verbose=False)
    return BoostModel(model, log_target, list(X.columns))


def fit_quantile(X: pd.DataFrame, y: np.ndarray, alpha: float, params: dict | None = None) -> "BoostModel":
    params = {**DEFAULT_LGB, **(params or {}), "objective": "quantile", "alpha": alpha}
    model = lgb.LGBMRegressor(**params)
    model.fit(X, np.log1p(y))
    return BoostModel(model, True, list(X.columns))


class BoostModel:
    def __init__(self, model, log_target: bool, columns: list[str]):
        self.model = model
        self.log_target = log_target
        self.columns = columns

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw = self.model.predict(X[self.columns])
        return np.clip(np.expm1(raw) if self.log_target else raw, 0, None)
