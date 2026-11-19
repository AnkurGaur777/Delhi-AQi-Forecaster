from dataclasses import dataclass

import numpy as np
import pandas as pd

from aqi import aqi_index, config


@dataclass(frozen=True)
class Fold:
    name: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(year=ts.year, month=ts.month, day=1, tz="UTC")


def walk_forward_folds(
    data_end: pd.Timestamp, n_test_months: int = 12, holdout_months: int = 3
) -> tuple[list[Fold], Fold]:
    """Expanding-window monthly folds followed by an untouched holdout block."""
    holdout_start = month_start(data_end) - pd.DateOffset(months=holdout_months - 1)
    folds = []
    for i in range(n_test_months, 0, -1):
        start = holdout_start - pd.DateOffset(months=i)
        end = holdout_start - pd.DateOffset(months=i - 1)
        folds.append(Fold(start.strftime("%Y-%m"), start, start, end))
    holdout = Fold("holdout", holdout_start, holdout_start, data_end + pd.Timedelta(hours=1))
    return folds, holdout


def train_mask(index: pd.DatetimeIndex, train_end: pd.Timestamp, horizon: int) -> np.ndarray:
    """Origins whose target time is still before the training cut-off."""
    return np.asarray(index + pd.Timedelta(hours=horizon) <= train_end)


def test_mask(index: pd.DatetimeIndex, fold: Fold, origin_step_hours: int = 6) -> np.ndarray:
    in_block = (index >= fold.test_start) & (index < fold.test_end)
    on_grid = index.hour % origin_step_hours == 0
    return np.asarray(in_block & on_grid)


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def skill_score(model_error: float, baseline_error: float) -> float:
    return 1.0 - model_error / baseline_error


def season_of(index: pd.DatetimeIndex) -> np.ndarray:
    ist = index.tz_convert("Asia/Kolkata")
    return np.where(np.isin(ist.month, [10, 11, 12, 1, 2]), "winter", "other")


def point_metrics(preds: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in preds.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        rows.append(
            {
                **dict(zip(by, key)),
                "n": len(group),
                "mae": mae(group["y_true"], group["y_pred"]),
                "rmse": rmse(group["y_true"], group["y_pred"]),
            }
        )
    return pd.DataFrame(rows)


def interpolate_path(observed_last24: np.ndarray, anchors: list[int], values: np.ndarray) -> np.ndarray:
    """Hourly path of length 24 + 48: last 24 observed hours then interpolated forecasts."""
    hours = np.arange(1, config.FORECAST_STEPS + 1)
    forecast = np.interp(hours, [0] + list(anchors), np.concatenate([[observed_last24[-1]], values]))
    return np.concatenate([observed_last24, forecast])


def predicted_24h_means(observed_last24: np.ndarray, anchors: list[int], values: np.ndarray) -> np.ndarray:
    path = interpolate_path(observed_last24, anchors, values)
    return np.array([path[h : h + 24].mean() for h in anchors])


def category_metrics(codes_true: np.ndarray, codes_pred: np.ndarray) -> dict:
    valid = (codes_true >= 0) & (codes_pred >= 0)
    t, p = codes_true[valid], codes_pred[valid]
    severe = t >= 3
    return {
        "category_accuracy": float(np.mean(t == p)),
        "within_one": float(np.mean(np.abs(t - p) <= 1)),
        "poor_or_worse_recall": float(np.mean(p[severe] >= 3)) if severe.any() else float("nan"),
        "n": int(valid.sum()),
    }


def interval_metrics(y_true, lo, hi) -> dict:
    y_true, lo, hi = map(np.asarray, (y_true, lo, hi))
    return {"coverage": float(np.mean((y_true >= lo) & (y_true <= hi))), "mean_width": float(np.mean(hi - lo))}


def true_24h_mean_at(hourly_pm: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    rolling = aqi_index.rolling_24h_mean(hourly_pm)
    return rolling.reindex(origins + pd.Timedelta(hours=horizon)).to_numpy()
