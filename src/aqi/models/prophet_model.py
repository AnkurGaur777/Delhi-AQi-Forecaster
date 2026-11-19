import logging

import numpy as np
import pandas as pd

from aqi import config

log = logging.getLogger(__name__)
REGRESSORS = ["wind_speed_10m", "relative_humidity_2m", "temperature_2m", "precipitation"]


def prophet_available() -> bool:
    try:
        import prophet  # noqa: F401
    except ImportError:
        return False
    return True


def diwali_holidays() -> pd.DataFrame:
    festivals = config.load_festivals()
    diwali = festivals[festivals["name"] == "Diwali"]
    return pd.DataFrame({"holiday": "diwali", "ds": diwali["date"], "lower_window": -2, "upper_window": 3})


def _frame(history: pd.DataFrame) -> pd.DataFrame:
    out = history[REGRESSORS].copy()
    out["ds"] = history.index.tz_convert("Asia/Kolkata").tz_localize(None)
    return out.reset_index(drop=True)


def fit(history: pd.DataFrame):
    """Fit Prophet on log PM2.5 using only rows in `history`."""
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    model = Prophet(
        holidays=diwali_holidays(),
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_prior_scale=5.0,
    )
    for name in REGRESSORS:
        model.add_regressor(name)
    train = _frame(history)
    train["y"] = np.log1p(history["pm2_5"].to_numpy())
    model.fit(train.dropna(subset=["y"]))
    return model


def predict(model, future: pd.DataFrame) -> pd.Series:
    frame = _frame(future)
    forecast = model.predict(frame)
    values = np.clip(np.expm1(forecast["yhat"].to_numpy()), 0, None)
    return pd.Series(values, index=future.index)
