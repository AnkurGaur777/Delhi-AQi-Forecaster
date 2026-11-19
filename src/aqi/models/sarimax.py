import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

ORDER = (2, 0, 1)
FOURIER_PERIODS = [24, 12]
EXOG_COLUMNS = ["wind_speed_10m", "relative_humidity_2m"]
FIT_WINDOW_HOURS = 24 * 42
STATE_WINDOW_HOURS = 24 * 10


def fourier_terms(index: pd.DatetimeIndex) -> pd.DataFrame:
    hours = np.asarray(index.tz_convert("Asia/Kolkata").hour + index.minute / 60)
    out = {}
    for period in FOURIER_PERIODS:
        out[f"sin{period}"] = np.sin(2 * np.pi * hours / period)
        out[f"cos{period}"] = np.cos(2 * np.pi * hours / period)
    return pd.DataFrame(out, index=index)


def build_exog(frame: pd.DataFrame) -> pd.DataFrame:
    exog = frame[EXOG_COLUMNS].copy()
    exog["wind_speed_10m"] = np.log1p(exog["wind_speed_10m"])
    exog["relative_humidity_2m"] = exog["relative_humidity_2m"] / 100
    return exog.join(fourier_terms(frame.index))


def fit_params(history: pd.DataFrame) -> np.ndarray:
    """Fit on the most recent window of log PM2.5 and return the parameter vector."""
    window = history.iloc[-FIT_WINDOW_HOURS:]
    model = SARIMAX(np.log1p(window["pm2_5"]), exog=build_exog(window), order=ORDER, trend="c")
    return model.fit(disp=False, maxiter=60).params.to_numpy()


def forecast_from_origin(
    history: pd.DataFrame, future: pd.DataFrame, params: np.ndarray, steps: int = 48
) -> np.ndarray:
    """Filter recent data with fixed parameters, then forecast `steps` hours ahead."""
    recent = history.iloc[-STATE_WINDOW_HOURS:]
    model = SARIMAX(np.log1p(recent["pm2_5"]), exog=build_exog(recent), order=ORDER, trend="c")
    result = model.smooth(params)
    prediction = result.forecast(steps=steps, exog=build_exog(future.iloc[:steps]))
    return np.clip(np.expm1(prediction.to_numpy()), 0, None)


def residual_check(history: pd.DataFrame, params: np.ndarray, lags: int = 24) -> float:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    recent = history.iloc[-FIT_WINDOW_HOURS:]
    model = SARIMAX(np.log1p(recent["pm2_5"]), exog=build_exog(recent), order=ORDER, trend="c")
    resid = model.smooth(params).resid
    return float(acorr_ljungbox(resid, lags=[lags])["lb_pvalue"].iloc[0])
