import numpy as np
import pandas as pd


def persistence(pm: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    return pm.reindex(origins).to_numpy()


def seasonal_naive(pm: pd.Series, origins: pd.DatetimeIndex, horizon: int) -> np.ndarray:
    days_back = int(np.ceil(horizon / 24))
    lag = 24 * days_back - horizon
    return pm.reindex(origins - pd.Timedelta(hours=lag)).to_numpy()
