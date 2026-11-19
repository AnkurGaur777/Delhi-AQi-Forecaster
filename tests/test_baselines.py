import pandas as pd

from aqi.models import baselines


def test_persistence_returns_the_origin_value():
    index = pd.date_range("2025-01-01", periods=100, freq="h", tz="UTC")
    pm = pd.Series(range(100), index=index, dtype=float)
    origins = index[50:52]
    assert list(baselines.persistence(pm, origins, 24)) == [50.0, 51.0]


def test_seasonal_naive_uses_the_same_hour_of_the_target_day():
    index = pd.date_range("2025-01-01", periods=200, freq="h", tz="UTC")
    pm = pd.Series(range(200), index=index, dtype=float)
    origin = index[100:101]
    assert baselines.seasonal_naive(pm, origin, 24)[0] == 100.0
    assert baselines.seasonal_naive(pm, origin, 6)[0] == 82.0
    assert baselines.seasonal_naive(pm, origin, 48)[0] == 100.0
