import numpy as np

from aqi import clean


def test_short_gaps_are_filled_and_flagged(hourly_frame):
    frame = hourly_frame.drop(columns="location").copy()
    frame.iloc[100:102, frame.columns.get_loc("pm2_5")] = np.nan
    out = clean.fill_short_gaps(frame)
    assert out["pm2_5"].iloc[100:102].notna().all()
    assert out["pm25_imputed"].iloc[100:102].all()
    assert not out["pm25_imputed"].iloc[:100].any()


def test_long_gaps_stay_missing(hourly_frame):
    frame = hourly_frame.drop(columns="location").copy()
    frame.iloc[200:210, frame.columns.get_loc("pm2_5")] = np.nan
    out = clean.fill_short_gaps(frame)
    assert out["pm2_5"].iloc[203:207].isna().all()


def test_impossible_values_are_removed(hourly_frame):
    frame = hourly_frame.drop(columns="location").copy()
    frame.iloc[5, frame.columns.get_loc("pm2_5")] = -4
    frame.iloc[6, frame.columns.get_loc("pm2_5")] = 5000
    out = clean.cap_impossible(frame)
    assert np.isnan(out["pm2_5"].iloc[5]) and np.isnan(out["pm2_5"].iloc[6])


def test_longest_gap():
    import pandas as pd

    series = pd.Series([1, np.nan, np.nan, 2, np.nan, 3])
    assert clean.longest_gap(series) == 2
