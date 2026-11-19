import numpy as np
import pandas as pd

from aqi import features


def test_past_features_ignore_the_future(hourly_frame):
    frame = hourly_frame.drop(columns="location")
    origin = frame.index[600]
    full = features.past_features(frame).loc[origin]

    truncated = frame.loc[:origin]
    cut = features.past_features(truncated).loc[origin]
    pd.testing.assert_series_equal(full, cut, check_names=False)


def test_past_features_change_when_only_future_changes_nothing(hourly_frame):
    frame = hourly_frame.drop(columns="location")
    origin = frame.index[700]
    altered = frame.copy()
    altered.loc[altered.index > origin, ["pm2_5", "pm10", "wind_speed_10m"]] = 1e6
    a = features.past_features(frame).loc[origin]
    b = features.past_features(altered).loc[origin]
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_target_is_not_a_feature(hourly_frame):
    origin_table = features.build_origin_table(hourly_frame)
    table = features.build_horizon_table(hourly_frame, origin_table, 24)
    assert "target" not in features.feature_columns(table)
    assert "horizon" not in features.feature_columns(table)


def test_target_is_the_value_h_hours_ahead(hourly_frame):
    origin_table = features.build_origin_table(hourly_frame)
    table = features.build_horizon_table(hourly_frame, origin_table, 12)
    origin = hourly_frame.index[300]
    expected = hourly_frame["pm2_5"].loc[origin + pd.Timedelta(hours=12)]
    assert table.loc[origin, "target"] == expected


def test_future_weather_is_shifted_by_horizon(hourly_frame):
    frame = hourly_frame.drop(columns="location")
    out = features.future_weather_features(frame, 6)
    origin = frame.index[100]
    assert out.loc[origin, "fw_temperature_2m"] == frame["temperature_2m"].loc[origin + pd.Timedelta(hours=6)]


def test_last_rows_have_no_target(hourly_frame):
    origin_table = features.build_origin_table(hourly_frame)
    table = features.build_horizon_table(hourly_frame, origin_table, 24)
    assert table["target"].tail(24).isna().all()


def test_wind_components_have_the_right_magnitude(hourly_frame):
    frame = features.wind_components(hourly_frame.drop(columns="location"))
    magnitude = np.hypot(frame["wind_u"], frame["wind_v"])
    np.testing.assert_allclose(magnitude, frame["wind_speed_10m"], rtol=1e-6)
