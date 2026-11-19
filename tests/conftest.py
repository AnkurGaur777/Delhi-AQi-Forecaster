import numpy as np
import pandas as pd
import pytest

from aqi import config


@pytest.fixture
def hourly_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    index = pd.date_range("2025-01-01", periods=24 * 60, freq="h", tz="UTC", name="time")
    n = len(index)
    frame = pd.DataFrame(
        {
            "pm2_5": 60 + 30 * np.sin(np.arange(n) / 24) + rng.normal(0, 5, n),
            "pm10": 120 + rng.normal(0, 10, n),
            "nitrogen_dioxide": 30 + rng.normal(0, 3, n),
            "sulphur_dioxide": 10 + rng.normal(0, 1, n),
            "carbon_monoxide": 500 + rng.normal(0, 30, n),
            "ozone": 40 + rng.normal(0, 5, n),
            "temperature_2m": 15 + rng.normal(0, 3, n),
            "relative_humidity_2m": 60 + rng.normal(0, 8, n),
            "dew_point_2m": 8 + rng.normal(0, 2, n),
            "surface_pressure": 1005 + rng.normal(0, 2, n),
            "precipitation": np.where(rng.random(n) < 0.05, 1.0, 0.0),
            "wind_speed_10m": np.abs(rng.normal(8, 3, n)),
            "wind_direction_10m": rng.uniform(0, 360, n),
            "cloud_cover": rng.uniform(0, 100, n),
        },
        index=index,
    )
    frame["location"] = config.location_slug(config.load_locations()["name"].iloc[0])
    return frame
