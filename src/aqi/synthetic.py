import json
import logging
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from aqi import config, fetch

log = logging.getLogger(__name__)

DIWALI_DATES = ["2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"]


def make_weather(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    day = index.dayofyear.to_numpy()
    hour_ist = (index.hour.to_numpy() + 5.5) % 24
    season = np.cos(2 * np.pi * (day - 15) / 365)
    temp = 25 - 11 * season + 5 * np.sin(2 * np.pi * (hour_ist - 9) / 24) + rng.normal(0, 1.2, len(index))
    monsoon = np.exp(-(((day - 215) / 35.0) ** 2))
    humidity = np.clip(
        55 + 25 * monsoon + 10 * season - 12 * np.sin(2 * np.pi * (hour_ist - 9) / 24) + rng.normal(0, 6, len(index)),
        12,
        100,
    )
    wind = np.clip(rng.gamma(2.2, 3.0, len(index)) * (1.15 - 0.35 * season) + 2 * monsoon, 0.3, 45)
    precip = np.where(rng.random(len(index)) < 0.03 + 0.18 * monsoon, rng.exponential(1.6, len(index)), 0.0)
    pressure = 1000 + 7 * season + rng.normal(0, 2.0, len(index)).cumsum() * 0.02
    pressure = pressure - (pressure - pressure.mean()) * 0.5
    cloud = np.clip(30 + 45 * monsoon + rng.normal(0, 25, len(index)), 0, 100)
    direction = rng.normal(290, 70, len(index)) % 360
    dew = temp - (100 - humidity) / 5
    blh = np.clip(
        250
        + 900 * np.clip(np.sin(2 * np.pi * (hour_ist - 7) / 24), 0, None) * (0.6 + 0.4 * (1 - season))
        + rng.normal(0, 60, len(index)),
        60,
        2500,
    )
    return pd.DataFrame(
        {
            "temperature_2m": temp,
            "relative_humidity_2m": humidity,
            "dew_point_2m": dew,
            "surface_pressure": pressure,
            "precipitation": precip,
            "wind_speed_10m": wind,
            "wind_direction_10m": direction,
            "cloud_cover": cloud,
            "boundary_layer_height": blh,
        },
        index=index,
    )


def make_air(weather: pd.DataFrame, rng: np.random.Generator, offset: float) -> pd.DataFrame:
    index = weather.index
    day = index.dayofyear.to_numpy()
    hour_ist = (index.hour.to_numpy() + 5.5) % 24
    winter = np.cos(2 * np.pi * (day - 5) / 365)
    log_base = 4.25 + 0.75 * winter
    diurnal = 0.30 * np.cos(2 * np.pi * (hour_ist - 22) / 24)
    dispersion = (
        -0.30 * np.log1p(weather["wind_speed_10m"].to_numpy()) - 0.00022 * weather["boundary_layer_height"].to_numpy()
    )
    humidity = 0.006 * (weather["relative_humidity_2m"].to_numpy() - 55)
    rain = -0.35 * (pd.Series(weather["precipitation"].to_numpy()).rolling(6, min_periods=1).sum().to_numpy() > 0.5)
    stubble = 0.35 * ((index.month == 10) & (index.day >= 15) | (index.month == 11) & (index.day <= 15))
    diwali = np.zeros(len(index))
    for d in DIWALI_DATES:
        centre = pd.Timestamp(d, tz="UTC") + pd.Timedelta(hours=14)
        hours = (index - centre) / pd.Timedelta(hours=1)
        diwali += 0.7 * np.exp(-(((hours.to_numpy() - 6) / 14.0) ** 2))
    noise = np.zeros(len(index))
    shocks = rng.normal(0, 0.09, len(index))
    for i in range(1, len(index)):
        noise[i] = 0.97 * noise[i - 1] + shocks[i]
    log_pm = log_base + diurnal + dispersion + humidity + rain + stubble + diwali + noise + offset
    pm25 = np.clip(np.exp(log_pm), 3, 900)
    pm10 = pm25 * rng.uniform(1.4, 2.0, len(index))
    return pd.DataFrame(
        {
            "pm2_5": pm25,
            "pm10": pm10,
            "nitrogen_dioxide": np.clip(20 + 0.18 * pm25 + rng.normal(0, 5, len(index)), 1, None),
            "sulphur_dioxide": np.clip(8 + 0.05 * pm25 + rng.normal(0, 2, len(index)), 0.5, None),
            "carbon_monoxide": np.clip(300 + 4.5 * pm25 + rng.normal(0, 40, len(index)), 50, None),
            "ozone": np.clip(40 + 30 * np.sin(2 * np.pi * (hour_ist - 9) / 24) + rng.normal(0, 8, len(index)), 1, None),
        },
        index=index,
    )


def generate(years: int | None = None, seed: int = 7) -> None:
    years = years or config.HISTORY_YEARS
    rng = np.random.default_rng(seed)
    cutoff = date.today() - timedelta(days=config.ARCHIVE_LAG_DAYS)
    start = date(cutoff.year - years + 1, 1, 1)
    end_ts = pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(hours=23)
    index = pd.date_range(pd.Timestamp(start, tz="UTC"), end_ts + pd.Timedelta(hours=72), freq="h", name="time")
    shared_weather = make_weather(index, rng)
    offsets = [0.0, 0.08, -0.04, 0.05, -0.07]

    for row, offset in zip(config.load_locations().itertuples(), offsets):
        slug = config.location_slug(row.name)
        jitter = rng.normal(0, 0.6, (len(index), 3))
        weather = shared_weather.copy()
        weather["temperature_2m"] += jitter[:, 0] * 0.3
        weather["wind_speed_10m"] = np.clip(weather["wind_speed_10m"] + jitter[:, 1], 0.3, None)
        weather["relative_humidity_2m"] = np.clip(weather["relative_humidity_2m"] + jitter[:, 2] * 2, 10, 100)
        air = make_air(weather, rng, offset)

        history_air = air.loc[:end_ts]
        history_weather = weather.loc[:end_ts, config.WEATHER_VARIABLES]
        for kind, frame in (("air", history_air), ("weather", history_weather)):
            for year, part in frame.groupby(frame.index.year):
                path = fetch.chunk_path(kind, slug, date(year, 1, 1))
                path.parent.mkdir(parents=True, exist_ok=True)
                part.to_parquet(path)

        recent_start = end_ts - pd.Timedelta(days=8)
        recent_air = air.loc[recent_start:end_ts]
        recent_weather = weather.loc[recent_start:, config.WEATHER_VARIABLES]
        fetch.recent_path(slug, "air").parent.mkdir(parents=True, exist_ok=True)
        recent_air.to_parquet(fetch.recent_path(slug, "air"))
        recent_weather.to_parquet(fetch.recent_path(slug, "weather"))

    meta = {
        "start": start.isoformat(),
        "end": cutoff.isoformat(),
        "years": years,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "SYNTHETIC",
    }
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("synthetic data written for %d locations", len(offsets))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    generate()
