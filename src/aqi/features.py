import numpy as np
import pandas as pd

from aqi import config

PM_LAGS = [1, 2, 3, 6, 12, 24, 48, 72, 168]
ROLL_WINDOWS = [3, 6, 12, 24, 72]
CHANGE_WINDOWS = [6, 24]
WEATHER_NOW = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
]
FUTURE_WEATHER = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "cloud_cover",
    "wind_u",
    "wind_v",
]


def wind_components(frame: pd.DataFrame) -> pd.DataFrame:
    radians = np.deg2rad(frame["wind_direction_10m"])
    frame = frame.copy()
    frame["wind_u"] = -frame["wind_speed_10m"] * np.sin(radians)
    frame["wind_v"] = -frame["wind_speed_10m"] * np.cos(radians)
    return frame


def festival_flags(index: pd.DatetimeIndex) -> pd.DataFrame:
    festivals = config.load_festivals()
    days = index.tz_convert("Asia/Kolkata").normalize().tz_localize(None)
    diwali_dates = festivals.loc[festivals["name"] == "Diwali", "date"]
    holiday_dates = set(festivals["date"])
    distance = np.full(len(index), 99.0)
    for date in diwali_dates:
        distance = np.where(np.abs((days - date).days) < np.abs(distance), (days - date).days, distance)
    flags = pd.DataFrame(index=index)
    flags["diwali_window"] = (np.abs(distance) <= config.DIWALI_WINDOW_DAYS).astype(int)
    flags["days_from_diwali"] = np.clip(distance, -10, 10)
    flags["is_holiday"] = days.isin(holiday_dates).astype(int)
    month, day = days.month, days.day
    start_m, start_d = config.STUBBLE_START
    end_m, end_d = config.STUBBLE_END
    in_season = ((month == start_m) & (day >= start_d)) | ((month == end_m) & (day <= end_d))
    flags["stubble_season"] = in_season.astype(int)
    return flags


def calendar_features(index: pd.DatetimeIndex, prefix: str) -> pd.DataFrame:
    ist = index.tz_convert("Asia/Kolkata")
    out = pd.DataFrame(index=index)
    out[f"{prefix}hour_sin"] = np.sin(2 * np.pi * ist.hour / 24)
    out[f"{prefix}hour_cos"] = np.cos(2 * np.pi * ist.hour / 24)
    out[f"{prefix}dow"] = ist.day_of_week
    out[f"{prefix}month_sin"] = np.sin(2 * np.pi * (ist.month - 1) / 12)
    out[f"{prefix}month_cos"] = np.cos(2 * np.pi * (ist.month - 1) / 12)
    flags = festival_flags(index).add_prefix(prefix)
    return out.join(flags)


def past_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Features at origin time t built only from rows at or before t."""
    frame = wind_components(frame)
    pm = frame["pm2_5"]
    out = pd.DataFrame(index=frame.index)
    out["pm25_now"] = pm
    for lag in PM_LAGS:
        out[f"pm25_lag{lag}"] = pm.shift(lag)
    for window in ROLL_WINDOWS:
        roll = pm.rolling(window, min_periods=max(2, window // 2))
        out[f"pm25_mean{window}"] = roll.mean()
        out[f"pm25_max{window}"] = roll.max()
        out[f"pm25_std{window}"] = roll.std()
    for window in CHANGE_WINDOWS:
        out[f"pm25_change{window}"] = pm - pm.shift(window)
    for column in ["pm10", "nitrogen_dioxide", "carbon_monoxide"]:
        out[f"{column}_now"] = frame[column]
        out[f"{column}_change6"] = frame[column] - frame[column].shift(6)
    out["pm_ratio"] = pm / frame["pm10"].replace(0, np.nan)
    for column in WEATHER_NOW:
        out[f"{column}_now"] = frame[column]
    out["pressure_tendency6"] = frame["surface_pressure"] - frame["surface_pressure"].shift(6)
    out["wind_change6"] = frame["wind_speed_10m"] - frame["wind_speed_10m"].shift(6)
    out["rain_sum24"] = frame["precipitation"].rolling(24, min_periods=1).sum()
    out["wind_u_now"] = frame["wind_u"]
    out["wind_v_now"] = frame["wind_v"]
    out = out.join(calendar_features(frame.index, "now_"))
    return out


PAST_FEATURE_NAMES: list[str] = []


def future_weather_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Weather at the target hour t+h plus mean wind and total rain over (t, t+h]."""
    frame = wind_components(frame)
    out = pd.DataFrame(index=frame.index)
    for column in FUTURE_WEATHER:
        out[f"fw_{column}"] = frame[column].shift(-horizon)
    out["fw_wind_mean"] = frame["wind_speed_10m"].rolling(horizon, min_periods=1).mean().shift(-horizon)
    out["fw_rain_sum"] = frame["precipitation"].rolling(horizon, min_periods=1).sum().shift(-horizon)
    return out


def target_calendar(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    target_index = frame.index + pd.Timedelta(hours=horizon)
    calendar = calendar_features(target_index, "tgt_")
    calendar.index = frame.index
    return calendar


def location_codes() -> dict[str, int]:
    return {config.location_slug(n): i for i, n in enumerate(config.load_locations()["name"])}


def build_origin_table(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per-origin past features for all locations, with a location code."""
    codes = location_codes()
    parts = []
    for slug, part in hourly.groupby("location"):
        base = past_features(part.drop(columns=["location"]))
        base["location_code"] = codes[slug]
        base["location"] = slug
        parts.append(base)
    return pd.concat(parts)


def build_horizon_table(
    hourly: pd.DataFrame, origin_table: pd.DataFrame, horizon: int, use_future_weather: bool = True
) -> pd.DataFrame:
    parts = []
    for slug, part in hourly.groupby("location"):
        base = origin_table[origin_table["location"] == slug]
        pieces = [base, target_calendar(part, horizon)]
        if use_future_weather:
            pieces.append(future_weather_features(part.drop(columns=["location"]), horizon))
        table = pd.concat(pieces, axis=1)
        table["target"] = part["pm2_5"].shift(-horizon)
        table["horizon"] = horizon
        parts.append(table)
    return pd.concat(parts)


def feature_columns(table: pd.DataFrame) -> list[str]:
    drop = {"target", "horizon", "location"}
    return [c for c in table.columns if c not in drop]


def future_only_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in table.columns if c.startswith("fw_")]


def feature_dictionary() -> pd.DataFrame:
    rows = [
        ("pm25_now, pm25_lag{k}", "PM2.5 at the origin and k hours earlier (k in 1,2,3,6,12,24,48,72,168)", "past"),
        ("pm25_mean/max/std{w}", "Rolling mean, max, std of PM2.5 over the last w hours (3,6,12,24,72)", "past"),
        ("pm25_change6/24", "PM2.5 now minus 6 or 24 hours ago", "past"),
        (
            "pm10, nitrogen_dioxide, carbon_monoxide _now/_change6",
            "Other pollutants at the origin and their 6-hour change",
            "past",
        ),
        ("pm_ratio", "PM2.5 divided by PM10 at the origin", "past"),
        ("weather *_now", "Temperature, humidity, pressure, rain, wind, cloud, wind u/v at the origin", "past"),
        ("pressure_tendency6, wind_change6, rain_sum24", "Recent weather change features", "past"),
        (
            "now_* calendar",
            "Hour, weekday, month (cyclical), Diwali window, stubble season, holiday at the origin (IST)",
            "past",
        ),
        ("tgt_* calendar", "Same calendar fields for the target hour; known in advance", "deterministic"),
        (
            "fw_* weather",
            "Weather at the target hour and mean wind / total rain until then; forecast weather at serving time, actual ERA5 in training",
            "future (forecast)",
        ),
        ("location_code", "Integer id of the grid point", "static"),
    ]
    return pd.DataFrame(rows, columns=["feature", "meaning", "availability"])
