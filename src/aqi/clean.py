import logging

import numpy as np
import pandas as pd

from aqi import config, fetch

log = logging.getLogger(__name__)

MAX_GAP_HOURS = 3
MAX_PM25 = 1000.0
FLATLINE_HOURS = 24


def build_location_frame(slug: str) -> pd.DataFrame:
    air = fetch.load_raw("air", slug)
    weather = fetch.load_raw("weather", slug)
    joined = air.join(weather, how="outer")
    full_index = pd.date_range(joined.index.min(), joined.index.max(), freq="h", name="time")
    return joined.reindex(full_index)


def cap_impossible(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in config.AIR_VARIABLES:
        frame.loc[frame[column] < 0, column] = np.nan
    frame.loc[frame["pm2_5"] > MAX_PM25, "pm2_5"] = np.nan
    frame["relative_humidity_2m"] = frame["relative_humidity_2m"].clip(0, 100)
    frame["cloud_cover"] = frame["cloud_cover"].clip(0, 100)
    return frame


def fill_short_gaps(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["pm25_imputed"] = frame["pm2_5"].isna()
    filled = frame.interpolate(limit=MAX_GAP_HOURS, limit_area="inside")
    filled["pm25_imputed"] = frame["pm25_imputed"] & filled["pm2_5"].notna()
    return filled


def clean_location(slug: str) -> pd.DataFrame:
    frame = fill_short_gaps(cap_impossible(build_location_frame(slug)))
    frame["location"] = slug
    return frame


def clean_all() -> pd.DataFrame:
    frames = [clean_location(config.location_slug(name)) for name in config.load_locations()["name"]]
    combined = pd.concat(frames)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(config.PROCESSED_DIR / "hourly.parquet")
    log.info("cleaned %d rows for %d locations", len(combined), len(frames))
    return combined


def load_processed() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / "hourly.parquet")


def longest_gap(series: pd.Series) -> int:
    missing = series.isna().to_numpy()
    longest = current = 0
    for flag in missing:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def longest_flatline(series: pd.Series) -> int:
    runs = series.ne(series.shift()).cumsum()
    return int(series.groupby(runs).transform("size").max())


def audit(combined: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rows = []
    for slug, part in combined.groupby("location"):
        pm = part["pm2_5"]
        rows.append(
            {
                "location": slug,
                "start": part.index.min(),
                "end": part.index.max(),
                "hours": len(part),
                "missing_pm25": int(pm.isna().sum()),
                "imputed_pm25": int(part["pm25_imputed"].sum()),
                "longest_gap_h": longest_gap(pm),
                "longest_flatline_h": longest_flatline(pm.dropna()),
                "negative_values": int((pm < 0).sum()),
            }
        )
    wide = combined.reset_index().pivot(index="time", columns="location", values="pm2_5")
    return {"summary": pd.DataFrame(rows), "correlation": wide.corr()}


def monthly_profile(combined: pd.DataFrame) -> pd.Series:
    return combined.groupby(combined.index.month)["pm2_5"].mean()


def hourly_profile_ist(combined: pd.DataFrame) -> pd.Series:
    ist = combined.index.tz_convert("Asia/Kolkata")
    return combined.groupby(ist.hour)["pm2_5"].mean()
