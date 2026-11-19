import json
import logging
import os
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd

from aqi import aqi_index, clean, config, features, fetch
from aqi.train_final import BUNDLE_NAME

log = logging.getLogger(__name__)
STALE_AFTER_HOURS = 6


def load_bundle() -> dict:
    path = config.MODEL_DIR / BUNDLE_NAME
    if not path.exists():
        raise FileNotFoundError("no trained model found; run `make train` first")
    return joblib.load(path)


def merged_frame(slug: str, refresh: bool = False) -> tuple[pd.DataFrame, bool]:
    """Recent observations plus forecast weather. Returns (frame, used_cache_only)."""
    used_cache = True
    if refresh:
        try:
            fetch.refresh_recent()
            used_cache = False
        except Exception as error:
            log.warning("live refresh failed, using cached data: %s", error)
    air, weather = fetch.load_recent(slug)
    history = clean.load_processed()
    history = history[history["location"] == slug].drop(columns=["location", "pm25_imputed"])
    joined = air.join(weather, how="outer")
    joined = joined.combine_first(history[history.index < joined.index.min()])
    joined = joined.sort_index()
    full = pd.date_range(joined.index.min(), joined.index.max(), freq="h", name="time")
    joined = joined.reindex(full)
    return joined, used_cache


def last_observed(frame: pd.DataFrame, now=None) -> pd.Timestamp:
    """Latest hour with a PM2.5 value that is not in the future."""
    reference = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    valid = frame["pm2_5"][frame.index <= reference.floor("h")].dropna()
    return valid.index[-1]


def origin_row(frame: pd.DataFrame, origin: pd.Timestamp, horizon: int, location_code: int) -> pd.DataFrame:
    window = frame.loc[origin - pd.Timedelta(hours=200) : origin + pd.Timedelta(hours=horizon)].copy()
    window.loc[window.index > origin, config.AIR_VARIABLES] = np.nan
    window = window.astype(float).interpolate(limit=3, limit_area="inside")
    base = features.past_features(window)
    base["location_code"] = location_code
    pieces = [base, features.target_calendar(window, horizon), features.future_weather_features(window, horizon)]
    return pd.concat(pieces, axis=1).loc[[origin]]


def forecast_location(
    slug: str, bundle: dict | None = None, refresh: bool | None = None, now: datetime | None = None
) -> dict:
    bundle = bundle or load_bundle()
    if refresh is None:
        refresh = os.getenv("AQI_LIVE", "0") == "1"
    reference = now or datetime.now(UTC)
    frame, used_cache = merged_frame(slug, refresh)
    origin = last_observed(frame, reference)
    code = features.location_codes()[slug]
    anchors = bundle["meta"]["horizons"]

    point, lo, hi = [], [], []
    for h in anchors:
        entry = bundle["fitted"][h]
        row = origin_row(frame, origin, h, code)[entry["columns"]]
        point.append(float(entry["point"].predict(row)[0]))
        scale = entry.get("interval_scale", 1.0)
        raw_lo, raw_hi = float(entry["lo"].predict(row)[0]), float(entry["hi"].predict(row)[0])
        centre = point[-1]
        lo.append(max(centre - (centre - min(raw_lo, centre)) * scale, 0.0))
        hi.append(centre + (max(raw_hi, centre) - centre) * scale)
    point, lo, hi = np.array(point), np.array(lo), np.array(hi)

    observed = frame["pm2_5"].loc[origin - pd.Timedelta(hours=23) : origin].to_numpy()
    steps = np.arange(1, config.FORECAST_STEPS + 1)
    now_value = float(observed[-1])
    pm = np.interp(steps, [0] + anchors, np.concatenate([[now_value], point]))
    pm_lo = np.interp(steps, [0] + anchors, np.concatenate([[now_value], lo]))
    pm_hi = np.interp(steps, [0] + anchors, np.concatenate([[now_value], hi]))
    path = np.concatenate([observed, pm])
    rolling = np.array([path[i + 1 : i + 25].mean() for i in range(len(pm))])

    current_mean = float(np.nanmean(observed[-24:]))
    times = [origin + pd.Timedelta(hours=int(s)) for s in steps]
    age_hours = (pd.Timestamp(reference) - origin).total_seconds() / 3600
    hours = [
        {
            "time": t.isoformat(),
            "pm25": round(float(v), 1),
            "pm25_lo": round(float(a), 1),
            "pm25_hi": round(float(b), 1),
            "aqi": int(round(float(aqi_index.pm25_to_aqi(r)))),
            "category": aqi_index.category_name(int(aqi_index.category_code(r))),
            "category_code": int(aqi_index.category_code(r)),
        }
        for t, v, a, b, r in zip(times, pm, pm_lo, pm_hi, rolling)
    ]
    return {
        "location": slug,
        "latest_data_time": origin.isoformat(),
        "data_age_hours": round(float(age_hours), 1),
        "stale": bool(age_hours > STALE_AFTER_HOURS),
        "used_cached_data_only": used_cache,
        "model_version": bundle["meta"]["model_version"],
        "data_source": bundle["meta"]["data_source"],
        "current": {
            "pm25": round(now_value, 1),
            "aqi": int(round(float(aqi_index.pm25_to_aqi(current_mean)))),
            "category": aqi_index.category_name(int(aqi_index.category_code(current_mean))),
            "category_code": int(aqi_index.category_code(current_mean)),
        },
        "forecast": hours,
    }


def data_metadata() -> dict:
    path = config.RAW_DIR / "metadata.json"
    return json.loads(path.read_text()) if path.exists() else {}


def worst_and_best(forecast: list[dict]) -> dict:
    worst = max(forecast, key=lambda r: (r["category_code"], r["aqi"]))
    best = min(forecast, key=lambda r: (r["category_code"], r["aqi"]))
    return {"worst": worst, "best": best}


def best_window(forecast: list[dict], hours: int = 3) -> dict | None:
    """Consecutive stretch with the lowest mean hourly PM2.5."""
    if len(forecast) < hours:
        return None
    means = [np.mean([r["pm25"] for r in forecast[i : i + hours]]) for i in range(len(forecast) - hours + 1)]
    start = int(np.argmin(means))
    return {
        "start": forecast[start]["time"],
        "end": forecast[start + hours - 1]["time"],
        "mean_pm25": float(means[start]),
    }


ADVICE = {
    0: "Air quality is good. Normal outdoor activity is generally fine.",
    1: "Air quality is satisfactory. Sensitive people may want to take it easier outdoors.",
    2: "Moderate air quality. Consider shorter outdoor activity if you are sensitive to air pollution.",
    3: "Poor air quality. Consider limiting prolonged outdoor exercise.",
    4: "Very poor air quality. Consider avoiding outdoor exertion and keeping windows closed.",
    5: "Severe air quality. Consider staying indoors where possible.",
}


def advice_for(code: int) -> str:
    return ADVICE.get(code, "No advice available.")
