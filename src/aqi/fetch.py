import json
import logging
import time
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import requests

from aqi import config

log = logging.getLogger(__name__)

SERVICES = {
    "air": (config.AIR_QUALITY_URL, config.AIR_VARIABLES),
    "weather": (config.ARCHIVE_URL, config.WEATHER_VARIABLES),
}


def year_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    cursor = start
    while cursor <= end:
        chunk_end = min(date(cursor.year, 12, 31), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def request_json(url: str, params: dict) -> dict:
    last_error: Exception | None = None
    for attempt in range(config.MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=60)
            if response.status_code == 429:
                raise requests.HTTPError("rate limited (429)")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            wait = 2**attempt * 2
            log.warning("request failed (%s), retrying in %ss", error, wait)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {url} after {config.MAX_RETRIES} tries: {last_error}")


def hourly_frame(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise ValueError("response has no hourly block")
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    return frame.set_index("time").sort_index()


def chunk_path(kind: str, slug: str, start: date):
    return config.RAW_DIR / kind / slug / f"{start.year}.parquet"


def fetch_chunk(kind: str, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    url, variables = SERVICES[kind]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(variables),
        "timezone": "GMT",
    }
    return hourly_frame(request_json(url, params))


def download_history(years: int | None = None, refresh_partial: bool = True) -> dict:
    years = years or config.HISTORY_YEARS
    cutoff = date.today() - timedelta(days=config.ARCHIVE_LAG_DAYS)
    start = date(cutoff.year - years + 1, 1, 1)
    locations = config.load_locations()
    summary = {"downloaded": 0, "skipped": 0}

    for row in locations.itertuples():
        slug = config.location_slug(row.name)
        for kind in SERVICES:
            for chunk_start, chunk_end in year_chunks(start, cutoff):
                path = chunk_path(kind, slug, chunk_start)
                is_partial = chunk_end >= cutoff
                if path.exists() and not (is_partial and refresh_partial):
                    summary["skipped"] += 1
                    continue
                log.info("fetching %s %s %s", kind, slug, chunk_start.year)
                frame = fetch_chunk(kind, row.latitude, row.longitude, chunk_start, chunk_end)
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path)
                summary["downloaded"] += 1
                time.sleep(config.REQUEST_PAUSE_SECONDS)

    write_metadata(start, cutoff, years)
    return summary


def write_metadata(start: date, end: date, years: int) -> None:
    meta = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "years": years,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "Open-Meteo (CAMS air quality, ERA5 weather)",
    }
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_raw(kind: str, slug: str) -> pd.DataFrame:
    folder = config.RAW_DIR / kind / slug
    files = sorted(folder.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no cached {kind} data for {slug}; run the download first")
    frame = pd.concat([pd.read_parquet(f) for f in files])
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def fetch_recent(
    lat: float, lon: float, past_days: int = 8, forecast_days: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    air_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(config.AIR_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "GMT",
    }
    weather_params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(config.WEATHER_VARIABLES),
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "GMT",
    }
    air = hourly_frame(request_json(config.AIR_QUALITY_URL, air_params))
    weather = hourly_frame(request_json(config.FORECAST_URL, weather_params))
    return air, weather


def recent_path(slug: str, kind: str):
    return config.RAW_DIR / "recent" / f"{slug}_{kind}.parquet"


def refresh_recent() -> None:
    for row in config.load_locations().itertuples():
        slug = config.location_slug(row.name)
        air, weather = fetch_recent(row.latitude, row.longitude)
        recent_path(slug, "air").parent.mkdir(parents=True, exist_ok=True)
        air.to_parquet(recent_path(slug, "air"))
        weather.to_parquet(recent_path(slug, "weather"))
        time.sleep(config.REQUEST_PAUSE_SECONDS)


def load_recent(slug: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    air_file, weather_file = recent_path(slug, "air"), recent_path(slug, "weather")
    if not air_file.exists() or not weather_file.exists():
        raise FileNotFoundError(f"no recent cache for {slug}; run refresh first")
    return pd.read_parquet(air_file), pd.read_parquet(weather_file)
