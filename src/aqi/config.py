import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / os.getenv("AQI_DATA_DIR", "data")
MODEL_DIR = ROOT / os.getenv("AQI_MODEL_DIR", "models")
CONFIG_DIR = ROOT / "config"
REPORTS_DIR = ROOT / "reports"

RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

HISTORY_YEARS = int(os.getenv("AQI_HISTORY_YEARS", "3"))
ARCHIVE_LAG_DAYS = 6
REQUEST_PAUSE_SECONDS = 1.0
MAX_RETRIES = 5

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AIR_VARIABLES = [
    "pm2_5",
    "pm10",
    "nitrogen_dioxide",
    "sulphur_dioxide",
    "carbon_monoxide",
    "ozone",
]
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
]

HORIZONS = [1, 3, 6, 12, 24, 36, 48]
SERVING_HORIZONS = [1, *range(3, 49, 3)]
TARGET = "pm2_5"
FORECAST_STEPS = 48

STUBBLE_START = (10, 15)
STUBBLE_END = (11, 15)
DIWALI_WINDOW_DAYS = 3


def load_locations() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "locations.csv")


def load_festivals() -> pd.DataFrame:
    return pd.read_csv(CONFIG_DIR / "festivals.csv", parse_dates=["date"])


def location_slug(name: str) -> str:
    return name.lower().replace(" ", "_")
