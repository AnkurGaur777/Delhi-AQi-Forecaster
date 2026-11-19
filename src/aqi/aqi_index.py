import numpy as np
import pandas as pd

CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

# CPCB breakpoints for PM2.5 (24-hour average, ug/m3) and the matching AQI bands.
# The top band is capped at 380 ug/m3 -> AQI 500, a common convention for "250+".
PM25_BREAKPOINTS = [0, 30, 60, 90, 120, 250, 380]
AQI_BREAKPOINTS = [0, 50, 100, 200, 300, 400, 500]


def pm25_to_aqi(concentration):
    """Convert a 24-hour mean PM2.5 value to the Indian AQI sub-index."""
    values = np.clip(np.asarray(concentration, dtype=float), 0, PM25_BREAKPOINTS[-1])
    return np.interp(values, PM25_BREAKPOINTS, AQI_BREAKPOINTS)


def category_code(concentration):
    """Return the category index (0 = Good ... 5 = Severe) for 24-hour mean PM2.5."""
    values = np.asarray(concentration, dtype=float)
    codes = np.digitize(values, PM25_BREAKPOINTS[1:5], right=True)
    codes = np.where(values > PM25_BREAKPOINTS[5], 5, codes)
    return np.where(np.isnan(values), -1, codes).astype(int)


def category_name(code: int) -> str:
    return CATEGORIES[code] if 0 <= code < len(CATEGORIES) else "Unknown"


def rolling_24h_mean(series: pd.Series) -> pd.Series:
    # CPCB asks for at least 16 valid hours in the 24-hour window
    return series.rolling(24, min_periods=16).mean()
