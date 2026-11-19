from pydantic import BaseModel


class CurrentConditions(BaseModel):
    pm25: float
    aqi: int
    category: str
    category_code: int


class HourForecast(BaseModel):
    time: str
    pm25: float
    pm25_lo: float
    pm25_hi: float
    aqi: int
    category: str
    category_code: int


class ForecastResponse(BaseModel):
    location: str
    latest_data_time: str
    data_age_hours: float
    stale: bool
    used_cached_data_only: bool
    model_version: str
    data_source: str
    current: CurrentConditions
    forecast: list[HourForecast]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    trained_until: str
    data_source: str
    last_data_time: str | None


class LocationsResponse(BaseModel):
    locations: list[str]
