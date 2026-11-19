import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from aqi import config, forecast
from aqi.api.schemas import ForecastResponse, HealthResponse, LocationsResponse

log = logging.getLogger("aqi.api")
CACHE_SECONDS = int(os.getenv("AQI_CACHE_SECONDS", "3600"))
LIVE_REFRESH = os.getenv("AQI_LIVE", "0") == "1"

state: dict = {"bundle": None, "cache": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["bundle"] = forecast.load_bundle()
    yield


app = FastAPI(title="Delhi AQI 48-hour forecaster", version="0.1.0", lifespan=lifespan)


def location_slugs() -> list[str]:
    return [config.location_slug(n) for n in config.load_locations()["name"]]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    meta = state["bundle"]["meta"]
    last = None
    try:
        frame, _ = forecast.merged_frame(location_slugs()[0])
        last = forecast.last_observed(frame).isoformat()
    except FileNotFoundError:
        log.warning("no recent data cache found")
    return HealthResponse(
        status="ok",
        model_version=meta["model_version"],
        trained_until=meta["trained_until"],
        data_source=meta["data_source"],
        last_data_time=last,
    )


@app.get("/locations", response_model=LocationsResponse)
def locations() -> LocationsResponse:
    return LocationsResponse(locations=location_slugs())


@app.get("/forecast", response_model=ForecastResponse)
def get_forecast(location: str = Query(..., description="location slug, e.g. near_ito")) -> ForecastResponse:
    if location not in location_slugs():
        raise HTTPException(status_code=404, detail=f"unknown location '{location}'")
    cached = state["cache"].get(location)
    if cached and time.time() - cached[0] < CACHE_SECONDS:
        return cached[1]
    try:
        result = forecast.forecast_location(location, state["bundle"], refresh=LIVE_REFRESH)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    response = ForecastResponse(**result)
    state["cache"][location] = (time.time(), response)
    return response
