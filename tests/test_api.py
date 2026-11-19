import pytest
from fastapi.testclient import TestClient

from aqi import config
from aqi.api import main

pytestmark = pytest.mark.skipif(not (config.MODEL_DIR / "bundle.joblib").exists(), reason="trained model not available")


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_locations_lists_five(client):
    assert len(client.get("/locations").json()["locations"]) == 5


def test_forecast_schema(client):
    body = client.get("/forecast", params={"location": "near_ito"}).json()
    assert len(body["forecast"]) == 48
    first = body["forecast"][0]
    assert {"time", "pm25", "pm25_lo", "pm25_hi", "aqi", "category"} <= set(first)
    assert all(h["pm25_lo"] <= h["pm25"] <= h["pm25_hi"] for h in body["forecast"])
    assert "stale" in body and "data_age_hours" in body


def test_unknown_location_is_404(client):
    assert client.get("/forecast", params={"location": "atlantis"}).status_code == 404


def test_missing_location_is_422(client):
    assert client.get("/forecast").status_code == 422
