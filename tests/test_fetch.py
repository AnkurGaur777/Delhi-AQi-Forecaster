from datetime import date
from unittest import mock

import pandas as pd
import pytest

from aqi import config, fetch


def payload(start: str, hours: int) -> dict:
    times = pd.date_range(start, periods=hours, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {"hourly": {"time": times, "pm2_5": list(range(hours))}}


def test_year_chunks_split_at_new_year():
    chunks = fetch.year_chunks(date(2023, 1, 1), date(2025, 3, 10))
    assert [c[0].year for c in chunks] == [2023, 2024, 2025]
    assert chunks[-1][1] == date(2025, 3, 10)


def test_hourly_frame_is_utc_indexed():
    frame = fetch.hourly_frame(payload("2025-01-01", 5))
    assert str(frame.index.tz) == "UTC"
    assert frame["pm2_5"].tolist() == [0, 1, 2, 3, 4]


def test_hourly_frame_rejects_empty_payload():
    with pytest.raises(ValueError):
        fetch.hourly_frame({})


def test_cached_chunks_are_not_downloaded_again(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "REQUEST_PAUSE_SECONDS", 0)
    calls = []

    def fake_fetch(kind, lat, lon, start, end):
        calls.append((kind, start.year))
        return fetch.hourly_frame(payload(f"{start.year}-01-01", 3))

    monkeypatch.setattr(fetch, "fetch_chunk", fake_fetch)
    first = fetch.download_history(years=2, refresh_partial=False)
    second = fetch.download_history(years=2, refresh_partial=False)
    assert first["downloaded"] > 0
    assert second["downloaded"] == 0
    assert second["skipped"] == first["downloaded"]


def test_request_retries_then_fails(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    with mock.patch.object(fetch.requests, "get", side_effect=fetch.requests.ConnectionError("down")) as get:
        with pytest.raises(RuntimeError):
            fetch.request_json("http://example.test", {})
    assert get.call_count == config.MAX_RETRIES
