from __future__ import annotations

import httpx
import pytest

from flight_delay.data.weather_client import WeatherClient


@pytest.mark.asyncio
async def test_weather_client_validates_http_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["timezone"] == "GMT"
        return httpx.Response(200, json={"hourly": {"time": ["2027-01-01T00:00"]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        payload = await WeatherClient(client).fetch_live_forecast(40.6, -73.7)
    assert "hourly" in payload

