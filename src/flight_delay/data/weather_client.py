"""Reliable live Open-Meteo client; not used for offline benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class WeatherSnapshot:
    provider: str
    retrieval_time_utc: datetime
    forecast_valid_time_utc: datetime
    values: dict[str, float | int | None]


class WeatherClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        self._external_client = client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(multiplier=0.25, max=2.0),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError)),
        reraise=True,
    )
    async def fetch_live_forecast(self, latitude: float, longitude: float) -> dict[str, Any]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m,cloud_cover",
            "timezone": "GMT",
            "forecast_days": 3,
        }
        if self._external_client is not None:
            response = await self._external_client.get(FORECAST_URL, params=params)
            response.raise_for_status()
            return response.json()
        async with httpx.AsyncClient(timeout=12.0) as client:
            response = await client.get(FORECAST_URL, params=params)
            response.raise_for_status()
            return response.json()

