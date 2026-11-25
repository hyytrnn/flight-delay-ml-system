"""Reliable live Open-Meteo client and T-24h fallback resolver."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np
import pandas as pd
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from flight_delay.data.weather_asof import WEATHER_BASE_VARIABLES

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_FEATURES = [
    *[f"origin_{v}" for v in WEATHER_BASE_VARIABLES],
    *[f"dest_{v}" for v in WEATHER_BASE_VARIABLES],
    "origin_weather_missing",
    "dest_weather_missing",
]


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


def _parse_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    parsed = pd.Timestamp(str(value))
    if pd.isna(parsed):
        return None
    dt = parsed.to_pydatetime()
    return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)


def _payload_available_before(
    payload: dict[str, Any],
    prediction_at_utc: datetime,
    assumed_availability_lag_hours: int,
) -> bool:
    avail = _parse_utc(payload.get("model_availability_time_utc") or payload.get("availability_time_utc"))
    if avail is not None:
        return avail <= prediction_at_utc
    run_init = _parse_utc(payload.get("run_initialization_utc"))
    if run_init is not None:
        return (run_init + timedelta(hours=assumed_availability_lag_hours)) <= prediction_at_utc
    retrieved = _parse_utc(payload.get("retrieval_time_utc"))
    if retrieved is not None:
        return retrieved <= prediction_at_utc
    return True


def _resolve_single_airport_weather(
    *,
    role: str,
    meta: dict[str, Any],
    target_utc: datetime,
    prediction_at_utc: datetime,
    fetcher: Callable[[float, float], dict[str, Any]] | None,
    assumed_availability_lag_hours: int,
    tolerance_minutes: int,
) -> tuple[dict[str, float], bool]:
    side: dict[str, float] = {f"{role}_{v}": float(np.nan) for v in WEATHER_BASE_VARIABLES}
    side[f"{role}_weather_missing"] = 1.0

    lat = meta.get("latitude")
    lon = meta.get("longitude")
    if fetcher is None or lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return side, False

    try:
        payload = fetcher(float(lat), float(lon))
        if not isinstance(payload, dict) or not _payload_available_before(
            payload, prediction_at_utc, assumed_availability_lag_hours
        ):
            return side, False
        hourly = payload.get("hourly") or {}
        times = hourly.get("time") or []
        best_idx: int | None = None
        best_diff = tolerance_minutes * 60.0 + 1.0
        for idx, raw_t in enumerate(times):
            dt = _parse_utc(raw_t)
            if dt is None:
                continue
            diff = abs((dt - target_utc).total_seconds())
            if diff <= tolerance_minutes * 60.0 and diff < best_diff:
                best_diff = diff
                best_idx = idx
        if best_idx is None:
            return side, False

        extracted: dict[str, float] = {}
        for var in WEATHER_BASE_VARIABLES:
            series = hourly.get(var) or []
            raw_val = series[best_idx] if best_idx < len(series) else None
            if raw_val is None or pd.isna(raw_val):
                return side, False
            extracted[f"{role}_{var}"] = float(raw_val)
    except (httpx.HTTPError, TimeoutError, OSError, ValueError, TypeError, KeyError, IndexError):
        return side, False

    side.update(extracted)
    side[f"{role}_weather_missing"] = 0.0
    return side, True


def resolve_t24_weather_features(
    *,
    origin_meta: dict[str, Any],
    dest_meta: dict[str, Any],
    scheduled_departure_utc: datetime,
    scheduled_arrival_utc: datetime,
    prediction_at_utc: datetime,
    fetcher: Callable[[float, float], dict[str, Any]] | None = None,
    assumed_availability_lag_hours: int = 6,
    tolerance_minutes: int = 30,
) -> tuple[dict[str, float], str]:
    """Resolve leakage-safe T-24h airport weather or degrade cleanly to NaN + missing flags."""
    dep_utc = _parse_utc(scheduled_departure_utc) or scheduled_departure_utc
    arr_utc = _parse_utc(scheduled_arrival_utc) or scheduled_arrival_utc
    pred_utc = _parse_utc(prediction_at_utc) or prediction_at_utc

    origin_side, origin_ok = _resolve_single_airport_weather(
        role="origin",
        meta=origin_meta,
        target_utc=dep_utc,
        prediction_at_utc=pred_utc,
        fetcher=fetcher,
        assumed_availability_lag_hours=assumed_availability_lag_hours,
        tolerance_minutes=tolerance_minutes,
    )
    dest_side, dest_ok = _resolve_single_airport_weather(
        role="dest",
        meta=dest_meta,
        target_utc=arr_utc,
        prediction_at_utc=pred_utc,
        fetcher=fetcher,
        assumed_availability_lag_hours=assumed_availability_lag_hours,
        tolerance_minutes=tolerance_minutes,
    )
    features = {**origin_side, **dest_side}
    status = "live_weather_ok" if (origin_ok and dest_ok) else "degraded_fallback_missing_weather"
    return features, status

