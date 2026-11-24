"""As-of weather forecast retrieval, provenance tracking, and UTC joins."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import numpy as np
import pandas as pd
import polars as pl
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
WEATHER_BASE_VARIABLES = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
)


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(multiplier=0.5, max=6.0),
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError)),
    reraise=True,
)
def fetch_previous_runs_airport(
    client: httpx.Client,
    *,
    airport: str,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    model: str = "ecmwf_ifs025",
    lead_days: int = 2,
    availability_lag_hours: int = 6,
    variables: tuple[str, ...] = WEATHER_BASE_VARIABLES,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Fetch issued previous-dayN forecast archive for one airport with full provenance."""

    suffix = f"_previous_day{lead_days}"
    hourly_vars = [f"{var}{suffix}" for var in variables]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_vars),
        "models": model,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "GMT",
    }
    response = client.get(PREVIOUS_RUNS_URL, params=params)
    response.raise_for_status()
    payload = response.json()
    retrieved_at = datetime.now(UTC)
    payload_sha256 = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    units = json.dumps(payload.get("hourly_units", {}), sort_keys=True)

    rows: list[dict[str, Any]] = []
    for idx, time_str in enumerate(times):
        valid_utc = datetime.fromisoformat(time_str).replace(tzinfo=UTC)
        run_init_utc = valid_utc - timedelta(days=lead_days)
        avail_utc = run_init_utc + timedelta(hours=availability_lag_hours)
        row: dict[str, Any] = {
            "airport": airport.upper(),
            "provider": "open-meteo-previous-runs",
            "model": model,
            "latitude": float(payload.get("latitude", latitude)),
            "longitude": float(payload.get("longitude", longitude)),
            "forecast_valid_time_utc": valid_utc,
            "run_initialization_utc": run_init_utc,
            "model_availability_time_utc": avail_utc,
            "retrieved_at_utc": retrieved_at,
            "units": units,
            "checksum": payload_sha256,
        }
        for var, api_col in zip(variables, hourly_vars, strict=True):
            series = hourly.get(api_col, [])
            val = series[idx] if idx < len(series) else None
            row[var] = float(val) if val is not None else np.nan
        rows.append(row)

    frame = pl.DataFrame(rows) if rows else pl.DataFrame()
    log_entry = {
        "airport": airport.upper(),
        "http_status": response.status_code,
        "rows": len(rows),
        "checksum": payload_sha256,
        "retrieved_at_utc": retrieved_at.isoformat(),
    }
    return frame, log_entry


def fetch_weather_cache_for_airports(
    airports_df: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    model: str = "ecmwf_ifs025",
    lead_days: int = 2,
    availability_lag_hours: int = 6,
    sleep_seconds: float = 0.25,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    frames: list[pl.DataFrame] = []
    logs: list[dict[str, Any]] = []
    with httpx.Client(timeout=30.0) as client:
        for row in airports_df.itertuples():
            df, log_entry = fetch_previous_runs_airport(
                client,
                airport=str(row.iata_code),
                latitude=float(row.latitude),
                longitude=float(row.longitude),
                start_date=start_date,
                end_date=end_date,
                model=model,
                lead_days=lead_days,
                availability_lag_hours=availability_lag_hours,
            )
            if not df.is_empty():
                frames.append(df)
            logs.append(log_entry)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    combined = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    return combined, logs


def join_weather_asof(
    flights: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    role: str,
    tolerance_minutes: int = 30,
    default_availability_lag_hours: int = 6,
) -> pd.DataFrame:
    """Join issued forecasts to flights strictly enforcing availability <= prediction_at_utc."""

    if role not in {"origin", "dest"}:
        raise ValueError(f"role must be 'origin' or 'dest', got {role}")
    airport_col = "origin" if role == "origin" else "dest"
    result = flights.copy()
    result["_flight_idx"] = np.arange(len(result))

    pred_at = pd.to_datetime(result["prediction_at_utc"], utc=True)
    sched_dep = pd.to_datetime(result["scheduled_departure_utc"], utc=True)
    if role == "origin":
        target_utc = sched_dep
    else:
        elapsed = pd.to_numeric(result["crs_elapsed_time"], errors="coerce").fillna(0.0)
        target_utc = sched_dep + pd.to_timedelta(elapsed, unit="m")
    result["_target_utc"] = target_utc
    result["_pred_at_utc"] = pred_at

    w = weather.copy()
    w["airport"] = w["airport"].astype(str).str.upper()
    w["forecast_valid_time_utc"] = pd.to_datetime(w["forecast_valid_time_utc"], utc=True)
    w["run_initialization_utc"] = pd.to_datetime(w["run_initialization_utc"], utc=True)
    if "model_availability_time_utc" in w.columns and w["model_availability_time_utc"].notna().all():
        w["model_availability_time_utc"] = pd.to_datetime(
            w["model_availability_time_utc"], utc=True
        )
    else:
        w["model_availability_time_utc"] = w["run_initialization_utc"] + pd.Timedelta(
            hours=default_availability_lag_hours
        )
    w["_valid_hour_utc"] = w["forecast_valid_time_utc"].dt.round("h")

    # Round target time to nearest hour so we can merge efficiently, then verify tolerance and availability
    result["_target_hour_utc"] = result["_target_utc"].dt.round("h")
    result["_airport_key"] = result[airport_col].astype(str).str.upper()

    merged = result[["_flight_idx", "_airport_key", "_target_utc", "_target_hour_utc", "_pred_at_utc"]].merge(
        w,
        left_on=["_airport_key", "_target_hour_utc"],
        right_on=["airport", "_valid_hour_utc"],
        how="left",
    )
    time_diff_min = (
        (merged["forecast_valid_time_utc"] - merged["_target_utc"]).abs().dt.total_seconds() / 60.0
    )
    legal = (
        merged["model_availability_time_utc"].notna()
        & (merged["model_availability_time_utc"] <= merged["_pred_at_utc"])
        & (time_diff_min <= float(tolerance_minutes))
    )
    valid_matches = merged.loc[legal].copy()
    valid_matches["_diff_min"] = time_diff_min.loc[legal]
    if not valid_matches.empty:
        valid_matches = valid_matches.sort_values(
            ["_flight_idx", "model_availability_time_utc", "_diff_min"],
            ascending=[True, False, True],
        ).drop_duplicates(subset=["_flight_idx"], keep="first")

    out_cols = {
        var: f"{role}_{var}" for var in WEATHER_BASE_VARIABLES if var in valid_matches.columns
    }
    matched_slim = valid_matches[["_flight_idx", "_diff_min", *out_cols.keys()]].rename(
        columns={"_diff_min": f"{role}_weather_mismatch_min", **out_cols}
    )
    joined = result.merge(matched_slim, on="_flight_idx", how="left")

    for var in WEATHER_BASE_VARIABLES:
        col = f"{role}_{var}"
        if col not in joined.columns:
            joined[col] = np.nan
    joined[f"{role}_weather_missing"] = (
        joined[f"{role}_temperature_2m"].isna().astype(int)
    )
    return joined.drop(columns=["_flight_idx", "_target_utc", "_pred_at_utc", "_target_hour_utc", "_airport_key"])
