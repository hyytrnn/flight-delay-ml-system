# Deployment & Train-Serving Parity Guide (`V2.1` Active)

## 1. Active Production Serving Contract (`t24-v2.1-F3`)

- **Default Bundle Path**: `models/flight_delay_v2_1_bundle.joblib` (`FLIGHT_DELAY_MODEL_PATH` default in `src/flight_delay/api/main.py`, `Dockerfile`, and `docker-compose.yml`).
- **Model Version**: `flight-only-v2.1.0-xgboost` (`F3`, SHA256 `8b48e8f0b730ac09d11536f4cf7e206962e116a0fcd4f762b02357c59a83d7f6`).
- **Backward Compatibility**: Overriding `FLIGHT_DELAY_MODEL_PATH` with `models/flight_delay_v2_bundle.joblib` (V2) or `models/flight_delay_bundle.joblib` (V1) continues to serve cleanly without code changes (`tests/test_api.py`, `scripts/verify_fresh_clone_v2_1.py`).
- **Timezone Fail-Closed Policy**: `require_origin_timezone = True`. Requests must supply timezone-aware `scheduled_departure` and `prediction_at` (`<= scheduled_departure - 24h`, 5-minute tolerance), and `origin` must map to a known IANA timezone in `bundle.airport_timezones` / `airport_lookup`; unknown origin stations fail closed with HTTP `422`.
- **Train-Serving Feature Parity**: Both offline training (`scripts/evaluate_holdout_v2_1.py`) and live API inference (`ModelBundle._build_request_features`) call `flight_delay.features.v2.build_v2_feature_frame(..., "F3")` after localized `ZoneInfo` conversion.
- **Unseen Routes**: Unknown `(origin|dest)` pairs mark `route_status = "fallback_global_median"` and compute deterministic `haversine_distance_km` + global median route duration/distance while preserving airport coordinates/elevations where known.

## 2. Health, Logging & Optional `F5` Weather Fallback

- **`GET /health`**: Returns `status`, `model_loaded`, `model_version`, `feature_contract_version`, `bundle_path`, and `weather_provider_status = "not_required"`.
- **Structured JSON Request Logs**: Logger `flight_delay.api` emits `request_id`, `path`, `status`, `latency_ms`, and `model_version` on every request alongside Prometheus `/metrics`.
- **Optional `F5` Live-Weather Fallback (`src/flight_delay/data/weather_client.py` & `ModelBundle.predict_request_with_weather`)**:
  - Not enabled in the default flight-only serving path (`F3`, `100%` U.S. airport coverage).
  - For optional `F5` weather bundles, `resolve_t24_weather_features()` enforces `model_availability_time_utc <= prediction_at_utc` and degrades cleanly on HTTP timeouts, `429` rate limits, missing coordinates, or >30 min hourly mismatch by filling `NaN` + `*_weather_missing=1.0` with `weather_status="degraded_fallback_missing_weather"` (`tests/test_weather_serving_fallback.py`).

## 3. Verification & Container Commands

### Windows PowerShell (Fresh Clone Smoke + Local API)
```powershell
.\.venv\Scripts\python.exe scripts/verify_fresh_clone_v2_1.py
.\.venv\Scripts\uvicorn.exe flight_delay.api.main:app --host 127.0.0.1 --port 8000
```

Observed `verify_fresh_clone_v2_1.py` output against `models/flight_delay_v2_1_bundle.joblib`:
- `GET /health` -> `200`, `model_version = flight-only-v2.1.0-xgboost`, `feature_contract_version = t24-v2.1-F3`, `weather_provider_status = not_required`
- `POST /predict` (`2027-02-10T13:30:00Z` vs `2027-02-10T08:30:00-05:00`) -> identical `delay_probability = 0.10089181363582611`, `threshold = 0.185`
- Unseen route (`JFK -> ZZZ`) -> `200`, `route_status = fallback_global_median`
- Unknown origin (`ZZZ -> LAX`) -> `422` fail-closed

### Docker (Linux `python:3.11-slim` + `libgomp1`)
```bash
docker build -t flight-delay-api:2.1.0 .
docker run --rm -p 8000:8000 flight-delay-api:2.1.0
```
