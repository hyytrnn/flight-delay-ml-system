# V2 Deployment & Train-Serving Parity Guide

## Serving Contract (`t24-v2-F3`)

- **Bundle Path**: `models/flight_delay_v2_bundle.joblib` (loaded at startup via `FLIGHT_DELAY_MODEL_PATH`, default V1 fallback remains compatible when V1 bundle is loaded).
- **Timezone Fail-Closed Policy**: V2 bundles set `require_origin_timezone = True`. Input `scheduled_departure_utc` MUST be timezone-aware (UTC `'Z'` or explicit offset like `2027-02-09T08:30:00-05:00`) and `origin` MUST resolve to a known IANA timezone in `bundle.airport_timezones` / `airport_lookup`. Unknown origin airports return HTTP `422` rather than defaulting to UTC.
- **Train-Serving Feature Parity**: `ModelBundle.build_features_from_payload` computes identical local departure hours (`ZoneInfo` DST-safe), `crs_elapsed_time`, `distance`, airport coordinates/elevation deltas, cyclical encodings, and U.S. federal holiday proximity (`days_to_nearest_us_federal_holiday`, `is_us_federal_holiday`) via `flight_delay.features.v2`.
- **Unseen Routes**: New origin-dest pairs are flagged `route_status = "unseen"`, receive deterministic `haversine_distance_km` + airport metadata fallback (`crs_elapsed_time` median/estimate, `distance` haversine fallback when omitted), and keep `CategoricalFrameTransformer` unknown category handling.

## Verification Commands

```powershell
# API smoke check against V2 bundle (UTC vs offset parity verified)
.\.venv\Scripts\python.exe scripts/api_smoke_v2.py models/flight_delay_v2_bundle.joblib

# Run FastAPI locally
$env:FLIGHT_DELAY_MODEL_PATH="models/flight_delay_v2_bundle.joblib"
.\.venv\Scripts\uvicorn.exe flight_delay.api.main:app --host 0.0.0.0 --port 8000
```

Observed `api_smoke_v2.py` parity check on `models/flight_delay_v2_bundle.joblib`:
- `GET /health` -> `200`, `model_version = flight-only-v2.0.0-xgboost`, `weather_provider_status = not_required`
- `POST /predict` with `2027-02-09T13:30:00Z` vs `2027-02-09T08:30:00-05:00` -> identical `delay_probability = 0.10089181363582611`, `threshold = 0.185`, `feature_contract_version = t24-v2-F3`
- `GET /metrics` -> `200`

## Container Build

```bash
docker build -t flight-delay-ml-system:v2 .
docker run -p 8000:8000 -e FLIGHT_DELAY_MODEL_PATH=/app/models/flight_delay_v2_bundle.joblib -v ./models:/app/models flight-delay-ml-system:v2
```
