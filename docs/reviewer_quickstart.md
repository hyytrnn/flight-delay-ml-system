# Reviewer Quickstart (`V2.2`)

This guide lets any reviewer verify the active **V2.1** production model (`flight-only-v2.1.0-xgboost`, `F3`) and **V2.2** reliability/demo additions on a fresh clone **without BTS raw archives, Open-Meteo credentials, or API keys**.

> **Portfolio Boundary**: This is a portfolio benchmark and educational demo built on sampled BTS data (`T-24h` schedule-proxy contract), not an operational airline dispatch system.

## 1. Local Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 2. Verify Fresh-Clone Parity, Tests & Lint

```powershell
python -m ruff check src scripts tests
python -m pytest --cov=flight_delay --cov-report=term-missing
python scripts/verify_fresh_clone_v2_1.py
```

`verify_fresh_clone_v2_1.py` confirms:
- SHA-256 match for `models/flight_delay_v2_1_bundle.joblib` (`8b48e8f0b730ac09d11536f4cf7e206962e116a0fcd4f762b02357c59a83d7f6`)
- `GET /health` -> `200`, `model_version = flight-only-v2.1.0-xgboost`, `weather_provider_status = not_required`
- Exact `POST /predict` probability parity between `UTC` and `-05:00` offsets
- Unseen route fallback (`route_status = fallback_global_median`)
- Fail-closed `422` for unknown origin airport or `prediction_at > scheduled_departure - 24h`

## 3. Run the Local API & Interactive Demo UI

Start FastAPI:
```powershell
uvicorn flight_delay.api.main:app --host 127.0.0.1 --port 8000
```

Then open either:
- **Interactive Demo UI**: [`demo/index.html`](../demo/index.html) directly in a browser (includes 1-click synthetic presets from [`data/examples/sample_flights.json`](../data/examples/sample_flights.json): normal `JFK -> LAX`, DST boundary, unseen route fallback, and invalid `T-24h` violation check)
- **Swagger UI**: `http://127.0.0.1:8000/docs`

## 4. Run via Docker (`python:3.11-slim` + `libgomp1`)

```bash
docker build -t flight-delay-api:2.2.0 .
docker run --rm -p 8000:8000 flight-delay-api:2.2.0
```
