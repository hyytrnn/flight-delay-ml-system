# Deployment

## Local API

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn flight_delay.api.main:app --host 127.0.0.1 --port 8000
```

Readiness is available at `/health`, model metadata at `/model-info`, Swagger at
`/docs`, single prediction at `/predict`, bounded batch prediction at
`/predict/batch`, and Prometheus text metrics at `/metrics`.

## Docker

```powershell
docker build -t flight-delay-api:1.0.0 .
docker run --rm -p 8000:8000 flight-delay-api:1.0.0
```

The image runs as a non-root user and contains the trusted model artifact built by
this repository. Raw BTS data and local reports are excluded from the build
context. The API returns HTTP 503 when the artifact cannot be loaded.

Docker was not installed on the Windows development machine. The repository's
GitHub Actions workflow performs the authoritative Docker build on an Ubuntu
runner after push.

