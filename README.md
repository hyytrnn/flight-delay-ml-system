# Flight Delay Prediction ML System

End-to-end machine-learning system that predicts whether a U.S. domestic flight
will depart at least 15 minutes late. The project uses official BTS Reporting
Carrier data, a fixed T-24h decision policy, chronological evaluation, shared
training/serving features, a FastAPI service, Docker, CI and monitoring evidence.

## Results

The locked test is January 2025 with 60,000 deterministically sampled eligible
flights. XGBoost was selected on 2024 validation PR-AUC, calibrated on October
2024, and evaluated once on the locked test.

| Metric | Locked test value |
|---|---:|
| PR-AUC | 0.2213 |
| ROC-AUC | 0.5841 |
| Precision | 0.2108 |
| Recall | 0.7989 |
| F1 | 0.3336 |
| Brier score | 0.1483 |
| Decision threshold | 0.135 |

The test positive rate is 0.1825, so the test PR-AUC uplift over a constant-prior
reference is 0.0388 absolute. Performance is modest and lower than validation,
which is documented as temporal shift rather than hidden through re-tuning.

Local API latency over 100 HTTP requests to Uvicorn was 25.2 ms p50 and 30.0 ms
p95 on the development machine; this is not a production SLO.

## Problem and leakage policy

- Target: BTS `DepDel15`, departure delay at least 15 minutes.
- Decision time: exactly 24 hours before scheduled departure, with five-minute API tolerance.
- Cancelled, diverted and missing-label flights are excluded and reported.
- Actual departure/arrival times, delay minutes, taxi/wheels fields, arrival delay
  and delay causes are blocked from model inputs.
- BTS schedule fields are a retrospective proxy because BTS does not provide their
  original schedule-publication timestamp.
- Weather is not used in offline metrics because one uniform issued-run archive is
  not available across the complete 2023-2025 benchmark window.

## Data and partitions

Eight official monthly BTS ZIP archives produced 4,638,240 raw rows and 4,528,598
eligible rows before deterministic monthly sampling. The bounded benchmark has
480,000 rows:

| Partition | Months | Rows | Purpose |
|---|---|---:|---|
| Train | 2023-01, 04, 07, 10 | 240,000 | Fit preprocessing and models |
| Validation | 2024-01, 07 | 120,000 | Select model family |
| Calibration | 2024-10 | 60,000 | Calibrate probabilities and choose threshold |
| Locked test | 2025-01 | 60,000 | Final evaluation |

Raw data is not committed. The manifest records official URLs, selected columns,
file sizes and SHA-256 hashes.

## Architecture

```text
BTS monthly ZIPs + versioned airport timezones
                  |
          schema and quality checks
                  |
       clean Parquet + quarantine report
                  |
          shared feature builder
                  |
 train -> validation -> calibration -> locked test
                  |
      model + preprocessor + threshold + metadata
                  |
          FastAPI /predict and /health
                  |
       metrics, logs and drift report
```

## Reproduce on Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python scripts\build_airport_reference.py
python scripts\download_bts.py --months 2023-01 2023-04 2023-07 2023-10 2024-01 2024-07 2024-10 2025-01
python scripts\prepare_dataset.py --max-clean-per-file 60000 --max-quarantine-per-file 1000
python scripts\run_eda.py
python scripts\train.py
python scripts\error_analysis.py
python scripts\monitor.py
python -m pytest -q
```

## Run the API

```powershell
uvicorn flight_delay.api.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` or the static demo in `demo/index.html`.

```json
{
  "airline": "AA",
  "origin": "JFK",
  "destination": "LAX",
  "scheduled_departure": "2027-02-10T08:30:00-05:00",
  "prediction_at": "2027-02-09T08:30:00-05:00"
}
```

The verified local response returned probability `0.0859`, threshold `0.135`,
model version `flight-only-v1.0.0`, and `weather_status=not_used_flight_only_model`.

## Docker and quality checks

```powershell
docker build -t flight-delay-api:1.0.0 .
docker run --rm -p 8000:8000 flight-delay-api:1.0.0
python -m ruff check src scripts tests
python -m pytest --cov=flight_delay --cov-report=term-missing
python -m compileall -q src scripts tests
```

CI repeats lint, tests, coverage and compile checks on Python 3.11.

## Evidence

- [EDA findings](reports/eda_findings.md)
- [Model comparison](reports/metrics/model_comparison.csv)
- [Locked-test metrics](reports/metrics/test_metrics.json)
- [Model card](reports/model_card.md)
- [Error analysis](reports/error_analysis.md)
- [Monitoring report](reports/monitoring/weekly_report.json)
- [Data quality report](reports/data_quality.json)
- [Leakage audit](docs/leakage_audit.md)
- [Weather as-of policy](docs/weather_asof_policy.md)

## Limitations

The model is a portfolio benchmark, not an operational airline decision system.
It uses a bounded monthly sample, retrospective BTS schedule fields, no live
schedule feed and no weather feature. January 2025 performance shows temporal
degradation. Predictions for unseen routes use global median distance and duration
and are explicitly marked in the API response.

## Data sources

- [BTS Reporting Carrier On-Time Performance](https://www.transtats.bts.gov/DL_SelectFields.aspx?QO_fu146_anzr=b0-gvzr&gnoyr_VQ=FGJ)
- [BTS pre-zipped archives](https://www.transtats.bts.gov/PREZIP/)
- [Open-Meteo Previous Runs documentation](https://open-meteo.com/en/docs/previous-runs-api)
- [Open-Meteo Single Runs documentation](https://open-meteo.com/en/docs/single-runs-api)
