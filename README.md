# Flight Delay Prediction ML System

This repository implements Milestone 1 of an end-to-end flight delay prediction
system: a reproducible, leakage-safe ingestion and cleaning pipeline for U.S. BTS
on-time performance data.

The frozen prediction policy is **T-24h before scheduled departure** and the target
is **DepDel15**. Post-departure values such as actual departure time, taxi-out,
arrival delay and delay-cause fields are outcome-only data and are never eligible
model inputs.

## Current status

- Problem, label, disruption and feature-availability policies are documented.
- CSV and ZIP ingestion, schema aliases, checksums, validation, quarantine and
  Parquet output are implemented.
- Synthetic smoke tests are executable and do not claim real BTS performance.
- Real BTS ingestion: **NOT EXECUTED** because no BTS input file was supplied.
- EDA, feature modeling, training, evaluation, API and deployment: **NOT STARTED**.

## Quick start on Windows PowerShell

Use Python 3.11 for the reproducible project environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
python scripts\smoke_test.py
```

To process real BTS files after downloading them:

```powershell
python scripts\prepare_data.py `
  --input data\raw\bts\2023\01\*.csv `
  --airports data\external\airports.csv `
  --output data\processed\clean_flights.parquet `
  --quarantine data\interim\quarantine.parquet `
  --report reports\data_quality.json `
  --manifest data\download_manifest.csv `
  --run-kind real
```

The command fails closed when required columns are missing. Invalid airport codes,
times, labels, disruptions and duplicate flight keys are quarantined with explicit
reasons instead of being silently repaired.

## Milestone 1 evidence

- Contract and policies: `docs/`
- Pipeline code: `src/flight_delay/data/`
- Leakage contract: `src/flight_delay/features/contracts.py`
- Tests: `tests/`
- Synthetic-only evidence: `reports/smoke/`

Do not add model metrics, latency, screenshots or deployment claims until the
corresponding commands have run and their artifacts are committed or released.

