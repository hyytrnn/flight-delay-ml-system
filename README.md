# Flight Delay Prediction ML System (`V2.1` Active)

End-to-end, strictly leak-free **T-24h** U.S. domestic flight departure delay (`DepDel15 >= 15` minutes) prediction system built on official BTS Reporting Carrier On-Time Performance data (`3.4M` sampled flights across `35` continuous months `2023-01..2025-11`), engineered with shared training/serving feature builders, expanding-window Optuna validation, paired calendar-block bootstrap uncertainty quantification, FastAPI + Docker serving, GitHub Actions CI/Release automation, and delayed-label drift monitoring.

## Key Results & Verified Version Lineage

All prior model releases remain frozen and SHA256-locked (`models/locked_v2_manifest.json`). Each evaluation reflects its exact temporal evaluation contract:

| Version & Bundle | Model & Features | Evaluation Window (Role) | Rows | Prevalence | PR-AUC | Uplift vs Prev | ROC-AUC | F1 (@Thr) | Brier | ECE | P@Top5% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **V1** (`flight-only-v1.0.0`) | `F0` XGBoost (`thr=0.135`) | `2025-01` (Locked historical test) | 60,000 | 0.1825 | `0.2213` | `+0.0388 (+21.3%)` | `0.5841` | `0.3336` | `0.1483` | — | — |
| **V2** (`flight-only-v2.0.0-xgboost`) | `F3` XGBoost (`thr=0.185`) | `2025-09` (V2 viewed holdout) | 100,000 | 0.1655 | `0.2950` | `+0.1296 (+78.3%)` | `0.6868` | `0.3657` | `0.1298` | `0.0139` | `0.3952` |
| **V2.1** (`flight-only-v2.1.0-xgboost`) | `F3` XGBoost (`thr=0.185`) | `2025-06..08` (Pre-holdout val) | 300,000 | 0.2605 | `0.4419` | `+0.1815 (+69.7%)` | `0.7092` | `0.4873` | `0.1731` | `0.0207` | `0.5598` |
| **V2.1 (`2025-10` Prospective)** | `F3` XGBoost (`thr=0.185`) | **`2025-10` (Untouched Holdout)** | **100,000** | **0.1959** | **`0.3384`** | **`+0.1425 (+72.7%)`** | **`0.6870`** | **`0.3933`** | **`0.1491`** | `0.0417` | **`0.4534`** |
| **V2.1 (`2025-11` Prospective)** | `F3` XGBoost (`thr=0.185`) | **`2025-11` (Untouched Holdout)** | **100,000** | **0.1980** | **`0.3122`** | **`+0.1142 (+57.7%)`** | **`0.6648`** | **`0.3686`** | **`0.1532`** | `0.0442` | **`0.4108`** |
| **V2.1 Prospective Combined** | `F3` XGBoost (`thr=0.185` / `0.285`) | **`2025-10..11` (Evaluated ONCE)** | **200,000** | **0.1969** | **`0.3265`** | **`+0.1296 (+65.8%)`** | **`0.6759`** | `0.3811` / `P=0.4260` | **`0.1511`** | `0.0430` | **`0.4262`** |

> **Important disambiguation**: Percent PR-AUC uplifts (`+78.3%` in Sept, `+65.8%` in Oct–Nov) measure relative improvement over each month's own random-guessing `DepDel15` base rate, NOT cross-season comparisons against V1 (`2025-01`). V2.1 `95%` calendar-block bootstrap CI on `2025-10..11`: PR-AUC `[0.2929, 0.3610]`, ROC-AUC `[0.6653, 0.6872]`. Unseen routes (`n = 1,381`, `0.69%`) achieved `PR-AUC = 0.2536` vs `0.1608` prevalence (`+57.8%`).

## What V2.1 Improved Over the Prior Pipeline

### 1. Training, Feature Selection & Model Benchmark Improvements (`2023-01..2025-01`, 4 Expanding Folds)

| Pipeline Stage | Prior Baseline (`V1` / `V2`) | `V2.1` Improvement | Verified Delta / Impact | Evidence |
|---|---|---|---|---|
| **Early Stopping & Refit** | `probe_only` (evaluated inner early-stopping probe on `85%` train) | Unified `refit_full_train` (`100%` outer fold at `best_iteration`) | XGBoost PR-AUC **`+0.003083`** `[0.001598, 0.004919]`; LightGBM **`+0.003942`** `[0.002021, 0.006062]`; lower Brier (`-0.00037` / `-0.00041`) | [`docs/early_stopping_parity.md`](docs/early_stopping_parity.md) |
| **Fair Feature Selection** *(10 Optuna trials each)* | `F0` baseline (`0.340358` mean PR-AUC, worst fold `0.255867`) | `F3` calendar + carrier-route + geo (`0.342768` mean, worst `0.264426`) | **`+0.00241`** mean PR-AUC and **`+0.00856`** worst-fold robustness (`fold_4`) | [`configs/best_v2_1/selection_summary.json`](configs/best_v2_1/selection_summary.json) |
| **Multi-Model Tournament** *(on `F3`, time-budgeted)* | Single-track tuning (`18` XGB, `16` LGBM, `3` CatBoost + HistGB/LogReg/Dummy) | `xgboost` (`none` weighting): **`0.342768`** PR-AUC, `0.677582` ROC-AUC, `0.151285` Brier | Leads LightGBM (**`+0.004027`** `[+0.001445, +0.007113]`), HistGB (`+0.008584`), LogReg (`+0.012172`), `balanced` (`+0.003179`, Brier `0.2193`), CatBoost (`ROC +0.001962`) | [`reports/metrics/v2_1_model_comparison.csv`](reports/metrics/v2_1_model_comparison.csv) |

### 2. Operational Policy, Feature Ablations & Production Engineering

| Component | Before (`V2`) | `V2.1` Upgrade | Measured Outcome / Decision | Evidence |
|---|---|---|---|---|
| **Summer Alert Threshold** *(`2025-06..08`, 300k flights)* | Single `max_f1 = 0.185` (`62.4%` alert rate, `122,622` false alerts) | Added `precision_target_35 = 0.285` (`33.7%` alert rate, `57,577` false alerts, `P=0.4304`) | **Cuts false alerts by >50%** (`-65,045`), preserves F1 (`0.4855` vs `0.4873`), matches `2:1` cost optimum (`0.42280` vs `0.42256` at `0.290`) | [`docs/threshold_policy_v2_1.md`](docs/threshold_policy_v2_1.md) |
| **Lagged Historical Priors** *(2-mo BTS publication lag)* | Untested | Strict as-of smoothed priors (`first_day(M+3) <= prediction_at_utc`) | Only `+0.000279 [-0.001506, +0.002299]` over `F3` → **rejected** for serving simplicity | [`reports/ablation/historical_priors_asof.csv`](reports/ablation/historical_priors_asof.csv) |
| **Weather (`ecmwf_ifs025` `F5`)** *(`2024-04..2025-08`)* | Matched top-25 subset only | 3-fold Track A matched (**`+0.013053`**) + Track B 1.0M fallback (**`+0.012047`**) + serving fallback tests | Proved uplift, exposed `28–30%` both-endpoint coverage & carrier skew (`G4/HA 0%`, `OO 7.6%`) → keep default **`F3` (`100%` coverage)**, isolate `F5` with fallback | [`docs/weather_release_provenance.md`](docs/weather_release_provenance.md) |
| **Drift & Outcome Monitoring** | Covariate PSI only | Added score deciles/`score_psi` + delayed `DepDel15` `ECE`/`Brier`/`PR-AUC` alerts | Caught `2025-10..11` calibration shift (`ECE = 0.0430 >= 0.04` → `retraining_recommended = true`) while covariate PSI stayed flat (`<= 0.0023`) | [`docs/drift_report.md`](docs/drift_report.md) |
| **Serving, CI & Release** | Manual smoke, V1 default path | V2.1 default bundle, `verify_fresh_clone_v2_1.py`, `Dockerfile` + `libgomp1`, CI `docker run /health`, auto-release `v2.1.0` | Reproducible fresh-clone UTC/`-05:00` parity (`0.10089181363582611`), `422` fail-closed, V1 backward compatibility | [`docs/deployment_v2.md`](docs/deployment_v2.md) |

## Architecture & Serving Contract (`t24-v2.1-F3`)

```text
BTS Monthly PREZIP (2023-01..2025-11) + Airport IANA Timezones / Coordinates
                  |
       Schema, Quarantine & Leakage Guard (T-24h cutoff; post-outcome blocked)
                  |
   Shared `flight_delay.features.v2` (`F3`: calendar, carrier-route, airport geo)
                  |
Expanding Folds (`refit_full_train`) -> Calibrate (`2025-02..03`) -> Freeze Thresholds (`2025-04..05`)
                  |
    `models/flight_delay_v2_1_bundle.joblib` (SHA256-manifested, `F3` XGBoost)
                  |
  FastAPI (`/health`, `/predict`, `/predict/batch`, `/model-info`, `/metrics`)
                  |
   Delayed-Outcome & PSI Drift Monitor (`scripts/monitor.py` -> `reports/monitoring/`)
```

- **Default Bundle**: `FLIGHT_DELAY_MODEL_PATH=models/flight_delay_v2_1_bundle.joblib` (`flight-only-v2.1.0-xgboost`, backward compatible with V1 and V2 bundles via env var).
- **Fail-Closed Validation**: Requires timezone-aware timestamps with `prediction_at <= scheduled_departure - 24h` (5-minute tolerance) and known `origin` airport IANA timezone (`422` otherwise). Unknown `(origin|dest)` pairs score via `fallback_global_median` + haversine distance.

## Reproduce & Verify on Windows PowerShell

```powershell
# 1. Setup environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 2. Fresh-clone verification (manifest SHA256, API UTC/-05:00 parity, 422 checks, V1 compat)
python scripts/verify_fresh_clone_v2_1.py

# 3. Full quality suite
python -m ruff check src scripts tests
python -m pytest --cov=flight_delay --cov-report=term-missing
python -m compileall -q src scripts tests

# 4. Run FastAPI server locally
uvicorn flight_delay.api.main:app --host 127.0.0.1 --port 8000
```

## Docker Deployment (Linux `python:3.11-slim` + `libgomp1`)

```bash
docker build -t flight-delay-api:2.1.0 .
docker run --rm -p 8000:8000 flight-delay-api:2.1.0
```

Sample `/predict` request (UTC vs `-05:00` produce identical `delay_probability = 0.10089181363582611`):

```json
{
  "airline": "AA",
  "origin": "JFK",
  "destination": "LAX",
  "scheduled_departure": "2027-02-10T13:30:00Z",
  "prediction_at": "2027-02-09T13:30:00Z"
}
```

## Documentation & Evidence Map

- **Evaluation & Validation**: [`docs/evaluation_protocol_v2_1.md`](docs/evaluation_protocol_v2_1.md) · [`docs/early_stopping_parity.md`](docs/early_stopping_parity.md) · [`reports/holdout_v2_1/PROSPECTIVE_EVAL_ONCE.md`](reports/holdout_v2_1/PROSPECTIVE_EVAL_ONCE.md)
- **Model Card & CV Evidence**: [`reports/model_card_v2_1.md`](reports/model_card_v2_1.md) · [`docs/cv_claims_evidence.md`](docs/cv_claims_evidence.md) · [`reports/eda_v2_1/hypothesis_experiment_matrix.csv`](reports/eda_v2_1/hypothesis_experiment_matrix.csv)
- **Error, Thresholds & Weather**: [`docs/error_analysis_v2_1.md`](docs/error_analysis_v2_1.md) · [`docs/threshold_policy_v2_1.md`](docs/threshold_policy_v2_1.md) · [`docs/weather_release_provenance.md`](docs/weather_release_provenance.md)
- **Serving, Monitoring & Drift**: [`docs/deployment_v2.md`](docs/deployment_v2.md) · [`docs/monitoring_v2_1.md`](docs/monitoring_v2_1.md) · [`docs/drift_report.md`](docs/drift_report.md) · [`docs/retraining_trigger_policy.md`](docs/retraining_trigger_policy.md)
- **V1 & V2 Historical Artifacts**: [`CHANGELOG-V2.md`](CHANGELOG-V2.md) · [`docs/v2_research_report.md`](docs/v2_research_report.md) · [`reports/holdout/FINAL_ONCE.md`](reports/holdout/FINAL_ONCE.md)

## Limitations

- Portfolio benchmark built on deterministically sampled `100,000`-flight monthly BTS partitions, not a live airline tail/crew/ATC feed.
- BTS schedule fields (`CRSDepTime`, `CRSElapsedTime`) reflect post-publication archive tables rather than guaranteed T-24h snapshots.
- Default inference excludes weather (`F3` flight-only) for 100% route availability; `F5` weather fallback exists for research/optional bundles (`tests/test_weather_serving_fallback.py`).
- Prospective autumn calibration shift (`2025-10..11` `ECE = 0.0430`) shows annual/quarterly recalibration or retraining is required after summer-heavy training regimes.
