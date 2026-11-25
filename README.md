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

## What V2.1 Audited & Proved

1. **Unified Refit-Full-Train Early Stopping Parity (`docs/early_stopping_parity.md`)**: Switching final fold evaluation from inner early-stopping probes (`probe_only`) to full-fold refit at `best_iteration` (`refit_full_train`) lifted XGBoost PR-AUC by `+0.003083 [0.001598, 0.004919]` and LightGBM by `+0.003942 [0.002021, 0.006062]`.
2. **Equal-Budget Feature & Model Tournament (`configs/best_v2_1/selection_summary.json`, `reports/metrics/v2_1_model_comparison.csv`)**: With identical 10-trial Optuna budgets, `F3` achieved `0.342768` mean PR-AUC (worst fold `0.264426`) over `F0` (`0.340358`, worst `0.255867`); across models (`18` XGBoost, `16` LightGBM, `3` CatBoost trials under wall-clock caps plus HistGB/LogReg/Dummy), `xgboost` (`none` weighting) led LightGBM (`+0.004027 [+0.001445, +0.007113]`), HistGB (`+0.008584`), LogReg (`+0.012172`), `scale_pos_weight=balanced` (`+0.003179`, Brier `0.2193`), and CatBoost (`ROC +0.001962`).
3. **Operational Error, Threshold & Cost Policy (`docs/error_analysis_v2_1.md`, `docs/threshold_policy_v2_1.md`)**: Spring `max_f1 = 0.185` saturated summer `2025-06..08` alerts at `62.4%` (`122,622` false alerts per 300k flights), whereas spring `precision_target_35 = 0.285` halved false alerts (`57,577`, alert rate `33.7%`, `P = 0.4304`) and matched the global `2:1` cost optimum (`0.42280` vs `0.42256` at `0.290`). Top `T-24h` errors cluster in unpredictable evening LCC hubs vs morning spoke ATC/rotation incidents.
4. **Honest Negative & Research Ablations (`docs/weather_release_provenance.md`, `reports/ablation/historical_priors_asof.csv`)**:
   - Strict 2-month lagged BTS historical priors gained `+0.000279 [-0.001506, +0.002299]` over `F3` — rejected for serving simplicity.
   - `ecmwf_ifs025` `_previous_day2` weather (`2024-04..2025-08`) lifted PR-AUC on both matched top-25 routes (`+0.013053 [+0.009874, +0.015969]`) and full 1.0M fallback validation (`+0.012047 [+0.009478, +0.014908]`), but only covered `28–30%` of both-endpoint routes (excluding `G4`/`HA` `0%`, `MQ`/`OO` `6–8%`), so production remains **flight-only `F3`** while `F5` ships with tested graceful fallback (`resolve_t24_weather_features`, `tests/test_weather_serving_fallback.py`).
5. **Delayed-Label Monitoring Catching Calibration Drift (`docs/drift_report.md`, `docs/retraining_trigger_policy.md`)**: On `2025-10..11`, schedule covariates remained flat (`distance PSI = 0.00046`, `crs_elapsed_time PSI = 0.00226`, `unseen_route_rate = 0.0069`), while delayed BTS labels caught autumn calibration shift (`ECE = 0.0430 >= 0.04` in both Oct and Nov), automatically triggering `retraining_recommended = true`.

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
