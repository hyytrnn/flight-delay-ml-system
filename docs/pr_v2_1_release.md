# Pull Request Summary: `feature/v2.1-validation-weather` -> `main` (`v2.1.0`)

## 1. Executive Summary

This release promotes the **V2.1** (`flight-only-v2.1.0-xgboost`, feature group `F3`, contract `t24-v2.1-F3`) system to `main` alongside full statistical validation rigor, equal-budget feature/model comparisons, operational threshold/cost analysis, weather & historical-prior ablation audits, single prospective holdout verification on untouched `2025-10..2025-11` flights, default FastAPI/Docker serving switch, delayed-label drift monitoring, CI fresh-clone + Docker smoke checks, and release artifact automation.

Locked historical baselines remain SHA256-verified and unmodified (`models/locked_v2_manifest.json`):
- V1 (`flight-only-v1.0.0`): `33cef59201a640db870c00f51cb38dc3fb6fbec1b6b6b13128eb6de2a71926ee`
- V2 (`flight-only-v2.0.0-xgboost`): `e5098119a9449bcd9ccf35f42d954ea58c654d4a062bdb734a3b3d47e46a678c`
- **V2.1 (`flight-only-v2.1.0-xgboost`)**: `8b48e8f0b730ac09d11536f4cf7e206962e116a0fcd4f762b02357c59a83d7f6`

## 2. Gate-by-Gate Engineering & Verification Summary

- **G0 (`38d10e6`)**: Split & leakage governance (`configs/splits_v2_1.yaml`, `docs/evaluation_protocol_v2_1.md`), `HOLDOUT_EXCLUDED_MONTHS`, and corrected baseline / 3-paper citations in `docs/v2_research_report.md`.
- **G2 (`be146e4`)**: Exact grouped-bincount weighted calendar-block bootstrap CI (`calibrate_v2.py`), `eval_strategy="refit_full_train"` early-stopping parity audit (`docs/early_stopping_parity.md`: XGBoost `+0.003083 [0.001598, 0.004919]`, LightGBM `+0.003942 [0.002021, 0.006062]`), and read-only Optuna stale-trial audit.
- **G1 & G3 (`ec390a3`)**: Equal-budget 10-trial feature selection (`F3 = 0.342768`, worst fold `0.264426` vs `F0 = 0.340358`, worst `0.255867`) and time-budgeted multi-model tournament (`18` XGBoost, `16` LightGBM, `3` CatBoost trials + HistGB, LogReg, Dummy) selecting `xgboost` (`none`, `PR-AUC = 0.342768`, `ROC-AUC = 0.677582`, `Brier = 0.151285`).
- **G5 (`b1c8466`)**: Strict 2-month publication-lagged BTS priors rejected for serving (`+0.000279 [-0.001506, +0.002299]` over `F3`).
- **G4 (`e2a1fb1`)**: Monthly error, calibration & threshold/cost study across 19 validation months (`reports/eda_v2_1/`, `docs/error_analysis_v2_1.md`, `docs/threshold_policy_v2_1.md`).
- **G6 (`325d048`)**: Weather provenance & carrier selection-bias audit (`28.2–29.8%` both-matched top-25 coverage; Track A `+0.013053`, Track B 1.0M fallback `+0.012047`) plus optional `F5` serving fallback (`tests/test_weather_serving_fallback.py`).
- **G7–G8 (`001226d`, `25ec7b6`)**: Froze `models/flight_delay_v2_1_bundle.joblib` pre-holdout; evaluated untouched `2025-10..2025-11` once (`200,000` sampled from `1,156,865` eligible: **`PR-AUC = 0.3265 [0.2929, 0.3610]`**, `+65.81%` vs `0.1969` prevalence, `ROC-AUC = 0.6759`, `P@5% = 0.4262`, `P@10% = 0.4019`, `ECE = 0.0430`); switched default API/Docker serving and upgraded monitoring (`reports/monitoring/v2_1_latest.json`), CI/release workflows, model card, and `README.md`.

## 3. Pre-Push Verification Results

- `python -m ruff check src scripts tests` -> `All checks passed!`
- `python -m pytest --cov=flight_delay --cov-report=term-missing` -> `71 passed in 17.43s` (`80%` total coverage, zero skipped)
- `python -m compileall -q src scripts tests` -> exit `0`
- `python scripts/verify_fresh_clone_v2_1.py` -> `"status": "PASS"` (SHA256 match, UTC vs `-05:00` exact parity `0.10089181363582611`, unseen route `fallback_global_median`, fail-closed `422`, and V1 backward compatibility)
