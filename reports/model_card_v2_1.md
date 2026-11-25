# Model Card — Flight Delay ML System V2.1 (`flight-only-v2.1.0-xgboost`)

## 1. Model Identity & Artifact Provenance

- **Model Version**: `flight-only-v2.1.0-xgboost`
- **Production Bundle**: [flight_delay_v2_1_bundle.joblib](file:///D:/CV/flight-delay-ml-system/models/flight_delay_v2_1_bundle.joblib) (`884,607` bytes, SHA256 `8b48e8f0b730ac09d11536f4cf7e206962e116a0fcd4f762b02357c59a83d7f6`)
- **Manifest**: [manifest_v2_1.json](file:///D:/CV/flight-delay-ml-system/models/manifest_v2_1.json)
- **Locked Lineage**: V1 baseline (`flight-only-v1.0.0`, SHA256 `33cef59201a640db870c00f51cb38dc3fb6fbec1b6b6b13128eb6de2a71926ee`) and V2 release (`flight-only-v2.0.0-xgboost`, SHA256 `e5098119a9449bcd9ccf35f42d954ea58c654d4a062bdb734a3b3d47e46a678c`) preserved unmodified via `models/locked_v2_manifest.json`.
- **Model Type**: Calibrated-output gradient-boosted trees (`XGBClassifier` with native categorical `CategoricalFrameTransformer`, `F3` flight-only feature set).
- **Prediction Contract**: `t24-v2.1-F3` (`prediction_at_utc <= scheduled_departure_utc - 24h`, fail-closed origin airport IANA timezone lookup, binary BTS `DepDel15`: departure delay >= 15 min).

## 2. Intended Use & Out-of-Scope Boundaries

- **Intended Use**: Portfolio-grade, strictly leak-free `T-24h` U.S. domestic flight departure delay risk scoring and operational priority ranking (`Precision@Top5%`, `Precision@Top10%`, calibrated risk estimation).
- **Out-of-Scope / Non-Goals**:
  - Not a real-time airline dispatch control system (unobserved within-24h tail rotations, same-day crew legality, and FAA ATC ground-delay programs limit theoretical `T-24h` predictability).
  - Does **not** depend on live external weather APIs in default production inference (`weather_status = not_used_flight_only_v2_1`).
  - Uses deterministic `100,000`/month BTS stratified samples (`max_abs_sampling_positive_rate_delta = 0.003216` across dev months) rather than full un-sampled carrier rosters.

## 3. Chronological Splits & Governance

| Window | Date Span | Rows | Role |
|---|---|---:|---|
| Base Training | `2023-01-01 .. 2025-01-31` | `2,500,000` | Model fitting after fold validation & HP freeze |
| Historical V1 Reference | `2025-01-01 .. 2025-01-31` | `100,000` | Overlap reference check with V1 test period |
| Calibration Selection | `2025-02-01 .. 2025-03-31` | `200,000` | Compare `none`, `platt`, `isotonic` (`none` won: Brier `0.14714`, ECE `0.01680`) |
| Threshold Selection | `2025-04-01 .. 2025-05-31` | `200,000` | Select `max_f1 = 0.185` (`F1 = 0.4074`) & `precision_target_35 = 0.285` |
| Pre-Holdout Error/Cost Audit | `2025-06-01 .. 2025-08-31` | `300,000` | Error analysis, `2:1` cost curve (`0.290`), no re-tuning (`PR-AUC = 0.4419`) |
| V2 Reproducibility Check | `2025-09-01 .. 2025-09-30` | `100,000` | Viewed in V2 (`PR-AUC = 0.2950`, `ROC-AUC = 0.6868`); never tuned on in V2.1 |
| **V2.1 Prospective Holdout** | **`2025-10-01 .. 2025-11-30`** | **`200,000`** | **Downloaded/prepared & scored ONCE after freezing bundle SHA256 `8b48e8f...`** |

## 4. Development & Ablation Summary (`refit_full_train`, 4 Expanding Folds)

- **Early-Stopping Parity (`eval_strategy=refit_full_train` vs `probe_only`)**: Full-fold refit improved XGBoost `0.339104 -> 0.342768` (`+0.003083 [0.001598, 0.004919]`) and LightGBM `0.333825 -> 0.337582` (`+0.003942 [0.002021, 0.006062]`).
- **Equal-Budget Feature Selection (10 Optuna trials each)**: Selected `F3` (`mean PR-AUC = 0.342768`, `worst_fold = 0.264426`) over `F0` (`0.340358`, `worst_fold = 0.255867`) and `F2` (`0.341846`, `worst_fold = 0.264356`).
- **Multi-Model Tournament on `F3`**:
  - `xgboost` (`none`, `18` trials: `16` complete / `2` pruned in `2000s`): **`mean_pr_auc = 0.342768`**, `ROC-AUC = 0.677582`, `Brier = 0.151285`.
  - Outperformed `catboost` (`0.340253`, `3` trials in `900s`, paired ROC delta `+0.001962`), `lightgbm` (`0.337582`, `16` trials, paired PR delta `+0.004027`), `hist_gradient_boosting` (`0.333466`), `logistic_regression` (`0.329705`), and `xgboost` (`scale_pos_weight=balanced`, `0.341267`, `Brier = 0.219258`).
- **Offline Ablations Kept Outside Default Serving**:
  - **2-month lagged BTS priors**: Negligible incremental signal (`F3 + priors = 0.341679` vs `0.341370`, `+0.000279 [-0.001506, +0.002299]`). Rejected for serving simplicity.
  - **Issued `ecmwf_ifs025` `_previous_day2` weather (`F5`, 3 folds `2024-04..2025-08`)**: Significant uplift on Track A matched top-25 routes (`+0.013053 [+0.009874, +0.015969]`) and Track B 1.0M-row fallback (`+0.012047 [+0.009478, +0.014908]`), but limited to `28.15%–29.84%` both-endpoint coverage and missing in `2023-01..2024-03`, so isolated to optional/research fallback (`predict_request_with_weather`).

## 5. One-Time Prospective Holdout Results (`2025-10..2025-11`)

Source: [PROSPECTIVE_EVAL_ONCE.md](file:///D:/CV/flight-delay-ml-system/reports/holdout_v2_1/PROSPECTIVE_EVAL_ONCE.md), [v2_1_holdout_metrics.json](file:///D:/CV/flight-delay-ml-system/reports/holdout_v2_1/v2_1_holdout_metrics.json), [v2_1_monthly_holdout.csv](file:///D:/CV/flight-delay-ml-system/reports/holdout_v2_1/v2_1_monthly_holdout.csv), [v2_1_holdout_slices.csv](file:///D:/CV/flight-delay-ml-system/reports/holdout_v2_1/v2_1_holdout_slices.csv).

- **Combined `2025-10..11` (`200,000` sampled from `1,156,865` eligible flights, prevalence `0.1969`)**:
  - `PR-AUC`: **`0.3265`** (`+0.1296` / **`+65.81%`** uplift over prevalence, 95% calendar-block CI **`[0.2929, 0.3610]`**)
  - `ROC-AUC`: **`0.6759`** (95% CI **`[0.6653, 0.6872]`**)
  - At `max_f1` threshold `0.185`: `Precision = 0.3310`, `Recall = 0.4490`, `F1 = 0.3811`
  - At precision-targeted threshold `0.285`: `Precision = 0.4260`, `Recall = 0.1106`, `F1 = 0.1756`
  - Top-K alerts: `Precision@Top5% = 0.4262`, `Precision@Top10% = 0.4019`
  - Calibration: `Brier = 0.1511`, `LogLoss = 0.4730`, `ECE = 0.0430` (exceeds `0.04` monitoring threshold, documented in [drift_report.md](file:///D:/CV/flight-delay-ml-system/docs/drift_report.md))
- **Monthly & Slice Breakdown**:
  - `2025-10` (`prev = 0.1959`): `PR-AUC = 0.3384` (`[0.2990, 0.3709]`, `+72.74%`), `ROC-AUC = 0.6870`, `ECE = 0.0417`
  - `2025-11` (`prev = 0.1980`): `PR-AUC = 0.3122` (`[0.2619, 0.3615]`, `+57.70%`), `ROC-AUC = 0.6648`, `ECE = 0.0442`
  - Unseen routes (`route_known == False`, `n = 1,381`, `0.69%`): `PR-AUC = 0.2536` vs `0.1608` prevalence (`+57.78%`), `ROC-AUC = 0.6511`, `ECE = 0.0072`
  - Carrier spread: Highest ranking discrimination on high-utilization point-to-point/LCC schedules (`WN`: `0.4279`, `B6`: `0.3555`, `AA`: `0.3468`) vs regional feeders (`OO`: `0.2159`, `MQ`: `0.2231`, `AS`: `0.2382`).
- **Paired Check vs Locked V2**: `delta = 0.00000` (`PR-AUC = 0.3265` on both), confirming V2.1's `refit_full_train` search independently re-validated the same regularized `F3` XGBoost optimum (`configs/best_v2_1/xgboost_F3_final.yaml`) while adding governed error analysis, dual threshold policies, and automated drift/outcome monitoring.
