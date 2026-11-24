# Flight Delay V2 Research Report

## 1. Executive Summary & Benchmark Integrity

V2 upgrades the U.S. BTS `DepDel15` departure delay (`>= 15` minutes) system under a strict **T-24h operational contract**:

- **V1 Historical Baseline (Preserved Separately)**: Random-sampled 6-month subset, validation PR-AUC `0.3974`, Jan 2025 test PR-AUC `0.2213` (F1 `0.3336`, ROC-AUC `0.5841`, threshold `0.135`, prevalence `0.1825`). See [`reports/baseline/v1_verified_metrics.json`](../reports/baseline/v1_verified_metrics.json).
- **V2 Continuous Dataset**: 32 continuous monthly BTS partitions (`2023-01` through `2025-08`, `100,000` eligible rows/month = `3,200,000` development rows, global delay rate `0.2088`), with `2025-09` (`100,000` rows, `0.16545` prevalence) held strictly untouched until calibration and threshold freeze.
- **V2 Frozen Flight-Only Champion (`xgboost`, `F3`)**:
  - Rolling 4-fold Optuna CV (`2024-01..2024-12`): `mean_pr_auc = 0.33910`, `worst_fold = 0.26120`, `mean_roc_auc = 0.67527`, `mean_brier = 0.15166`.
  - Historical overlap reference (`2025-01`, `100,000` rows, `0.1829` prevalence): `PR-AUC = 0.2396` (`+0.0183` over V1 Jan 2025), `ROC-AUC = 0.6193` (`+0.0352` over V1).
  - Pre-holdout summer window (`2025-06..2025-08`, `300,000` rows, `0.2605` prevalence): `PR-AUC = 0.4419` (`+0.1815` over prevalence, `+69.7%` relative), `ROC-AUC = 0.7092`, `F1 = 0.4873`, `ECE = 0.0207`, `Precision@Top5% = 0.5598`.
  - **One-Time Untouched Holdout (`2025-09`, `100,000` rows, `0.1655` prevalence)**: `PR-AUC = 0.2950` (`+0.1296` over prevalence / `+78.31%` relative uplift, 95% calendar-block-bootstrap CI `[0.2661, 0.3169]`), `ROC-AUC = 0.6868` (95% CI `[0.6726, 0.6993]`), `F1 = 0.3657` (`precision = 0.2797`, `recall = 0.5280` at frozen threshold `0.185`), `Brier = 0.1298`, `LogLoss = 0.4197`, `ECE = 0.0139`, `Precision@Top5% = 0.3952`, `Precision@Top10% = 0.3587`.

## 2. Three-Paper Audit & Scientific Transfer

Three academic papers served strictly as methodological inspiration (`docs/paper_review_3_sources.md`, `reports/baseline/paper_transfer_audit.json`):

1. **Pineda-Jaramillo et al. (2024)** — Optuna hyperparameter tuning + tree ensembles + SHAP explainability: adapted into chronological expanding-window Optuna studies, `CatBoost`/`LightGBM`/`XGBoost` comparisons, and grouped SHAP + permutation explanations. Raw paper accuracies on South Florida operations are not directly comparable to national BTS `DepDel15`.
2. **Gui et al. (ADS-B / Air Traffic Flow)** — Separation of traffic-flow forecasting vs binary delay prediction (`SVR` / tree flow baselines): tested cleanly via hourly flow regression (`hist_gbr` RMSE `1.7024` vs 7-day seasonal naive `2.1322`) and downstream BTS `<= t-24h` flow lag ablation (`+0.00011` PR-AUC beyond `F3`), confirming retrospective full-day `P2_PROXY` congestion gains (`+0.0024` PR-AUC over `F0`) rely on retrospective schedule realization rather than 24h-lagged BTS sample counts.
3. **Ajala et al. (Weather + Resampling + Sensitivity)** — Preserved documented arithmetic audit discrepancy on Fig. 4 (`reported=0.73` vs `(48+44)/(48+18+24+44)=0.6866`), empirically refuted `scale_pos_weight=balanced` under `P5` (`XGBoost` PR-AUC `0.3375` vs `0.3391`, Brier `0.2189` vs `0.1517`), audited Sobol independence assumptions (`reports/explainability/sobol_if_valid.json`), and evaluated weather honestly only on matched issued-forecast coverage (`ecmwf_ifs025` `_previous_day2`, `2024-04..2025-08`).

## 3. Temporal EDA Findings (`2023-01..2025-08`)

- **Schedule Hour Dominance (H1)**: Delay rate rises monotonically from `7.52%` at 05:00 local (`n=91,611`) to `32.33%` at 20:00 local (`n=143,406`); `dep_hour` has `r = 0.189` and `MI = 0.0245`.
- **Seasonal & Monthly Non-Stationarity (H7)**: Fold prevalence shifts from `16.49%` (`fold_4` Q4 2024) to `26.05%` (`2025-06..08`), while `unseen_route_rate` rises over time (`0.23%` to `1.37%` across dev folds; `659 / 100,000` unseen routes on `2025-09` holdout).
- **Carrier & Route Structure (H2, H3)**: Supported carriers range from `12.73%` (`YX`) to `28.74%` (`F9`), and unseen routes remain ranked above base rate on `2025-09` (`PR-AUC = 0.2494` vs `0.1578` prevalence).
- **Geography & Holidays (H4, H6)**: Exact federal holidays (`0d`, `20.33%`) show slightly lower delay rates than `1-3d` (`21.39%`); high-elevation airports (`ASE` `32.22%`, `n=3,218`) exhibit localized signal (`F3` highest eligible `fold_3` PR-AUC `0.4067`).

## 4. Model Comparison, Calibration & Holdout Protocol

Across 4 expanding temporal folds (`2023-01..2024-12`, feature group `F3`):

| Model | Class Weight | Source | Mean PR-AUC | Worst Fold PR-AUC | Mean ROC-AUC | Mean Brier |
|---|---|---|---:|---:|---:|---:|
| `xgboost` | `none` | `optuna_4_trials` | **0.33910** | 0.26120 | **0.67527** | **0.15166** |
| `xgboost` | `balanced` | `class_weight_pilot` | 0.33752 | 0.25818 | 0.67410 | 0.21894 |
| `lightgbm` | `none` | `optuna_12_trials` | 0.33382 | 0.25846 | 0.67078 | 0.15216 |
| `hist_gradient_boosting` | `none` | `baseline` | 0.33347 | **0.26303** | 0.66970 | 0.15280 |
| `logistic_regression` | `none` | `baseline` | 0.32970 | 0.25643 | 0.66755 | 0.15249 |
| `catboost` | `none` | `optuna_2_trials` | 0.32464 | 0.24347 | 0.66388 | 0.20599 |
| `lightgbm` | `balanced` | `class_weight_pilot` | 0.29318 | 0.21346 | 0.62362 | 0.16227 |
| `dummy_prior` | `none` | `baseline` | 0.20485 | 0.16486 | 0.50000 | 0.16300 |

Calibration comparison (`2025-02..2025-03` slice) showed unweighted native `xgboost` log-loss outputs (`none`: `Brier = 0.14714`, `LogLoss = 0.46340`, `ECE = 0.01680`) outperformed Platt (`Brier = 0.14787`, `ECE = 0.03212`) and Isotonic (`Brier = 0.15021`, `ECE = 0.03772`). Threshold `0.185` was selected solely on `2025-04..2025-05` (`F1 = 0.4074`) and frozen before running `2025-09` once.
