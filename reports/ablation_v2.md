# V2 Feature, Model, Weather, and Flow Ablations

## 1. Feature Group Ablation (`2023-01..2024-12`, 4 Rolling Folds)

All runs use `xgboost` on identical chronological folds ([`reports/ablation/features_v2.csv`](ablation/features_v2.csv)):

| Feature Group | Eligibility | Cols | Mean PR-AUC | Worst Fold PR-AUC | Fold 3 PR-AUC | Mean Brier |
|---|---|---:|---:|---:|---:|---:|
| `F0` (Base schedule & route) | T-24h eligible | 12 | 0.33645 | 0.25408 | — | 0.15175 |
| `F1` (+ Cyclical & U.S. holidays) | T-24h eligible | 17 | 0.33329 | 0.26065 | — | 0.15265 |
| `F2` (+ Carrier/origin/route interactions) | T-24h eligible | 21 | 0.33548 | **0.26257** | 0.4059 | 0.15233 |
| `F3` (+ Airport geography & elevation) | T-24h eligible | 31 | 0.33510 | 0.26089 | **0.40674** | 0.15244 |
| `P2_PROXY` (`F3` + full-day BTS bank load) | Retrospective proxy only | 36 | **0.33889** | **0.26549** | — | 0.15208 |

Key takeaway: `P2_PROXY` gains `+0.00379` PR-AUC over `F3` (`+0.00244` over `F0`), confirming airport congestion is informative, but full-day BTS planned counts cannot be treated as guaranteed T-24h operational inputs without an OAG/FAA schedule feed (`docs/feature_eligibility_v2.csv`). `F3` was frozen as the rich T-24h flight-only feature set.

## 2. Optuna vs Untuned Baselines & Class-Weighting (`P5`)

Source: [`reports/metrics/v2_model_comparison.csv`](metrics/v2_model_comparison.csv).

- Replacing sparse One-Hot Encoding with `CategoricalFrameTransformer` (`min_frequency=100, max_categories=300`) + native XGBoost `enable_categorical=True, max_cat_to_onehot=16` reduced per-fold training latency from >14 minutes to `~16s` (`fold_1`) and raised Optuna `xgboost` CV to `0.33910` (vs untuned `hist_gradient_boosting` `0.33347`, `logistic_regression` `0.32970`, `lightgbm` `0.33382`, `catboost` `0.32464`).
- `scale_pos_weight = n_neg / n_pos` degraded both ranking and calibration (`xgboost`: PR-AUC `0.33752` vs `0.33910`, Brier `0.21894` vs `0.15166`; `lightgbm`: PR-AUC `0.29318` vs `0.33382`, Brier `0.16227` vs `0.15216`).

## 3. Matched-Coverage Issued Weather (`P4`)

Source: [`reports/ablation/weather_comparison.csv`](ablation/weather_comparison.csv), [`docs/weather_coverage_report.md`](../docs/weather_coverage_report.md).

- Provider: Open-Meteo `previous-runs` (`ecmwf_ifs025`, `_previous_day2` = 48h lead minus 6h assumed availability lag -> valid `<= T-42h <= T-24h`).
- Archive boundary: `ecmwf_ifs025` previous-runs archive begins `2024-04-01` (`2023-01..2024-03` unavailable). Matched evaluation covers `492,847` flights across top-25 U.S. airports (`2024-04-01..2025-08-31`) where both origin and destination forecasts exist:

| Variant | Window 1 (`2025-02..05`) PR-AUC | Window 2 (`2025-05..08`) PR-AUC | Mean PR-AUC | Mean ROC-AUC | Mean Brier |
|---|---:|---:|---:|---:|---:|
| `F3_flight_only_matched` | 0.28426 | 0.44133 | 0.36280 | 0.67103 | 0.16560 |
| `F5_flight_plus_weather_matched` | **0.30062** | **0.45258** | **0.37660** | **0.68009** | **0.16429** |

Issued weather produces a consistent `+0.01380` PR-AUC and `+0.00906` ROC-AUC gain on covered airport pairs, motivating `F5` as a documented opt-in weather track once live NWP ingestion is wired for all stations.

## 4. T-24h Air Traffic Flow Regression & Classifier Ablation (`P3`)

Sources: [`reports/ablation/flow_regression_v2.csv`](ablation/flow_regression_v2.csv), [`reports/ablation/flow_classifier_ablation_v2.csv`](ablation/flow_classifier_ablation_v2.csv).

- **Hourly Flow Regression** (4 rolling folds `2024-01..2024-12`):
  - `hist_gbr`: mean `RMSE = 1.7024`, `MAE = 1.1942` (`-20.2%` RMSE vs naive)
  - `svr_rbf_2500_sample`: mean `RMSE = 1.7726`, `MAE = 1.1864`
  - `seasonal_naive_7d`: mean `RMSE = 2.1322`, `MAE = 1.3941`
- **Downstream Classification (`F3` vs `F3_plus_asof_flow_lags`)**:
  - `F3_baseline`: `mean_pr_auc = 0.33612`, `mean_brier = 0.15205`
  - `F3_plus_asof_flow_lags` (`flow_lag_24h`, `flow_lag_7d`, `flow_asof_mean_hist`): `mean_pr_auc = 0.33623` (`+0.00011`), `mean_brier = 0.15208`
  - Strictly lagged (`<= t-24h`) sampled BTS counts add negligible incremental signal over `F3`.

## 5. Paper-Inspired Matrix

See [`reports/ablation/paper_inspired_v2.csv`](ablation/paper_inspired_v2.csv) (`P0`–`P9`).
