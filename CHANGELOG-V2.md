# Flight Delay Prediction V2 Changelog

## Phase 0 Baseline Audit

- Created the research branch `feature/v2-research-eda-optuna` from V1 commit
  `079e27e4a729e7828825b36040f7200b82935a59`.
- Froze V1 metrics as historical evidence. January 2025 has already been reviewed
  and is no longer eligible as an untouched V2 holdout.
- Corrected the experiment-log date mismatch without changing historical metrics.
- Added a three-paper transfer review with explicit target, timing, and leakage
  constraints.
- Added a paper-claim audit for the internally inconsistent confusion-matrix and
  summary metrics in the real-time meteorological paper.
- Recorded the execution environment and baseline test result.

No V2 performance claim was made in Phase 0; Phases 1–7 executed all V2 gates sequentially below.

## Phases 1–2 Continuous Data, Temporal EDA & Feature Ablation

- Downloaded and SHA256-verified 33 monthly BTS ZIP archives (`2023-01` through `2025-09`).
- Prepared 32 continuous development months (`2023-01`..`2025-08`, `3,200,000` sampled rows, positive rate `0.2088`) while excluding `2025-09` behind `--allow-holdout`.
- Generated 14 temporal/non-stationarity EDA figures (`reports/eda_v2/`), PSI drift tables, and 4-fold feature ablations (`F0..F3`, `P2_PROXY`).

## Phase 3 Expanding-Fold Optuna Tuning

- Compared `dummy_prior`, `logistic_regression`, `hist_gradient_boosting`, `lightgbm`, `catboost`, and `xgboost` (`CategoricalFrameTransformer` + `enable_categorical=True`) across 4 chronological folds (`F3`).
- `xgboost` (`none` weighting) led at `mean_pr_auc = 0.33910` (`mean_roc_auc = 0.67527`, `mean_brier = 0.15166`).
- Audited `P5` `scale_pos_weight=balanced` pilot: degraded both PR-AUC and Brier across `xgboost` and `lightgbm`.

## Phase 4 Separated Calibration, Threshold Freeze & One-Time Holdout

- Evaluated `none`, `platt`, and `isotonic` on `2025-02..2025-03`; retained native `none` (`Brier = 0.14714`, `ECE = 0.01680`).
- Froze threshold `0.185` on `2025-04..2025-05` (`F1 = 0.4074`).
- Evaluated untouched `2025-09` holdout (`100,000` rows, `0.16545` prevalence) **once**:
  `PR-AUC = 0.2950` (`+0.1296` / `+78.3%` over prevalence, 95% CI `[0.2661, 0.3169]`),
  `ROC-AUC = 0.6868` (`[0.6726, 0.6993]`), `F1 = 0.3657`, `Brier = 0.1298`, `ECE = 0.0139`, `Precision@Top5% = 0.3952`.

## Phases 5–7 Weather, Traffic Flow, Explainability, Parity & Delivery Docs

- Matched-coverage `ecmwf_ifs025` `_previous_day2` issued weather (`2024-04..2025-08`, `492,847` matched flights, top-25 airports) improved `F5` over `F3` (`PR-AUC = 0.3766` vs `0.3628`, `+0.0138`).
- Auxiliary hourly air-traffic flow `hist_gbr` reduced flow RMSE (`1.7024` vs `2.1322` 7-day naive), while strictly `<= t-24h` BTS sampled flow lags added `+0.00011` PR-AUC over `F3`.
- Generated TreeSHAP + grouped permutation rankings (`reports/explainability/`), verified UTC vs `-05:00` API parity (`scripts/api_smoke_v2.py`), and published V2 research/deployment/monitoring/model-card documentation.

