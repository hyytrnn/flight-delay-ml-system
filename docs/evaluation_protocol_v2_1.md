# V2.1 Evaluation & Leakage Protocol

## 1. Purpose & Non-Stationary Window Governance

V2.1 enforces strict separation between development/selection windows, historical viewed sets, and the new prospective holdout (`configs/splits_v2_1.yaml`):

| Window | Date Range (`[start, end)`) | Role in V2.1 | Constraints |
|---|---|---|---|
| Rolling Folds (`fold_1..4`) | `2023-01-01 .. 2025-01-01` (4 expanding quarters) | Feature ablation, model selection, Optuna tuning, paired block-bootstrap CIs | Same 4 folds & rows (`1,200,000` total val rows) across all candidates; preprocessors fit inside each outer train / inner train split |
| V1 Overlap Reference | `2025-01-01 .. 2025-02-01` | Historical comparison check vs V1 test period | Non-paired with V1 random sample (`V1` used 8 isolated months, `2025-01` only in test) |
| Calibration Selection | `2025-02-01 .. 2025-04-01` | Compare `none`, `platt`, `isotonic` (Feb fit -> Mar eval, refit on Feb–Mar) | Never touches labels outside `[2025-02, 2025-04)` |
| Threshold Policy Selection | `2025-04-01 .. 2025-06-01` | Select fixed `max_f1`, `precision_target`, and top-K alert budgets | Never tuned on `2025-06+` |
| Pre-Holdout Validation | `2025-06-01 .. 2025-09-01` | Monthly slice error audit, cost curves, calibration checks prior to freeze | Zero model/threshold re-tuning after viewing |
| V2 Historical Holdout | `2025-09-01 .. 2025-10-01` | **Historical reproducibility check ONLY** (`HISTORICAL_VIEWED_IN_V2`) | Opened in V2 (`PR-AUC = 0.2950`, `prevalence = 0.16545`, `+78.31%` uplift over Sept base rate); forbidden for V2.1 selection |
| **V2.1 Prospective Holdout** | `2025-10-01 .. 2025-12-01` (`2025-10`, `2025-11`) | **Final untouched evaluation (executed ONCE in Gate G8)** | Excluded from BTS preparation until feature group, model, params, calibration, and threshold are frozen |

## 2. Paired Comparison & Uncertainty Standard

1. Model/feature superiority is never claimed from single-point averages alone.
2. Every comparison reports: `mean_pr_auc`, `std_pr_auc`, `worst_fold_pr_auc`, `mean_roc_auc`, `mean_brier`, prevalence baseline, `compute_seconds`, **95% calendar-block bootstrap CI** (`n_boot >= 300`, resampling whole `flight_date` blocks) on the paired difference `candidate - baseline` over identical validation rows, and SHA256 hashes of OOF predictions.
3. Parsimony rule: if a larger feature group (`F2` or `F3`) fails to gain at least `+0.0005` `mean_pr_auc` over `F0` on equal Optuna budget (or degrades `worst_fold_pr_auc` by `> 0.0020` / `mean_brier` by `> 0.0005`), retain the simpler group.

## 3. Baseline Disambiguation (`V1` vs `V2` vs `V2.1`)

- **V1**: 8 isolated monthly BTS partitions, `flight-only-v1.0.0`, tested on Jan 2025 (`PR-AUC = 0.2213`, prevalence `0.1825`). Preserved frozen in `models/flight_delay_bundle.joblib` and `reports/baseline/v1_verified_metrics.json`.
- **V2**: 32 continuous dev months (`2023-01..2025-08`), `flight-only-v2.0.0-xgboost`, evaluated once on `2025-09` (`PR-AUC = 0.2950`, prevalence `0.16545`). `+78.31%` denotes relative uplift **over September 2025 base prevalence** (`0.2950 / 0.16545 - 1`), NOT gain over V1 (`2025-01` vs `2025-09` are unpaired seasonal months). Locked in `models/locked_v2_manifest.json`.
- **V2.1**: Equal-budget fair ablation, unified `refit_full_train` early stopping parity, time-budgeted multi-model Optuna comparisons (`18 XGBoost [16 complete/2 pruned in 2000s], 16 LightGBM [11 complete/5 pruned in 1200s], 3 CatBoost [in 900s]`), error/calibration/weather audits, and single prospective evaluation on `2025-10..2025-11` AFTER freezing `models/flight_delay_v2_1_bundle.joblib`.
