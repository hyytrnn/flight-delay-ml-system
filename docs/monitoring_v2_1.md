# V2.1 Monitoring Architecture & Protocol

Upgraded monitoring pipeline implemented in [report.py](../src/flight_delay/monitoring/report.py), [monitor.py](../scripts/monitor.py), and configured by [monitoring.yaml](../configs/monitoring.yaml).

## 1. Monitored Signals

Each run compares a target window against the training reference (`data/processed/v2`, `< 2025-02-01`, `2,500,000` rows, positive rate `0.2041`, reference Brier `0.14803`):

1. **Covariate Drift (`feature_psi`)**: Population Stability Index across `distance`, `crs_elapsed_time`, and `origin_elevation` (10 quantile bins).
2. **Route Novelty (`unseen_route_rate`)**: Share of `(origin|dest)` pairs absent from the reference corpus.
3. **Score Drift (`score_psi`, `score_histogram`, `predicted_score_summary`)**: 10-bin score histogram (`0.0..1.0`), `score_psi`, deciles (`p50`, `p90`, `p95`), and expected alert rates at both `0.185` (`max_f1`) and `0.285` (`precision_target_35`).
4. **Delayed-Label Outcome Metrics (`outcome_metrics`, `brier_delta`)**: Computed automatically when actual `DepDel15` outcomes settle (`PR-AUC`, uplift, `ROC-AUC`, `Precision`, `Recall`, `F1`, `Brier`, `LogLoss`, `ECE`, `Precision@Top5%`, `Precision@Top10%`).
5. **Automated Alert Evaluation (`alerts`)**: Evaluates warning and critical rules and populates `retraining_recommended` and `alert_reasons`.

## 2. Running Locally

```powershell
# Evaluate prospective 2025-10..2025-11 window against V2.1 frozen bundle
.\.venv\Scripts\python.exe scripts/monitor.py `
  --data-root data/processed/v2 `
  --current-data-root data/processed/v2_1_future_holdout `
  --reference-end 2025-02-01 `
  --current-start 2025-10-01 `
  --current-end 2025-12-01 `
  --bundle models/flight_delay_v2_1_bundle.joblib `
  --config configs/monitoring.yaml `
  --window-label 2025-10_to_2025-11 `
  --output reports/monitoring/v2_1_latest.json
```

Generated historical & prospective snapshots live in [reports/monitoring/history/](../reports/monitoring/history/) (`2025-06.json`..`2025-11.json`) and [v2_1_latest.json](../reports/monitoring/v2_1_latest.json); detailed findings are summarized in [drift_report.md](drift_report.md).
