# V2 Final Untouched Holdout Report (Evaluated Once)

- Model: `xgboost` (`F3`)
- Training window: `2023-01-01..2025-01-31` (2,500,000 rows)
- Calibration: `none` on `2025-02-01..2025-03-31` (200,000 rows)
- Threshold: `0.1850` (`max_f1` on `2025-04-01..2025-05-31`, 200,000 rows)
- Final Holdout: `2025-09-01..2025-09-30` (100,000 rows, prevalence=0.1655)

## Holdout Metrics
- PR-AUC: `0.2950` (uplift vs prevalence: `+0.1296`, 95% block-bootstrap CI `[0.2661, 0.3169]`)
- ROC-AUC: `0.6868` (95% CI `[0.6726, 0.6993]`)
- Precision: `0.2797` | Recall: `0.5280` | F1: `0.3657`
- Brier: `0.1298` | LogLoss: `0.4197` | ECE: `0.0139`
- Precision@Top5%: `0.3952` | Precision@Top10%: `0.3587`
