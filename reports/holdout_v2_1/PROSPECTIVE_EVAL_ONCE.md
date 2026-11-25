# V2.1 Prospective Untouched Holdout Report (Evaluated Once)

- Model: `flight-only-v2.1.0-xgboost` (`F3`)
- Frozen Bundle SHA256: `8b48e8f0b730ac09d11536f4cf7e206962e116a0fcd4f762b02357c59a83d7f6`
- Calibration: `none` (`2025-02-01..2025-03-31`)
- Threshold: `0.1850` (`max_f1` on `2025-04-01..2025-05-31`)
- Prospective Holdout Window: `2025-10-01..2025-11-30` (200,000 rows, prevalence `0.1969`)

## Prospective Holdout Metrics (`2025-10..2025-11`)
- PR-AUC: `0.3265` (uplift vs prevalence: `+0.1296` / `+65.81%`, 95% block-bootstrap CI `[0.2929, 0.3610]`)
- ROC-AUC: `0.6759` (95% CI `[0.6653, 0.6872]`)
- Precision (at max-F1 `0.1850`): `0.3310` | Recall: `0.4490` | F1: `0.3811`
- Precision-targeted operating point (`0.2850`): Precision `0.4260` | Recall `0.1106` | F1 `0.1756`
- Brier: `0.1511` | LogLoss: `0.4730` | ECE: `0.0430`
- Precision@Top5%: `0.4262` | Precision@Top10%: `0.4018`

## Post-Freeze Paired Comparison vs Locked V2 (`flight-only-v2.0.0-xgboost`, `F3`)
- V2 PR-AUC on `2025-10..11`: `0.3265` | ROC-AUC: `0.6759` | Brier: `0.1511`
- Paired delta PR-AUC (`V2.1 - V2`): `+0.00000` (95% block CI `[+0.00000, +0.00000]`, `P(>0)=0.000`)
