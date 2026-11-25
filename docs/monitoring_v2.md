# V2 Monitoring & Drift Protocol

## Design Principle

Feature drift (`PSI`, unseen route share) and predicted-score shift provide early operational signals, **not** proof of model performance degradation. Real `DepDel15` outcomes arrive with BTS monthly reporting lag (~60 days), so outcome metrics (`PR-AUC`, `ROC-AUC`, `Brier`, `ECE`, sliced by airline and route familiarity) must be computed once delayed labels arrive.

## Baseline Check (`reports/monitoring/v2_weekly.json`)

Executed with:

```powershell
.\.venv\Scripts\python.exe scripts/monitor.py `
  --data-root data/processed/v2 `
  --reference-end 2025-02-01 `
  --current-start 2025-06-01 `
  --current-end 2025-09-01 `
  --output reports/monitoring/v2_weekly.json
```

Observed results (`2,500,000` reference rows vs `300,000` summer `2025-06..08` rows):
- `crs_elapsed_time` PSI: `0.00148` (negligible schedule shift)
- `distance` PSI: `0.00194` (negligible network distance shift)
- `origin_elevation` PSI: `0.00238`
- `unseen_route_rate`: `0.00579` (`0.58%`)
- `reference_positive_rate`: `0.2041` vs `current_positive_rate`: `0.2605` (seasonal summer delay surge, not model failure — pre-holdout PR-AUC reached `0.4419` with `ECE = 0.0207`).

## Alerting & Recalibration Policy

1. **Covariate / Route Alerts**: Investigate if numeric `PSI >= 0.15` or `unseen_route_rate >= 0.03` or `422` unknown-origin rate spikes.
2. **Calibration Drift**: Re-check `ECE` and `Brier` quarterly once realized BTS labels close; keep decision threshold (`0.185`) separate from probability calibration (`none`, Platt, or Isotonic).
3. **Top-K Capacity Alerts**: Monitor `Precision@Top5%` and `Precision@Top10%` alongside fixed-threshold F1 when base prevalence fluctuates seasonally (`16.5%` autumn to `26.0%` summer).
