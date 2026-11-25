# V2.1 Retraining & Recalibration Trigger Policy

Threshold definitions are stored declaratively in [monitoring.yaml](file:///D:/CV/flight-delay-ml-system/configs/monitoring.yaml) and evaluated by `build_report()` in [report.py](file:///D:/CV/flight-delay-ml-system/src/flight_delay/monitoring/report.py).

## 1. Configured Alert Thresholds

| Signal | Warning Threshold | Critical / Action Threshold | Interpretation & Action |
|---|---|---|---|
| Covariate `PSI` (`distance`, `crs_elapsed_time`, `origin_elevation`) | `>= 0.15` | `>= 0.25` | Structural schedule/fleet shift; trigger retraining if sustained and accompanied by metric drop |
| `unseen_route_rate` | `>= 0.03` (`3%`) | `>= 0.06` (`6%`) | Network expansion beyond training routes; verify fallback metrics and schedule retraining |
| Calibration `ECE` (with delayed `DepDel15` labels) | `>= 0.04` | Sustained 2+ months `>= 0.04` | Probability shift; evaluate rolling recalibration vs full seasonal refit |
| `brier_delta` (`current_brier - ref_brier`) | `>= +0.025` | Accompanied by PR-AUC drop | Note: distinguish **prevalence-driven Brier inflation** in storm summers (`2025-06..07` had higher Brier despite peak `0.458–0.469` PR-AUC!) from discrimination decay |

## 2. Empirical Policy Lessons from `2025-06..2025-11`

1. **Do NOT blindly retrain on summer `brier_delta` warnings alone**: During `2025-06` and `2025-07`, positive prevalence surged (`0.275–0.285` vs `0.204` ref), raising Brier by `+0.0315..+0.0338`, yet ranking `PR-AUC` hit `0.4578–0.4688` (`ECE = 0.0246–0.0249 < 0.04`).
2. **Respond intentionally to sustained autumn `ECE >= 0.04`**: During `2025-10` (`ECE = 0.0417`) and `2025-11` (`ECE = 0.0442`), both consecutive prospective months crossed `0.04` (`score_psi = 0.293–0.491`, mean score `0.156` vs prevalence `0.197`), triggering `ece_warning` and `retraining_recommended = true`.
3. **Recalibration vs Full Refit Protocol**:
   - Never apply naive 1-month lagged Platt refits blindly across seasonal boundaries (verified in `rolling_recalibration_check.csv`, where 1-month lag worsened `2025-03` ECE `0.0168 -> 0.0321` and `2025-05` `0.0127 -> 0.0298`).
   - Instead, when `ECE >= 0.04` persists across two consecutive months (`2025-10` and `2025-11`), run a scheduled V2.2 training cycle incorporating recent settled BTS partitions (`2025-09..11`), re-select season-matched calibration, and freeze a brand-new prospective holdout (`2025-12..2026-02`) before promoting to `main`.
