# V2.2 Retraining, Calibration & Rollback Policy

Threshold definitions are stored declaratively in [monitoring.yaml](../configs/monitoring.yaml) and evaluated by `build_report()` in [report.py](../src/flight_delay/monitoring/report.py).

## 1. Configured Alert & Sample-Support Thresholds

| Signal | Minimum Support | Warning / Trigger Rule | Action Required |
|---|---:|---|---|
| Covariate `PSI` (`distance`, `crs_elapsed_time`, `origin_elevation`) | `>= 10,000` flights/window | `>= 0.15` warning; `>= 0.25` critical | Inspect schedule/network mix; trigger retraining if sustained across 2 consecutive months with metric drop |
| `unseen_route_rate` | `>= 10,000` flights/window | `>= 0.03` warning; `>= 0.06` critical | Audit fallback route metrics (`route_status = fallback_global_median`) and schedule route-table refresh |
| Calibration `ECE` (delayed `DepDel15` labels) | `>= 50,000` labeled flights/month (`>= 500` for subgroup audit) | `>= 0.04` warning; **2 consecutive months `>= 0.04`** triggers retraining | Run `scripts/audit_calibration_v2_2.py` on settled pre-holdout labels; schedule full refit if calibration shift persists |
| `brier_delta` (`current_brier - ref_brier`) | `>= 50,000` labeled flights/month | `>= +0.025` warning | Cross-check `brier_reliability` vs `brier_resolution` and `PR-AUC`: do **not** retrain solely on storm-season prevalence surges (`2025-06..07` had higher Brier despite peak `0.458–0.469` PR-AUC) |
| **Missing / Delayed Labels** | BTS ~60-day lag | `dep_del15` absent or `< 50,000` settled rows | Emit explicit label-availability warning; rely on `score_psi` & `feature_psi` only and defer ECE/Brier claims until labels settle |

## 2. Empirical Policy Lessons (`2025-04..2025-11`)

1. **Do NOT force post-hoc recalibration that hurts forward windows**: In [calibration_drift_v2_2.md](../reports/calibration_drift_v2_2.md), fitting Platt (`delta_Brier = +0.000354`), Isotonic (`+0.004205`), or 1-parameter temperature scaling (`T = 1.0072`, `-0.000007`) on `2025-02..03` failed to improve `2025-04..08` over `none_frozen`.
2. **Respond intentionally to sustained autumn `ECE >= 0.04`**: On `2025-10` (`ECE = 0.0417`) and `2025-11` (`ECE = 0.0442`), both consecutive prospective months crossed `0.04`, triggering `ece_warning` and `retraining_recommended = true`.

## 3. Release Gate & Rollback Protocol

- **Approval Gate**: Any candidate retrained bundle must pass (a) chronological out-of-time evaluation on a newly settled holdout window untouched during selection, (b) zero leakage unit tests (`ruff`, `pytest --cov`), and (c) explicit human review of the before/after calibration and subgroup (`reporting_airline`, `route_known`) report before merging to `main`.
- **Instant Rollback**: Set `FLIGHT_DELAY_MODEL_PATH` back to the SHA256-locked prior bundle (`models/flight_delay_v2_1_bundle.joblib`, `models/flight_delay_v2_bundle.joblib`, or `models/flight_delay_bundle.joblib`) and verify `/health` + `/predict` parity without code changes.
