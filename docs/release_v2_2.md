# V2.2 Release Notes — Reliability, Calibration Audit & Public Reviewer Demo

## Scope & Integrity Guarantee

- **Unmodified Historical Baselines**: V1 (`flight-only-v1.0.0`), V2 (`flight-only-v2.0.0-xgboost`), and V2.1 (`flight-only-v2.1.0-xgboost`, SHA256 `8b48e8f0b730ac09d11536f4cf7e206962e116a0fcd4f762b02357c59a83d7f6`) model bundles and holdout metrics are **completely unchanged**.
- **No Holdout Tuning**: Prospective `2025-10..2025-11` (`PR-AUC = 0.3265`, `ECE = 0.0430`) was not reused to tune features, hyperparameters, calibrators, or thresholds.
- **Default vs Experimental Boundary**: Default serving remains **flight-only `F3`** (`flight-only-v2.1.0-xgboost`). Weather (`F5`) and embedding research tracks remain strictly experimental.

## Engineering & Reliability Upgrades (No Metric Inflation)

1. **Documentation & Link Hygiene (`P0`)**:
   - Replaced `"strictly leak-free"` phrasing in `README.md` and `reports/model_card_v2_1.md` with accurate `"T-24h delay risk estimation with leakage safeguards and documented retrospective schedule-data limitations"`.
   - Replaced all local `file:///D:/...` links across `docs/` and `reports/` with relative GitHub links so every documentation link works directly on GitHub.
   - Added [`docs/reviewer_quickstart.md`](reviewer_quickstart.md) for zero-credential fresh-clone verification.
2. **Calibration Drift & Subgroup Audit (`P1`)**:
   - Added `scripts/audit_calibration_v2_2.py`, [`reports/calibration_drift_v2_2.md`](../reports/calibration_drift_v2_2.md), and `reports/figures/calibration_v2_2_reliability.png`.
   - Evaluated `none_frozen`, `platt`, `isotonic`, and `temperature_scaling` (`T = 1.0072`) fit on `2025-02..03` and tested forward on `2025-04..08` (`500,000` rows) with Murphy Brier decomposition, uniform (`10/15/20`) + quantile (`10`) ECE, 95% block-bootstrap CIs, and carrier/route subgroup calibration.
   - Strengthened [`docs/retraining_trigger_policy.md`](retraining_trigger_policy.md) with minimum sample thresholds, BTS ~60-day label-lag rules, instant bundle rollback, and human release approval gates.
3. **Public Reviewer Demo & API/CI Coverage (`P1`)**:
   - Upgraded [`demo/index.html`](../demo/index.html) with 1-click synthetic presets from [`data/examples/sample_flights.json`](../data/examples/sample_flights.json), portfolio disclaimer, summary cards, and `422` validation demonstrations.
   - Added Spring-forward DST (`2027-03-14`) parity + `sample_flights.json` integration tests in `tests/test_api.py` and expanded CI Docker container smoke to verify both `/health` and `/predict`.
