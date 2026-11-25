# V2.1 Validation Protocol

Canonical specification: [`docs/evaluation_protocol_v2_1.md`](evaluation_protocol_v2_1.md) and [`configs/splits_v2_1.yaml`](../configs/splits_v2_1.yaml).

- **Target**: BTS `DepDel15` (`>= 15` min departure delay, excluding cancelled/diverted flights), predicted strictly at `T-24h`.
- **CV Folds (`2023-01..2024-12`)**: 4 chronological expanding folds (`fold_1..fold_4`), identical rows & seeds across feature groups and model families; paired `flight_date` block bootstrap (`n_boot >= 300`) for difference CIs.
- **Calibration / Threshold / Pre-Holdout**: Disjoint future windows `2025-02..03` (calibration), `2025-04..05` (threshold policies), `2025-06..08` (pre-holdout audit).
- **Historical Viewed Checks**: `2025-01` (V1 overlap), `2025-09` (V2 final holdout; `REPRODUCIBILITY_CHECK_ONLY` in V2.1).
- **Untouched Prospective V2.1 Holdout**: `2025-10..2025-11` (`200,000` sampled eligible rows) downloaded, prepared, and evaluated **strictly once** only after `models/flight_delay_v2_1_bundle.joblib` and threshold are frozen.
