# V2 Interview & Design Defense Guide

## Concise Narrative

V1 proved end-to-end BTS ingestion, fast serving, and CI testing on a 6-month sampled split (Jan 2025 test `PR-AUC = 0.2213`). V2 rebuilds the entire scientific lifecycle across **33 continuous monthly BTS partitions** (`2023-01..2025-08` dev = `3.2M` rows, `2025-09` = `100k` untouched final holdout) under a strict **T-24h operational contract**:

1. **Why PR-AUC + Calibration instead of Accuracy / SMOTE / Class Weights?**
   `DepDel15` has a 16.5%–26.0% positive rate that swings with season. Accuracy rewards predicting majority on-time flights, whereas `PR-AUC` + prevalence uplift (`+78.3%` on `2025-09` holdout: `0.2950` vs `0.1655` base rate) measures ranking precision under realistic alert budgets (`Precision@Top5% = 0.3952`). Empirically testing `scale_pos_weight=balanced` (`P5`) harmed both ranking (`0.3375` vs `0.3391` XGBoost) and probability calibration (`Brier 0.2189` vs `0.1517`), confirming unweighted training + separated threshold tuning (`0.185`) is strictly superior.
2. **Why Native Categorical XGBoost (`CategoricalFrameTransformer`) over Sparse OHE?**
   On `F3` (~1,800 sparse OHE columns across `carrier_origin`, `carrier_route`, `origin_hour`, `route_season`), XGBoost took >14 minutes per trial. Mapping bounded categories (`min_frequency=100, max_categories=300`) into pandas `category` with XGBoost `enable_categorical=True, max_cat_to_onehot=16` cut `fold_1` training from >14 min to `16.0s` while achieving `0.33910` 4-fold CV PR-AUC.
3. **How Was Temporal Leakage Prevented?**
   - Strict `T-24h` schema contract drops post-outcome columns (`DepDelay`, `TaxiOut`, `WheelsOff`, `ArrDelay`, delay causes) and marks `P2_PROXY` full-day bank counts as offline proxy only.
   - Calibration (`2025-02..03`), threshold (`2025-04..05`), and pre-holdout (`2025-06..08`) sit in disjoint future months, and `2025-09` remained unprepared (`--allow-holdout`) until every modeling choice was frozen.
   - Open-Meteo `ecmwf_ifs025` weather used `_previous_day2` (48h lead minus 6h availability lag `<= T-42h`) and was strictly tested on matched archive coverage (`2024-04..2025-08`, `+0.0138` PR-AUC gain) instead of extrapolating missing `2023` weather.
4. **How Did You Handle the Three Academic Papers Rigorously?**
   We audited every claim rather than copying numbers across benchmarks: documented an arithmetic discrepancy in Ajala Fig. 4 (`0.73` reported vs `0.6866` confusion-matrix calculation), showed Gui-style flow regression excels at predicting hourly counts (`RMSE 1.70` vs `2.13` naive) while `<= t-24h` BTS sample lags add only `+0.00011` PR-AUC to `F3`, and noted why Sobol variance decomposition is invalid on correlated flight categoricals/trig pairs (`reports/explainability/sobol_if_valid.json`).
