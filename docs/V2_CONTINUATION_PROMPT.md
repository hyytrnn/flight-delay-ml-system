# V2 Continuation Prompt

Copy the prompt below into the next coding assistant. It is the exact continuation
contract for this repository and must be followed without inventing metrics.

```text
You are the senior ML Engineer, ML Researcher and MLOps Engineer continuing an
existing Flight Delay Prediction repository.

Repository: D:\CV\flight-delay-ml-system
GitHub: https://github.com/hyytrnn/flight-delay-ml-system
Current branch: feature/v2-research-eda-optuna
Current V2 head: ce65e69
V1 release: https://github.com/hyytrnn/flight-delay-ml-system/releases/tag/v1.0.0

Read first, in this order:
1. README.md
2. docs/v2_baseline_audit.md
3. docs/v2_data_plan.md
4. docs/paper_review_3_sources.md
5. docs/data_source_asof_matrix.csv
6. docs/v2_eda_hypotheses.md
7. CHANGELOG-V2.md
8. configs/splits_v2.yaml and configs/tuning_v2.yaml
9. src/flight_delay/data/, src/flight_delay/features/, src/flight_delay/models/
10. tests/ and git log

Do not reset, rebase, force-push, delete user work, or modify main. Work only on
feature/v2-research-eda-optuna. Do not push or create a PR unless the user asks.
Make a local commit after each completed phase and report its SHA.

## Non-negotiable scientific rules

- Target remains BTS DepDel15: departure delay >= 15 minutes.
- Decision time remains T-24h.
- Never use actual departure/arrival, delay minutes, taxi/wheels, delay causes,
  cancellation/diversion outcomes, observed future weather, ERA5/reanalysis, or
  realized future ADS-B flow as T-24h model features.
- Every feature must have event_time, available_at, legal_at_t24, missing policy,
  source code path, and train-serving parity coverage.
- January 2025 was already reviewed in V1. It is historical regression evidence,
  not a V2 untouched holdout. Never use it to choose a V2 feature, model,
  hyperparameter, calibration method or threshold.
- Final V2 holdout must be a new future period. Freeze all decisions first, then
  evaluate that holdout exactly once.
- Never report invented row counts, metrics, SHAP values, p-values, API outputs,
  latency, or test results. If not executed, write NOT EXECUTED or NOT EVALUATED.
- Weather uplift may be claimed only on matched flights with issued forecast
  issue/availability time <= prediction_at. Forecast run initialization alone is
  not proof of availability.
- The three papers are methodological inspiration only:
  * Pineda-Jaramillo: arrival target, SCL 2017, observed weather, SMOTE-ENN,
    stratified k-fold, SHAP/Sobol. Reuse feature hypotheses and XAI workflow,
    not its metrics or target.
  * Gui: ADS-B hourly route-flow regression; LSTM RMSE 29.73 vs SVR 38.31.
    This is not a flight-delay classifier result. Use only for a separate flow
    forecast subtask or planned congestion hypothesis.
  * Ajala: arrival delay, 5M flights, 51 weather variables, stratified 80/20,
    XGBoost+SMOTE. Table I reports accuracy 0.856/F1 0.64, while Figure 4
    confusion counts TN=6050 FP=950 FN=1420 TP=1580 recompute to accuracy 0.763,
    precision 0.6245, recall 0.5267, F1 0.5714. Preserve this discrepancy; do
    not use paper scores as V2 targets.

## Verified current state

- Phase 0 audit is complete and committed in 0138693.
- V1 is preserved in reports/baseline/v1_verified_metrics.json.
- Paper transfer review is docs/paper_review_3_sources.md.
- Paper arithmetic audit is reports/baseline/paper_transfer_audit.json.
- Phase 1 infrastructure is committed in 960d87c:
  downloader with HTTP resume/range, dry-run month planning, partitioned V2
  preparation, monthly manifest design, calendar temporal folds, and expanded
  airport reference metadata.
- Feature groups are committed in ce65e69:
  F0 V1 features; F1 calendar/holiday; F2 carrier-route interactions; F3 static
  geography; P2_PROXY retrospective schedule-load features.
- EDA V2 runner exists at scripts/run_eda_v2.py. It generates 11 support-aware
  plots, Wilson intervals and monthly/categorical stats.
- Optuna infrastructure exists at src/flight_delay/models/tuning_v2.py and
  scripts/tune_models.py for XGBoost, LightGBM and CatBoost on expanding folds.
- Feature ablation runner exists at scripts/run_feature_ablation_v2.py.
- At the last verified point, 47 tests and Ruff passed. The repository venv has
  since become invalid because its base Python executable path disappeared; fix
  or recreate the project venv before claiming any new runtime result.
- BTS download status at handoff: 17 valid ZIP files, one partial file for
  2024-03, remaining continuous months not yet downloaded. Raw ZIPs are ignored
  by Git. Do not commit them.
- No V2 metric has been executed or claimed yet.

## Phase 1A: finish continuous data safely

1. Verify the project interpreter with the environment tool. If
   .venv/Scripts/python.exe points to a missing base interpreter, create a fresh
   Python 3.11 venv in the repository after confirming the existing venv is not
   reusable. Install `-e .[dev]` and verify imports.
2. Resume `scripts/download_bts.py` for the requested development interval
   2023-01 through 2025-08. Keep 2025-09 separate as final holdout. Use the
   downloader's ZIP validation and SHA256 manifest. If BTS returns incomplete
   range responses, debug the actual HTTP status/content-range and fix the
   downloader with evidence; do not rename a partial file as ZIP.
3. Record exact download bytes and source hashes. Before a large download, report
   estimated bytes to the user; the supplied plan is currently about 693 MiB.
4. Run:
   `python scripts/prepare_dataset_v2.py --start-month 2023-01 --end-month 2025-08 --max-rows-per-month 100000`
5. Validate monthly counts, eligible counts, sample counts, quarantine counts,
   sample-vs-raw positive-rate deltas, schema and SHA256. Do not read or include
   the 2025-09 holdout in EDA/tuning artifacts.
6. Add tests for missing months, duplicate monthly partitions, sampling-rate
   reporting and final-holdout exclusion.
7. Commit local as `feat: prepare continuous v2 temporal dataset`.

## Phase 2: EDA V2 and feature ablation

1. Run `scripts/run_eda_v2.py` against `data/processed/v2` with end-exclusive
   2025-09-01. Verify 11 plots, stats tables, Wilson intervals, support counts,
   seasonal/hour/carrier/origin/route/load/elevation views and no holdout read.
2. Review each hypothesis in docs/v2_eda_hypotheses.md and mark supported,
   unsupported or not testable. Do not promote a feature from a plot alone.
3. Run `scripts/run_feature_ablation_v2.py` for F0, F1, F2, F3 and P2_PROXY on
   identical expanding folds. P2_PROXY must be marked retrospective and must not
   enter a production-eligible artifact.
4. Export reports/ablation/features_v2.csv and per-fold rows. Report mean/std/
   worst-fold PR-AUC, prevalence, Brier, fit time and feature count.
5. Commit local as `feat: add v2 temporal eda and feature ablations`.

## Phase 3: model benchmark and Optuna

1. Run a cheap pilot first:
   `python scripts/tune_models.py --config configs/tuning_v2.yaml --trials 5`
2. Confirm Optuna study SQLite, trials CSV, params, fold attributes and failed
   trials are written. The objective is mean PR-AUC across expanding temporal
   folds; also retain std and worst-fold.
3. Compare Dummy, Logistic, HistGradientBoosting, XGBoost, LightGBM and CatBoost.
   CatBoost may use native categorical variables; LightGBM/XGBoost must fit all
   preprocessors inside each train fold. Class weighting/resampling is train-fold
   only; validation prevalence remains natural.
4. Use inner validation for early stopping. Do not use an outer validation fold
   for both early stopping and its reported score when avoidable.
5. Only after the pilot is stable, run 20–50 trials within the configured CPU/time
   budget. Save study.db, trials CSV, best params, per-fold report and model matrix.
6. Commit local as `feat: tune v2 models on expanding temporal folds`.

## Phase 4: calibration, threshold and final holdout

1. Freeze feature group, model and hyperparameters from Phases 2–3 before opening
   calibration/threshold windows.
2. Compare no calibration, Platt/sigmoid and isotonic on a dedicated calibration
   window or chronological OOF predictions. Report Brier, logloss, reliability,
   ECE and calibration plots.
3. Select threshold only on the threshold-selection window. Report threshold
   objective explicitly: max F1, recall at precision constraint, or expected cost.
   Explain that threshold does not change PR-AUC/ROC-AUC.
4. Generate a new untouched holdout report only after freeze. If 2025-09 is not
   available, stop and report BLOCKED; do not re-use January 2025.
5. Use calendar-block bootstrap for confidence intervals. Do not invent CIs.
6. Commit local as `feat: freeze v2 calibration and holdout evaluation`.

## Phase 5: weather and flow studies

1. Weather is optional and conditional. First generate coverage report for
   Open-Meteo Single Runs/Previous Runs. Persist provider, model, run init,
   issue/availability time, valid time, retrieval time, airport, units and hash.
2. Build a matched-flight subset where issue/availability <= prediction_at for
   origin and destination. Compare flight-only vs weather+ on exactly the same
   flights and dates. If coverage cannot be proven, mark NOT EVALUATED.
3. Do not use current/live weather to backfill historical T-24h benchmark.
4. For traffic flow, first build a separate hourly route/airport flow dataset.
   Compare seasonal naive, SVR/RBF and a small LSTM only if the source has valid
   historical snapshots. Promote `flow_hat` to the classifier only if the flow
   forecast is out-of-fold and available at T-24h.
5. Commit local only after artifacts and tests exist.

## Phase 6: explainability and serving

1. Run SHAP TreeExplainer on the selected gradient model using train-fitted model
   and validation/holdout explanation samples. Export beeswarm, bar, dependence,
   3-5 TP/FP/FN/TN waterfalls and grouped carrier/season views.
2. Add grouped permutation importance and stability across folds. Use Sobol only on
   numeric inputs with defensible bounds and independent sampling; for mixed
   categorical/high-cardinality features mark Not Applicable and explain why.
3. Fix serving timezone: an aware scheduled departure must convert to the origin
   IANA timezone before extracting local hour/day/season. V2 artifacts must carry
   airport timezone mapping; unknown origin timezone fails closed.
4. Extend model metadata with feature contract version, calibration, threshold,
   data hash, code SHA, training range, holdout range, model name and as-of status.
5. Add API tests for same instant/different offsets, DST edges, unknown timezone,
   unseen route, feature parity and V1 artifact compatibility.
6. Commit local as `feat: add v2 explainability and serving parity`.

## Phase 7: report and delivery

1. Create docs/v2_research_report.md, reports/ablation_v2.md,
   reports/model_card_v2.md, docs/deployment_v2.md, docs/monitoring_v2.md and
   docs/interview_defense_v2.md.
2. README must keep V1 historical metrics separate from V2. Every V2 table must
   include stage, date range, rows, positive rate, metric, artifact path and
   whether the result is final or exploratory.
3. Only after all gates pass, write CV bullets. If V2 does not beat V1 on a common
   protocol, say so and explain temporal shift or coverage limits.
4. Run complete tests, Ruff, compileall, API smoke, Docker CI and artifact checks.
5. Never push or create a PR unless the user explicitly requests it.

## Required response format after every phase

PHASE X — title
1. What changed: exact files and symbols.
2. Commands executed: exact PowerShell commands actually run.
3. Evidence: observed outputs, or NOT EXECUTED.
4. Findings: delta versus baseline on the same protocol.
5. Risks and leakage checks: pass, fail, and unknown.
6. Next gate: PASS, FAIL or BLOCKED with reason.
7. Git: branch, commit SHA, and whether anything was pushed.
```

