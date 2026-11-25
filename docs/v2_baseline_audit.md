# V2 Baseline Audit

## Audit decision

V1 is a reproducible portfolio baseline, but January 2025 has already informed
model review and cannot serve as the untouched V2 holdout. V2 must preserve the V1
artifact and establish continuous monthly data, rolling temporal validation, a
new future holdout, and a stricter train-serving time contract before any new
performance claim.

## Verified repository state

- Audit base commit: `079e27e4a729e7828825b36040f7200b82935a59`.
- V2 branch: `feature/v2-research-eda-optuna`.
- Baseline tests executed before edits: 28 passed.
- Baseline Ruff check executed before edits: all checks passed.
- Frozen metrics: `reports/baseline/v1_verified_metrics.json`.
- Environment provenance: `reports/baseline/environment.json`.

## Code findings

| Area | Evidence | Finding | Gate |
|---|---|---|---|
| Target | `src/flight_delay/data/clean.py:166-170` | Uses BTS `dep_del15`; cancelled, diverted, and invalid labels are excluded. | PASS |
| Leakage blacklist | `src/flight_delay/features/contracts.py` and mutation tests | Post-outcome fields are blocked and changing future outcomes does not alter V1 features. | PASS |
| Preprocessing | `src/flight_delay/models/train.py:289-305` | Candidate pipelines fit on train only and transform validation/calibration separately. | PASS |
| Model selection | `src/flight_delay/models/train.py:292-301` | Validation PR-AUC is computed from probabilities and selects the model family. | PASS |
| Calibration | `src/flight_delay/models/train.py:303-311` | Platt calibration and F1 threshold selection share October 2024. Valid for V1, but V2 should separate or cross-fit them. | WARN |
| Historical test | `src/flight_delay/models/train.py:313-340` | January 2025 metrics are real but have already been reviewed. | FAIL FOR V2 HOLDOUT |
| Temporal coverage | `scripts/prepare_dataset.py:50-85` | Only eight isolated months are concatenated eagerly. Continuous seasonality and rolling folds are unavailable. | FAIL FOR V2 |
| Sampling | `scripts/prepare_dataset.py:31-34` | Stable per-file random sampling preserves the monthly natural label distribution, but representativeness is not quantified. | WARN |
| Split | `src/flight_delay/models/train.py:77-94` | Calendar order is protected, but boundaries are hard-coded and the test includes every available row from 2025 onward. | WARN |
| Time parsing | `src/flight_delay/data/clean.py:47-120` | HHMM 2400, DST ambiguity, nonexistent times, UTC conversion, and origin timezones are handled during cleaning. | PASS |
| Serving timezone | `src/flight_delay/models/predictor.py:44-50` | `replace(tzinfo=None)` keeps the request wall time instead of converting the instant to the origin IANA timezone. | FAIL |
| Route fallback | `src/flight_delay/models/predictor.py:40-50` | Unknown routes use global distance/duration medians; route status is exposed. | PASS WITH LIMITATION |
| Downloader | `scripts/download_bts.py` | ZIP validation, resume, SHA256, and official URLs exist; range and dry-run interfaces are missing. | WARN |
| Documentation | `docs/experiment_log.md` vs artifact metadata | The experiment log incorrectly listed January and July 2025; actual test was January only. | FIXED IN V2 |

## Metric interpretation

Validation PR-AUC 0.3974 and January 2025 PR-AUC 0.2213 use different class
prevalence and time periods. The decrease is evidence of temporal instability, but
the raw difference must not be described as a controlled model degradation without
paired data and uncertainty analysis. The V2 primary metric remains mean temporal
fold PR-AUC, accompanied by fold prevalence, standard deviation, worst-fold score,
Brier score, and runtime.

## Phase 0 gate

Phase 0 passes for baseline preservation and source audit. It remains blocked for
V2 final evaluation until a new untouched future holdout exists. Phase 1 may begin
with continuous-month acquisition and versioned rolling splits. No V2 metric or
CV bullet is authorized yet.

