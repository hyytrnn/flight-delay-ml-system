# V2 Interview Explanations

## Why the paper scores are not project targets

The papers use different countries, years, labels, feature availability, sampling,
and split protocols. Reusing their scores would misrepresent the project. V2 uses
the papers to define falsifiable experiments, then reports only artifacts generated
under the repository's BTS `DepDel15` T-24h protocol.

## Why January 2025 is no longer the final test

The V1 January 2025 score has already influenced review and V2 planning. It remains
a historical regression benchmark, but a new future period must be frozen before
V2 model and feature decisions are finalized.

## Why weather is difficult

Weather measured at takeoff or landing is informative but unavailable 24 hours
before departure. A valid production feature needs a forecast run that had already
been issued and available at prediction time, aligned to the scheduled origin and
destination times. V2 reports weather uplift only on matched flights with verified
issued-run coverage.

## Why LSTM is not automatically the next model

The ADS-B paper predicts hourly route flow, not individual flight delay. Its LSTM
result motivates an auxiliary flow-forecast experiment. V2 promotes that forecast
to a classifier feature only if out-of-fold flow predictions improve temporal
PR-AUC beyond a simpler seasonal or gradient-boosting baseline.

## Why accuracy is not the primary metric

Delay prevalence changes over time and across airports. Accuracy can improve by
predicting the majority class. V2 selects models by temporal-fold PR-AUC and also
reports prevalence, worst-fold PR-AUC, calibration, F1, precision, recall, and
alert burden.

