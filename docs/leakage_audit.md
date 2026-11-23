# Leakage Audit

## Blocked fields

The contract blocks actual departure and arrival times, delay minutes, taxi and
wheels times, cancellation/diversion outcomes, arrival-delay labels and all delay
cause fields. The target `dep_del15` is also excluded from candidate features.

## Required invariant

For any flight, changing all outcomes that occur after `prediction_at` must not
change its feature vector. Milestone 1 enforces the structural prerequisite by
maintaining an explicit allowlist and testing that known leakage columns are
rejected.

Historical weather is not implemented. A later weather join must prove
`forecast_issue_time_utc <= prediction_at_utc`; ERA5 or other reanalysis data
cannot satisfy this contract.

