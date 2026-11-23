# Data Availability Policy

Every candidate feature must have an event time and an `available_at` time. A
feature is eligible only when `available_at <= prediction_at`, where
`prediction_at = scheduled_departure_utc - 24 hours`.

| Feature group | Availability at T-24h | Milestone 1 policy |
|---|---:|---|
| Origin, destination, carrier, scheduled departure | Conditional | Retrospective BTS proxy; document limitation |
| Distance and scheduled elapsed time | Conditional | Eligible only as schedule or static reference data |
| Actual departure/arrival, taxi, wheels times | No | Outcome-only; blocked from model features |
| DepDelay and DepDelayMinutes | No | Label-related outcome; blocked |
| Arrival delays and delay causes | No | Outcome-only; blocked |
| Cancelled and diverted | No | Population policy only; never a T-24h feature |
| Weather forecast | Conditional | Future milestone; issue time must be <= prediction_at |
| ERA5 or reanalysis weather | No | EDA/oracle analysis only, never production-equivalent T-24h weather |

The first real data run must preserve source URL, retrieval time, bytes and SHA-256
in `data/download_manifest.csv`. No real BTS file has been supplied, so real data
availability remains **NOT EXECUTED**.

