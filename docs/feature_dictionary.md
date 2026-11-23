# Model Feature Dictionary

| Feature | Type | Source | T-24h policy |
|---|---|---|---|
| reporting_airline | categorical | BTS schedule proxy | Conditional; publication timestamp is unavailable in BTS |
| origin | categorical | BTS schedule proxy | Conditional |
| dest | categorical | BTS schedule proxy | Conditional |
| distance | numeric | BTS schedule/route field | Conditional; verified positive during cleaning |
| crs_elapsed_time | numeric | BTS scheduled duration | Conditional |
| dep_hour_sin/cos | numeric | Scheduled local departure | Eligible; never derived from actual departure |
| weekday_sin/cos | numeric | Scheduled service date | Eligible |
| month_sin/cos | numeric | Scheduled service date | Eligible |
| is_weekend | numeric | Scheduled service date | Eligible |

Actual flight times, delay minutes, taxi and wheels fields, arrival delay,
cancellation/diversion outcomes and delay causes are explicitly blocked. The same
`build_feature_frame` function is used by training and API inference.

