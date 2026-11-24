# Weather Coverage & As-Of Provenance Report (V2)

- Provider: `open-meteo-previous-runs` (`ecmwf_ifs025`, lead `2` days, assumed availability lag `6`h -> available at `T-42h <= T-24h`)
- Archive gap: 2023-01..2024-03 is not covered by `ecmwf_ifs025` previous-runs archive; weather study is restricted to `2024-04-01..2025-08-31`.
- Top airports fetched: 25 airports
- Flights in window `2024-04-01..2025-08-31`: 1,700,000
- Origin weather available: 1,003,669 (59.04%)
- Destination weather available: 1,004,181 (59.07%)
- Matched-coverage (both origin and dest available): 492,847 (28.99%)
