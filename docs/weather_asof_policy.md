# Weather As Of Policy

Weather is not part of the offline benchmark. Open-Meteo documentation currently
states that most Previous Runs models start in January 2024, ECMWF IFS HRES single
runs start in March 2024, and most other single-run archives start in April 2026.
That does not provide one uniform issued-run source across the selected 2023-2025
benchmark.

The repository therefore implements a resilient live forecast client for a future
online model but returns `weather_status=not_used_flight_only_model` from the v1
API. Historical Forecast stitched data and ERA5 reanalysis are not substituted for
a forecast that demonstrably existed by T-24h.

Any later Weather+ experiment must store provider, model/run, issue time, valid
time, retrieval time, airport code and lead hours, and must enforce
`forecast_issue_time_utc <= prediction_at_utc` in tests.

