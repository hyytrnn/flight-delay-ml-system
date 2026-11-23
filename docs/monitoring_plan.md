# Monitoring Plan

Service monitoring records request counts, HTTP status and latency histograms.
Structured logs include request ID, path, status and latency without tokens or
personal information.

The offline monitoring job compares train and locked-test periods using PSI for
distance and scheduled duration, plus label-rate context. PSI is reported as an
observed statistic without inventing an alert threshold. Data drift is not called
performance degradation; delayed ground-truth labels are required before model
performance can be measured.

Unknown-route fallback rates, invalid timestamps, model-load failures and future
weather provider failures should be monitored if the service is deployed.

