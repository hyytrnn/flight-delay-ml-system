# Flight Delay Prediction Problem Definition

## Decision and target

The system predicts whether a scheduled flight will depart at least 15 minutes
late. The target is BTS `DepDel15`: 1 means a departure delay of at least 15
minutes and 0 means otherwise.

The decision time is fixed at **T-24h**, exactly 24 hours before scheduled
departure. This repository treats BTS schedule fields as a retrospective offline
proxy until a schedule feed with publication timestamps proves that those fields
were available by T-24h.

## Population and label policy

The modeling population contains flights with a valid binary `DepDel15` label that
were neither cancelled nor diverted. Cancelled, diverted and missing-label records
are reported separately and excluded; they are never silently converted to
on-time flights.

## Non-goals for Milestone 1

Milestone 1 does not train a model, report metrics, use arrival delay as a target,
or integrate historical weather. It establishes the ingestion boundary that later
milestones must pass.

