# BTS Canonical Data Dictionary

No real BTS file was available in the workspace, so source columns have not yet
been empirically inspected. The table below defines the expected canonical
contract and the aliases exercised by synthetic tests. The first real run must
record unexpected, missing and newly mapped columns before this contract is
considered verified against a specific BTS extract.

| Canonical field | Typical BTS source | Type after cleaning | Purpose |
|---|---|---|---|
| flight_date | FlightDate | Date | Scheduled service date |
| reporting_airline | Reporting_Airline or UniqueCarrier | String | Schedule category |
| flight_number | Flight_Number_Reporting_Airline | String | Composite flight identity |
| origin | Origin | String | IATA origin code |
| dest | Dest | String | IATA destination code |
| crs_dep_time | CRSDepTime | Int | Scheduled HHMM; 2400 means next-day midnight |
| crs_arr_time | CRSArrTime | Int | Retained for audit; not parsed in Milestone 1 |
| crs_elapsed_time | CRSElapsedTime | Float | Planned duration candidate |
| distance | Distance | Float | Route distance candidate |
| dep_del15 | DepDel15 | Int 0/1 | Target only |
| cancelled | Cancelled | Int 0/1 | Population exclusion policy |
| diverted | Diverted | Int 0/1 | Population exclusion policy |
| scheduled_departure_local | Derived | Datetime without zone | Local wall-clock schedule |
| origin_timezone | Airport reference | IANA timezone | Provenance for localization |
| dest_timezone | Airport reference | IANA timezone | Destination validation and future arrival weather |
| scheduled_departure_utc | Derived | UTC Datetime | Time split and availability checks |
| prediction_at_utc | Derived | UTC Datetime | T-24h decision time |
| ambiguous_time | Derived | Boolean | DST overlap resolved to earliest fold |
| flight_key | Derived | String | Duplicate audit key |

Outcome fields may remain in an audit dataset, but the feature contract prevents
them from entering model inputs.


