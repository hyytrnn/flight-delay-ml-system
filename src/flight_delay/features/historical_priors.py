"""As-of historical BTS delay priors enforcing conservative publication lag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

HISTORICAL_PRIOR_COLUMNS = [
    "origin_hist_delay_rate_asof",
    "carrier_hist_delay_rate_asof",
    "route_hist_delay_rate_asof",
    "carrier_origin_hist_delay_rate_asof",
    "route_hist_support_asof",
    "route_hist_unseen_asof",
    "hist_prior_freshness_age_days",
]


def _ensure_utc_series(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    return parsed


def compute_label_available_utc(
    flight_dates: pd.Series, *, publication_lag_months: int = 2
) -> pd.Series:
    """Return conservative UTC timestamp when monthly BTS labels become public.

    For monthly release cycles with ``publication_lag_months=2``, all flights in month M
    become available on the first day of month M + 3 (UTC midnight), leaving at least
    two full calendar months after month-end before use at prediction_at_utc.
    """

    if publication_lag_months < 0:
        raise ValueError("publication_lag_months must be non-negative")
    periods = pd.to_datetime(flight_dates).dt.to_period("M") + (publication_lag_months + 1)
    avail = periods.dt.to_timestamp(how="start")
    return pd.to_datetime(avail, utc=True)


def compute_source_month_end_utc(flight_dates: pd.Series) -> pd.Series:
    periods = pd.to_datetime(flight_dates).dt.to_period("M")
    ends = periods.dt.to_timestamp(how="end").dt.floor("s")
    return pd.to_datetime(ends, utc=True)


@dataclass(frozen=True)
class HistoricalPriorConfig:
    alpha: float = 100.0
    min_support: int = 30
    publication_lag_months: int = 2
    cold_start_prior: float = 0.20


def _aggregate_counts(
    frame: pd.DataFrame, key_col: str
) -> tuple[dict[str, float], dict[str, int]]:
    grouped = frame.groupby(key_col, sort=False)["_label"].agg(["sum", "count"])
    pos = {str(k): float(v) for k, v in grouped["sum"].items()}
    cnt = {str(k): int(v) for k, v in grouped["count"].items()}
    return pos, cnt


def _merge_counts(
    acc_pos: dict[str, float],
    acc_cnt: dict[str, int],
    delta_pos: dict[str, float],
    delta_cnt: dict[str, int],
) -> None:
    for k, v in delta_pos.items():
        acc_pos[k] = acc_pos.get(k, 0.0) + v
    for k, v in delta_cnt.items():
        acc_cnt[k] = acc_cnt.get(k, 0) + v


def _smoothed_rates(
    keys: pd.Series,
    pos_map: dict[str, float],
    cnt_map: dict[str, int],
    *,
    global_prior: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    pos_arr = keys.map(pos_map).fillna(0.0).to_numpy(dtype=np.float64)
    cnt_arr = keys.map(cnt_map).fillna(0.0).to_numpy(dtype=np.float64)
    rates = (pos_arr + alpha * global_prior) / np.maximum(cnt_arr + alpha, 1e-9)
    return rates, cnt_arr


def compute_asof_historical_priors(
    train_history: pd.DataFrame | Any,
    target_frame: pd.DataFrame | Any,
    *,
    alpha: float = 100.0,
    min_support: int = 30,
    publication_lag_months: int = 2,
    cold_start_prior: float = 0.20,
) -> pd.DataFrame:
    """Compute strictly as-of smoothed historical delay priors respecting publication lag.

    Only historical rows from ``train_history`` whose ``label_available_utc <= prediction_at_utc``
    contribute to each target flight's priors. Validation or holdout rows must never be
    included in ``train_history``.
    """

    hist_pd = train_history.to_pandas() if hasattr(train_history, "to_pandas") else train_history
    tgt_pd = target_frame.to_pandas() if hasattr(target_frame, "to_pandas") else target_frame

    if len(tgt_pd) == 0:
        return pd.DataFrame(columns=HISTORICAL_PRIOR_COLUMNS)

    if "prediction_at_utc" in tgt_pd.columns:
        pred_utc = _ensure_utc_series(tgt_pd["prediction_at_utc"])
    else:
        sched = _ensure_utc_series(tgt_pd["scheduled_departure_utc"])
        pred_utc = sched - pd.Timedelta(hours=24)

    tgt_keys = pd.DataFrame(
        {
            "origin": tgt_pd["origin"].astype(str).str.upper(),
            "dest": tgt_pd["dest"].astype(str).str.upper(),
            "carrier": tgt_pd["reporting_airline"].astype(str).str.upper(),
        },
        index=tgt_pd.index,
    )
    tgt_keys["route"] = tgt_keys["origin"] + "|" + tgt_keys["dest"]
    tgt_keys["carrier_origin"] = tgt_keys["carrier"] + "|" + tgt_keys["origin"]

    out = pd.DataFrame(
        {
            "origin_hist_delay_rate_asof": np.full(len(tgt_pd), cold_start_prior, dtype=np.float64),
            "carrier_hist_delay_rate_asof": np.full(len(tgt_pd), cold_start_prior, dtype=np.float64),
            "route_hist_delay_rate_asof": np.full(len(tgt_pd), cold_start_prior, dtype=np.float64),
            "carrier_origin_hist_delay_rate_asof": np.full(
                len(tgt_pd), cold_start_prior, dtype=np.float64
            ),
            "route_hist_support_asof": np.zeros(len(tgt_pd), dtype=np.float64),
            "route_hist_unseen_asof": np.ones(len(tgt_pd), dtype=np.int32),
            "hist_prior_freshness_age_days": np.full(len(tgt_pd), np.nan, dtype=np.float64),
        },
        index=tgt_pd.index,
    )

    if len(hist_pd) == 0:
        return out

    hist_clean = pd.DataFrame(
        {
            "origin": hist_pd["origin"].astype(str).str.upper(),
            "carrier": hist_pd["reporting_airline"].astype(str).str.upper(),
            "dest": hist_pd["dest"].astype(str).str.upper(),
            "_label": pd.to_numeric(hist_pd["dep_del15"], errors="coerce"),
            "_avail_utc": compute_label_available_utc(
                hist_pd["flight_date"], publication_lag_months=publication_lag_months
            ),
            "_source_end_utc": compute_source_month_end_utc(hist_pd["flight_date"]),
        }
    ).dropna(subset=["_label", "_avail_utc"])

    if hist_clean.empty:
        return out

    hist_clean["route"] = hist_clean["origin"] + "|" + hist_clean["dest"]
    hist_clean["carrier_origin"] = hist_clean["carrier"] + "|" + hist_clean["origin"]

    checkpoints: list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]] = []
    for (avail_ts, end_ts), grp in hist_clean.groupby(["_avail_utc", "_source_end_utc"], sort=True):
        checkpoints.append((avail_ts, end_ts, grp))

    avail_ns = np.array([int(c[0].value) for c in checkpoints], dtype=np.int64)
    pred_ns = pred_utc.dt.as_unit("ns").astype("int64").to_numpy()
    ckpt_indices = np.searchsorted(avail_ns, pred_ns, side="right") - 1

    max_needed = int(ckpt_indices.max())
    if max_needed < 0:
        return out

    acc_origin_pos: dict[str, float] = {}
    acc_origin_cnt: dict[str, int] = {}
    acc_carrier_pos: dict[str, float] = {}
    acc_carrier_cnt: dict[str, int] = {}
    acc_route_pos: dict[str, float] = {}
    acc_route_cnt: dict[str, int] = {}
    acc_co_pos: dict[str, float] = {}
    acc_co_cnt: dict[str, int] = {}
    global_pos = 0.0
    global_cnt = 0

    for c_idx in range(max_needed + 1):
        _, source_end_ts, grp = checkpoints[c_idx]
        global_pos += float(grp["_label"].sum())
        global_cnt += len(grp)
        _merge_counts(acc_origin_pos, acc_origin_cnt, *_aggregate_counts(grp, "origin"))
        _merge_counts(acc_carrier_pos, acc_carrier_cnt, *_aggregate_counts(grp, "carrier"))
        _merge_counts(acc_route_pos, acc_route_cnt, *_aggregate_counts(grp, "route"))
        _merge_counts(acc_co_pos, acc_co_cnt, *_aggregate_counts(grp, "carrier_origin"))

        row_mask = ckpt_indices == c_idx
        if not np.any(row_mask):
            continue

        g_prior = (global_pos / global_cnt) if global_cnt > 0 else cold_start_prior
        sub = tgt_keys.loc[row_mask]

        origin_rate, _ = _smoothed_rates(
            sub["origin"], acc_origin_pos, acc_origin_cnt, global_prior=g_prior, alpha=alpha
        )
        carrier_rate, _ = _smoothed_rates(
            sub["carrier"], acc_carrier_pos, acc_carrier_cnt, global_prior=g_prior, alpha=alpha
        )
        route_rate, route_cnt = _smoothed_rates(
            sub["route"], acc_route_pos, acc_route_cnt, global_prior=g_prior, alpha=alpha
        )
        co_rate, _ = _smoothed_rates(
            sub["carrier_origin"], acc_co_pos, acc_co_cnt, global_prior=g_prior, alpha=alpha
        )
        freshness_days = (
            (pred_utc.loc[row_mask] - source_end_ts).dt.total_seconds().to_numpy(dtype=np.float64)
            / 86400.0
        )

        out.loc[row_mask, "origin_hist_delay_rate_asof"] = origin_rate
        out.loc[row_mask, "carrier_hist_delay_rate_asof"] = carrier_rate
        out.loc[row_mask, "route_hist_delay_rate_asof"] = route_rate
        out.loc[row_mask, "carrier_origin_hist_delay_rate_asof"] = co_rate
        out.loc[row_mask, "route_hist_support_asof"] = np.log1p(route_cnt)
        out.loc[row_mask, "route_hist_unseen_asof"] = (route_cnt < min_support).astype(np.int32)
        out.loc[row_mask, "hist_prior_freshness_age_days"] = freshness_days

    return out
