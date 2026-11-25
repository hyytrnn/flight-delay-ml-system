"""Auxiliary T-24h air traffic flow forecast study (Gui et al. inspiration).

Separates hourly airport traffic-flow regression (RMSE/MAE) from departure-delay
classification (PR-AUC). Features for target hour t use ONLY realized/published
counts from <= t - 24h, and BTS counts remain marked as retrospective proxy.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.svm import SVR

from flight_delay.data.temporal import TemporalFold
from flight_delay.features.v2 import build_v2_feature_frame, v2_feature_columns
from flight_delay.models.tuning_v2 import _fit_predict_onehot, evaluate_params, load_splits


def build_asof_hourly_flow_table(flights: pl.DataFrame) -> pl.DataFrame:
    """Aggregate hourly origin flight counts and strictly <= t-24h lag features."""

    hourly = (
        flights.with_columns(
            pl.col("scheduled_departure_utc").dt.truncate("1h").alias("hour_utc")
        )
        .group_by(["origin", "hour_utc"])
        .agg(pl.len().cast(pl.Float64).alias("hourly_flow"))
        .sort(["origin", "hour_utc"])
    )
    # Build complete hourly grid per top origin or join exact t-24h, t-48h, t-168h
    lag_24 = hourly.select(
        pl.col("origin"),
        (pl.col("hour_utc") + pl.duration(hours=24)).alias("hour_utc"),
        pl.col("hourly_flow").alias("flow_lag_24h"),
    )
    lag_48 = hourly.select(
        pl.col("origin"),
        (pl.col("hour_utc") + pl.duration(hours=48)).alias("hour_utc"),
        pl.col("hourly_flow").alias("flow_lag_48h"),
    )
    lag_168 = hourly.select(
        pl.col("origin"),
        (pl.col("hour_utc") + pl.duration(hours=168)).alias("hour_utc"),
        pl.col("hourly_flow").alias("flow_lag_7d"),
    )
    return (
        hourly.join(lag_24, on=["origin", "hour_utc"], how="left")
        .join(lag_48, on=["origin", "hour_utc"], how="left")
        .join(lag_168, on=["origin", "hour_utc"], how="left")
        .with_columns(
            pl.col("hour_utc").dt.hour().alias("hour_of_day"),
            pl.col("hour_utc").dt.weekday().alias("weekday"),
            pl.col("hour_utc").dt.month().alias("month"),
            pl.mean_horizontal(["flow_lag_24h", "flow_lag_48h", "flow_lag_7d"]).alias(
                "flow_asof_mean_hist"
            ),
            pl.lit("retrospective_bts_sample_proxy").alias("flow_source_status"),
        )
    )


def evaluate_flow_regressors(
    flow_df: pd.DataFrame, folds: list[TemporalFold], seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare seasonal-naive (lag_7d), HistGBR, and bounded SVR on hourly flow."""

    feature_cols = [
        "flow_lag_24h",
        "flow_lag_48h",
        "flow_lag_7d",
        "flow_asof_mean_hist",
        "hour_of_day",
        "weekday",
        "month",
    ]
    clean = flow_df.dropna(subset=["flow_lag_7d", "hourly_flow"]).copy()
    clean["date"] = pd.to_datetime(clean["hour_utc"]).dt.date
    metrics_rows: list[dict[str, object]] = []
    oof_frames: list[pd.DataFrame] = []

    for fold in folds:
        tr_m = (clean["date"] >= fold.train_start) & (clean["date"] < fold.train_end_exclusive)
        va_m = (clean["date"] >= fold.validation_start) & (
            clean["date"] < fold.validation_end_exclusive
        )
        if not tr_m.any() or not va_m.any():
            continue
        x_tr = clean.loc[tr_m, feature_cols].fillna(0.0)
        y_tr = clean.loc[tr_m, "hourly_flow"].to_numpy()
        x_va = clean.loc[va_m, feature_cols].fillna(0.0)
        y_va = clean.loc[va_m, "hourly_flow"].to_numpy()

        # 1. Seasonal naive (7-day lag)
        naive_pred = clean.loc[va_m, "flow_lag_7d"].to_numpy()
        metrics_rows.append(
            {
                "fold": fold.name,
                "model": "seasonal_naive_7d",
                "rows_eval": len(y_va),
                "rmse": float(root_mean_squared_error(y_va, naive_pred)),
                "mae": float(mean_absolute_error(y_va, naive_pred)),
            }
        )

        # 2. HistGradientBoostingRegressor
        hgbr = HistGradientBoostingRegressor(max_iter=150, max_depth=6, random_state=seed)
        hgbr.fit(x_tr, y_tr)
        hgbr_pred = np.clip(hgbr.predict(x_va), 0.0, None)
        metrics_rows.append(
            {
                "fold": fold.name,
                "model": "hist_gbr",
                "rows_eval": len(y_va),
                "rmse": float(root_mean_squared_error(y_va, hgbr_pred)),
                "mae": float(mean_absolute_error(y_va, hgbr_pred)),
            }
        )

        # 3. Bounded-sample SVR RBF (Gui et al. benchmark family)
        rng = np.random.default_rng(seed)
        svr_idx = (
            rng.choice(len(x_tr), size=2_500, replace=False)
            if len(x_tr) > 2_500
            else np.arange(len(x_tr))
        )
        svr = SVR(kernel="rbf", C=5.0)
        svr.fit(x_tr.iloc[svr_idx], y_tr[svr_idx])
        svr_pred = np.clip(svr.predict(x_va), 0.0, None)
        metrics_rows.append(
            {
                "fold": fold.name,
                "model": "svr_rbf_2500_sample",
                "rows_eval": len(y_va),
                "rmse": float(root_mean_squared_error(y_va, svr_pred)),
                "mae": float(mean_absolute_error(y_va, svr_pred)),
            }
        )

        va_oof = clean.loc[va_m, ["origin", "hour_utc"]].copy()
        va_oof["fold"] = fold.name
        va_oof["flow_hat_asof"] = hgbr_pred
        va_oof["flow_lag_7d"] = naive_pred
        oof_frames.append(va_oof)

    oof_df = pd.concat(oof_frames, ignore_index=True) if oof_frames else pd.DataFrame()
    return pd.DataFrame(metrics_rows), oof_df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/v2"))
    parser.add_argument("--splits", type=Path, default=Path("configs/splits_v2.yaml"))
    parser.add_argument("--end-exclusive", type=date.fromisoformat, default=date(2025, 1, 1))
    args = parser.parse_args()

    pattern = str(args.data_root / "year=*" / "month=*" / "flights.parquet")
    dev = (
        pl.scan_parquet(pattern)
        .filter(pl.col("flight_date") < args.end_exclusive)
        .collect(engine="streaming")
        .sort("flight_date")
    )
    folds = load_splits(args.splits)
    flow_pl = build_asof_hourly_flow_table(dev)
    flow_pd = flow_pl.to_pandas()

    reg_metrics, oof_flow = evaluate_flow_regressors(flow_pd, folds)
    Path("reports/ablation").mkdir(parents=True, exist_ok=True)
    reg_metrics.to_csv("reports/ablation/flow_regression_v2.csv", index=False)

    # Check if strictly as-of historical flow lags + OOF flow_hat help departure classifier on 2024 validation folds
    dev_pd = dev.to_pandas()
    dev_pd["hour_utc"] = pd.to_datetime(dev_pd["scheduled_departure_utc"], utc=True).dt.floor("h")
    flow_pd["hour_utc"] = pd.to_datetime(flow_pd["hour_utc"], utc=True)
    merged = dev_pd.merge(
        flow_pd[["origin", "hour_utc", "flow_lag_24h", "flow_lag_7d", "flow_asof_mean_hist"]],
        on=["origin", "hour_utc"],
        how="left",
    )
    if not oof_flow.empty:
        oof_flow["hour_utc"] = pd.to_datetime(oof_flow["hour_utc"], utc=True)
        merged = merged.merge(
            oof_flow[["origin", "hour_utc", "flow_hat_asof"]].drop_duplicates(["origin", "hour_utc"]),
            on=["origin", "hour_utc"],
            how="left",
        )
    else:
        merged["flow_hat_asof"] = np.nan

    f3_feats = build_v2_feature_frame(merged, "F3")
    flow_cols = ["flow_lag_24h", "flow_lag_7d", "flow_asof_mean_hist"]
    f3_flow_feats = pd.concat([f3_feats, merged[flow_cols]], axis=1)
    labels = merged["dep_del15"].to_numpy().astype(int)
    dates = pd.to_datetime(merged["flight_date"])

    xgb_params = {
        "n_estimators": 280,
        "max_depth": 6,
        "learning_rate": 0.08,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.0,
        "reg_lambda": 1.0,
        "reg_alpha": 0.05,
    }
    base_res = evaluate_params(
        model_name="xgboost",
        params=xgb_params,
        features=f3_feats,
        labels=labels,
        dates=dates,
        folds=folds,
        feature_group="F3",
        seed=42,
    )
    num_f3, cat_f3 = v2_feature_columns("F3")
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    clf_rows = [{"variant": "F3_baseline", **r.__dict__} for r in base_res]
    for fold in folds:
        tr_m = (dates >= pd.Timestamp(fold.train_start)) & (dates < pd.Timestamp(fold.train_end_exclusive))
        va_m = (dates >= pd.Timestamp(fold.validation_start)) & (dates < pd.Timestamp(fold.validation_end_exclusive))
        scores, fit_s = _fit_predict_onehot(
            "xgboost",
            xgb_params,
            num_f3 + flow_cols,
            cat_f3,
            f3_flow_feats.loc[tr_m],
            labels[tr_m],
            f3_flow_feats.loc[va_m],
            42,
            dates_train=dates.loc[tr_m],
        )
        clf_rows.append(
            {
                "variant": "F3_plus_asof_flow_lags",
                "fold": fold.name,
                "rows_train": int(tr_m.sum()),
                "rows_validation": int(va_m.sum()),
                "positive_rate_validation": float(labels[va_m].mean()),
                "pr_auc": float(average_precision_score(labels[va_m], scores)),
                "roc_auc": float(roc_auc_score(labels[va_m], scores)),
                "brier": float(brier_score_loss(labels[va_m], scores)),
                "fit_seconds": fit_s,
            }
        )
    clf_df = pd.DataFrame(clf_rows)
    clf_df.to_csv("reports/ablation/flow_classifier_ablation_v2.csv", index=False)
    print("Flow regression:\n", reg_metrics.groupby("model")[["rmse", "mae"]].mean())
    print("Classifier ablation:\n", clf_df.groupby("variant")[["pr_auc", "brier"]].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
