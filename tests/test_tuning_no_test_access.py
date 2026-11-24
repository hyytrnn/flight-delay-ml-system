from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import polars as pl

from flight_delay.data.temporal import TemporalFold
from flight_delay.features.v2 import build_v2_feature_frame
from flight_delay.models.tuning_v2 import (
    _inner_chronological_split,
    cleanup_stale_running_trials,
    evaluate_params,
    evaluate_params_with_oof,
)


def tiny_tuning_frame() -> pl.DataFrame:
    rows = []
    for year, month in [(2023, 1), (2023, 6), (2024, 1), (2024, 2)]:
        for index in range(20):
            rows.append(
                {
                    "flight_date": date(year, month, index % 20 + 1),
                    "scheduled_departure_local": datetime(year, month, index % 20 + 1, index % 24),
                    "reporting_airline": "AA" if index % 2 else "UA",
                    "origin": "JFK",
                    "dest": "LAX",
                    "distance": 2475.0,
                    "crs_elapsed_time": 360.0,
                    "dep_del15": index % 2,
                }
            )
    return pl.DataFrame(rows)


def test_temporal_objective_never_receives_future_holdout() -> None:
    frame = tiny_tuning_frame()
    features = build_v2_feature_frame(frame, "F0")
    dates = pd.to_datetime(frame.get_column("flight_date").to_pandas())
    labels = frame.get_column("dep_del15").to_numpy()
    fold = TemporalFold(
        name="fold",
        train_start=date(2023, 1, 1),
        train_end_exclusive=date(2024, 1, 1),
        validation_start=date(2024, 1, 1),
        validation_end_exclusive=date(2024, 3, 1),
    )
    results = evaluate_params(
        model_name="xgboost",
        params={
            "n_estimators": 10,
            "max_depth": 2,
            "learning_rate": 0.1,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
        },
        features=features,
        labels=labels,
        dates=dates,
        folds=[fold],
        feature_group="F0",
        seed=42,
    )
    assert len(results) == 1
    assert results[0].rows_train == 40
    assert results[0].rows_validation == 40


def test_baselines_and_inner_chronological_split_preserve_order() -> None:
    dates = pd.Series(pd.date_range("2023-01-01", periods=800, freq="D"))
    split = _inner_chronological_split(dates, fraction=0.15)
    assert split is not None
    in_tr, in_va = split
    assert dates.loc[in_tr].max() < dates.loc[in_va].min()

    frame = tiny_tuning_frame()
    features = build_v2_feature_frame(frame, "F0")
    frame_dates = pd.to_datetime(frame.get_column("flight_date").to_pandas())
    labels = frame.get_column("dep_del15").to_numpy()
    fold = TemporalFold(
        name="fold",
        train_start=date(2023, 1, 1),
        train_end_exclusive=date(2024, 1, 1),
        validation_start=date(2024, 1, 1),
        validation_end_exclusive=date(2024, 3, 1),
    )
    for model_name in ["dummy_prior", "logistic_regression", "hist_gradient_boosting"]:
        res = evaluate_params(
            model_name=model_name,  # type: ignore[arg-type]
            params={},
            features=features,
            labels=labels,
            dates=frame_dates,
            folds=[fold],
            feature_group="F0",
            seed=42,
            class_weight_mode="balanced" if model_name == "logistic_regression" else "none",
        )
        assert len(res) == 1
        assert 0.0 <= res[0].pr_auc <= 1.0


def test_probe_vs_refit_row_counts_and_stale_trial_cleanup(tmp_path) -> None:
    import optuna

    rows = []
    for m_idx in range(1, 13):
        month = ((m_idx - 1) % 6) + 1
        year = 2023 if m_idx <= 6 else 2024
        for day in range(1, 21):
            for k in range(7):
                rows.append(
                    {
                        "flight_date": date(year, month, day),
                        "scheduled_departure_local": datetime(year, month, day, (day + k) % 24),
                        "reporting_airline": "AA" if (day + k) % 2 == 0 else "UA",
                        "origin": "JFK",
                        "dest": "LAX",
                        "distance": 2475.0,
                        "crs_elapsed_time": 360.0,
                        "dep_del15": int((day + k) % 3 == 0),
                    }
                )
    frame = pl.DataFrame(rows)
    features = build_v2_feature_frame(frame, "F0")
    dates = pd.to_datetime(frame.get_column("flight_date").to_pandas())
    labels = frame.get_column("dep_del15").to_numpy()
    fold = TemporalFold(
        name="fold_test",
        train_start=date(2023, 1, 1),
        train_end_exclusive=date(2024, 1, 1),
        validation_start=date(2024, 1, 1),
        validation_end_exclusive=date(2024, 7, 1),
    )

    tr_mask = (dates >= pd.Timestamp(fold.train_start)) & (
        dates < pd.Timestamp(fold.train_end_exclusive)
    )
    va_mask = (dates >= pd.Timestamp(fold.validation_start)) & (
        dates < pd.Timestamp(fold.validation_end_exclusive)
    )
    inner = _inner_chronological_split(dates.loc[tr_mask], fraction=0.15)
    assert inner is not None
    in_tr, in_va = inner
    assert dates.loc[tr_mask].iloc[in_tr].max() < dates.loc[tr_mask].iloc[in_va].min()
    assert dates.loc[tr_mask].max() < dates.loc[va_mask].min()

    for model_name, params in [
        (
            "xgboost",
            {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.1},
        ),
        (
            "lightgbm",
            {"n_estimators": 80, "max_depth": 3, "learning_rate": 0.1, "num_leaves": 15},
        ),
        (
            "catboost",
            {"iterations": 70, "depth": 3, "learning_rate": 0.1},
        ),
    ]:
        probe_res, probe_oof = evaluate_params_with_oof(
            model_name=model_name,  # type: ignore[arg-type]
            params=params,
            features=features,
            labels=labels,
            dates=dates,
            folds=[fold],
            feature_group="F0",
            seed=42,
            eval_strategy="probe_only",
        )
        refit_res, refit_oof = evaluate_params_with_oof(
            model_name=model_name,  # type: ignore[arg-type]
            params=params,
            features=features,
            labels=labels,
            dates=dates,
            folds=[fold],
            feature_group="F0",
            seed=42,
            eval_strategy="refit_full_train",
        )
        assert probe_res[0].rows_inner_train + probe_res[0].rows_inner_val == probe_res[0].rows_train
        assert probe_res[0].rows_fit_actual == probe_res[0].rows_inner_train
        assert refit_res[0].rows_fit_actual == refit_res[0].rows_train
        assert probe_res[0].best_iteration is not None
        assert len(probe_oof) == len(refit_oof) == int(va_mask.sum())

    db_path = tmp_path / "test_study.db"
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(study_name="s1", storage=storage)
    study.ask()
    assert study.trials[0].state == optuna.trial.TrialState.RUNNING
    cleaned = cleanup_stale_running_trials(storage, "s1")
    assert cleaned == 1
    reloaded = optuna.load_study(study_name="s1", storage=storage)
    assert reloaded.trials[0].state == optuna.trial.TrialState.FAIL


