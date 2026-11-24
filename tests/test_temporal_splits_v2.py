from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from flight_delay.data.temporal import TemporalFold, fold_from_mapping, split_fold


def test_expanding_fold_keeps_validation_strictly_after_train() -> None:
    frame = pl.DataFrame(
        {
            "flight_date": [
                date(2023, 1, 1),
                date(2023, 12, 31),
                date(2024, 1, 1),
                date(2024, 3, 31),
            ],
            "dep_del15": [0, 1, 0, 1],
        }
    )
    fold = TemporalFold(
        name="fold_1",
        train_start=date(2023, 1, 1),
        train_end_exclusive=date(2024, 1, 1),
        validation_start=date(2024, 1, 1),
        validation_end_exclusive=date(2024, 4, 1),
    )
    train, validation = split_fold(frame, fold)
    assert train.get_column("flight_date").max() < validation.get_column("flight_date").min()
    assert train.height == validation.height == 2


def test_fold_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlapping"):
        TemporalFold(
            name="bad",
            train_start=date(2023, 1, 1),
            train_end_exclusive=date(2024, 2, 1),
            validation_start=date(2024, 1, 1),
            validation_end_exclusive=date(2024, 3, 1),
        ).validate()


def test_fold_mapping_parses_yaml_dates() -> None:
    fold = fold_from_mapping(
        {
            "name": "fold",
            "train_start": "2023-01-01",
            "train_end_exclusive": "2024-01-01",
            "validation_start": "2024-01-01",
            "validation_end_exclusive": "2024-04-01",
        }
    )
    assert fold.validation_start == date(2024, 1, 1)
