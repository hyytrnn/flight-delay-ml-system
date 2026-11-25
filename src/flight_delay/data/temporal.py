"""Calendar-boundary temporal folds for V2 model development."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_start: date
    train_end_exclusive: date
    validation_start: date
    validation_end_exclusive: date

    def validate(self) -> None:
        if not (
            self.train_start < self.train_end_exclusive
            <= self.validation_start
            < self.validation_end_exclusive
        ):
            raise ValueError(f"invalid or overlapping temporal fold: {self.name}")


def parse_date(value: str | date) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def fold_from_mapping(mapping: dict[str, Any]) -> TemporalFold:
    fold = TemporalFold(
        name=str(mapping["name"]),
        train_start=parse_date(mapping["train_start"]),
        train_end_exclusive=parse_date(mapping["train_end_exclusive"]),
        validation_start=parse_date(mapping["validation_start"]),
        validation_end_exclusive=parse_date(mapping["validation_end_exclusive"]),
    )
    fold.validate()
    return fold


def split_fold(frame: pl.DataFrame, fold: TemporalFold) -> tuple[pl.DataFrame, pl.DataFrame]:
    fold.validate()
    event_date = pl.col("flight_date")
    train = frame.filter(
        (event_date >= fold.train_start) & (event_date < fold.train_end_exclusive)
    )
    validation = frame.filter(
        (event_date >= fold.validation_start)
        & (event_date < fold.validation_end_exclusive)
    )
    if train.height == 0 or validation.height == 0:
        raise ValueError(f"empty train or validation partition for {fold.name}")
    if train.get_column("flight_date").max() >= validation.get_column("flight_date").min():
        raise ValueError(f"temporal contamination detected in {fold.name}")
    return train, validation

