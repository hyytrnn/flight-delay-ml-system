from __future__ import annotations

import argparse

import pytest

from scripts.download_bts import Month, month_range, parse_month


def test_month_range_crosses_year_boundary_inclusively() -> None:
    assert month_range(Month(2024, 11), Month(2025, 2)) == [
        Month(2024, 11),
        Month(2024, 12),
        Month(2025, 1),
        Month(2025, 2),
    ]


def test_month_range_rejects_reverse_order() -> None:
    with pytest.raises(ValueError, match="end month"):
        month_range(Month(2025, 2), Month(2024, 11))


def test_parse_month_rejects_invalid_month() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_month("2025-13")
