from __future__ import annotations

import pytest

from scripts.paper_transfer_audit import build_audit, confusion_metrics


def test_ajala_confusion_matrix_recomputes_reported_figure_metrics() -> None:
    metrics = confusion_metrics(tn=6050, fp=950, fn=1420, tp=1580)
    assert metrics["rows"] == 10_000
    assert metrics["accuracy"] == pytest.approx(0.763)
    assert metrics["precision"] == pytest.approx(1580 / 2530)
    assert metrics["recall"] == pytest.approx(1580 / 3000)
    assert metrics["f1"] == pytest.approx(3160 / 5530)


def test_paper_table_and_figure_discrepancy_is_explicit() -> None:
    audit = build_audit()
    discrepancy = audit["absolute_discrepancy"]
    assert isinstance(discrepancy, dict)
    assert discrepancy["accuracy"] > 0.09
    assert discrepancy["f1"] > 0.06
