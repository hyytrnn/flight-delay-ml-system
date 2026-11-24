"""Support-aware rates and drift metrics used by V2 EDA."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


def wilson_interval(positive: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    rate = positive / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2))
        / denominator
    )
    return center - half_width, center + half_width


def total_variation_distance(
    reference: Mapping[str, float], current: Mapping[str, float]
) -> float:
    keys = set(reference).union(current)
    return 0.5 * sum(abs(reference.get(key, 0.0) - current.get(key, 0.0)) for key in keys)


def jensen_shannon_divergence(
    reference: Mapping[str, float], current: Mapping[str, float]
) -> float:
    keys = sorted(set(reference).union(current))
    p = np.asarray([reference.get(key, 0.0) for key in keys], dtype=float)
    q = np.asarray([current.get(key, 0.0) for key in keys], dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return math.nan
    p /= p.sum()
    q /= q.sum()
    midpoint = 0.5 * (p + q)

    def kl(left: np.ndarray, right: np.ndarray) -> float:
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)

