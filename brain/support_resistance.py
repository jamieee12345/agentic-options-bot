"""Support/resistance levels, clustered from swing points.

Built on top of market_structure.find_swing_points() rather than a second,
separate swing-detection pass -- a support/resistance level IS a swing
point (or a cluster of nearby ones) that price has respected before, so
there's no reason to compute swings twice with different logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from brain.market_structure import SwingPoint

DEFAULT_CLUSTER_TOLERANCE_PCT = 0.005  # swing points within 0.5% of each other count as the same level


@dataclass(frozen=True)
class Level:
    price: float             # average of the clustered swing points
    kind: str                 # "support" or "resistance"
    touches: int               # how many swing points clustered into this level -- more touches = more significant


def cluster_levels(swing_points: List[SwingPoint], tolerance_pct: float = DEFAULT_CLUSTER_TOLERANCE_PCT) -> List[Level]:
    levels: List[Level] = []
    for kind, swing_kind in (("resistance", "high"), ("support", "low")):
        prices = sorted(p.price for p in swing_points if p.kind == swing_kind)
        cluster: List[float] = []
        for price in prices:
            if cluster and (price - cluster[-1]) / cluster[-1] > tolerance_pct:
                levels.append(Level(sum(cluster) / len(cluster), kind, len(cluster)))
                cluster = []
            cluster.append(price)
        if cluster:
            levels.append(Level(sum(cluster) / len(cluster), kind, len(cluster)))
    return levels


def nearest_level(current_price: float, levels: List[Level], kind: str) -> Optional[Level]:
    candidates = [lvl for lvl in levels if lvl.kind == kind]
    if not candidates:
        return None
    if kind == "resistance":
        above = [lvl for lvl in candidates if lvl.price >= current_price]
        return min(above, key=lambda lvl: lvl.price) if above else None
    below = [lvl for lvl in candidates if lvl.price <= current_price]
    return max(below, key=lambda lvl: lvl.price) if below else None


def distance_pct(current_price: float, level: Level) -> float:
    return abs(level.price - current_price) / current_price
