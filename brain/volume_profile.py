"""Volume profile: how much volume traded at each price level over a
lookback window, the Point of Control (POC, the single highest-volume
price), and the value area (the tightest band of bins holding ~70% of
total volume -- the standard value-area definition).

Per-bar volume is assigned to a single bin at the bar's typical price
((high+low+close)/3) rather than distributed proportionally across the
bar's high-low range. The proportional-distribution version is more
accurate but needs sub-bar price resolution this project doesn't have
(only OHLCV bars, no tick data) to do properly -- flagged as a
simplification, not a hidden approximation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

DEFAULT_NUM_BINS = 20
DEFAULT_VALUE_AREA_PCT = 0.70


@dataclass(frozen=True)
class VolumeProfile:
    bin_low: List[float]
    bin_high: List[float]
    bin_volume: List[float]
    poc_price: float             # midpoint of the highest-volume bin
    value_area_low: float
    value_area_high: float


def compute_volume_profile(bars: pd.DataFrame, num_bins: int = DEFAULT_NUM_BINS, value_area_pct: float = DEFAULT_VALUE_AREA_PCT) -> VolumeProfile:
    if bars.empty:
        raise ValueError("cannot compute a volume profile on an empty bars DataFrame")

    typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3
    price_min, price_max = float(typical_price.min()), float(typical_price.max())
    if price_max == price_min:
        price_max = price_min + 0.01  # degenerate single-price case -- widen trivially so binning doesn't divide by zero

    bin_width = (price_max - price_min) / num_bins
    bin_low = [price_min + i * bin_width for i in range(num_bins)]
    bin_high = [b + bin_width for b in bin_low]
    bin_volume = [0.0] * num_bins

    for price, volume in zip(typical_price, bars["volume"]):
        idx = min(int((price - price_min) / bin_width), num_bins - 1)
        bin_volume[idx] += float(volume)

    poc_idx = max(range(num_bins), key=lambda i: bin_volume[i])
    poc_price = (bin_low[poc_idx] + bin_high[poc_idx]) / 2

    total_volume = sum(bin_volume)
    target = total_volume * value_area_pct
    included = {poc_idx}
    accumulated = bin_volume[poc_idx]
    lo, hi = poc_idx, poc_idx
    while accumulated < target and (lo > 0 or hi < num_bins - 1):
        expand_low = bin_volume[lo - 1] if lo > 0 else -1
        expand_high = bin_volume[hi + 1] if hi < num_bins - 1 else -1
        if expand_high >= expand_low:
            hi += 1
            accumulated += bin_volume[hi]
            included.add(hi)
        else:
            lo -= 1
            accumulated += bin_volume[lo]
            included.add(lo)

    return VolumeProfile(bin_low, bin_high, bin_volume, poc_price, bin_low[lo], bin_high[hi])


def price_in_value_area(price: float, profile: VolumeProfile) -> bool:
    return profile.value_area_low <= price <= profile.value_area_high


DEFAULT_POC_DISTANCE_CAP = 2.0  # in value-area-widths -- see node_quality_check


def node_quality_check(price: float, profile: VolumeProfile, poc_distance_cap: float = DEFAULT_POC_DISTANCE_CAP) -> str:
    """A second, richer read of the same profile beyond price_in_value_area's
    binary in/out check -- this is what VPVR (Volume Profile Visible Range)
    is FOR: not just "inside or outside the value area" but "is this
    specific price a good place to enter." Two things, both aimed at the
    "don't take trades at bad points" goal:

    1. Is price sitting over a BELOW-average-volume node -- an "air
       pocket" the market moved through quickly rather than a congested
       zone it spent real time trading at? This is the same imbalance
       idea a Fair Value Gap itself represents (a gap IS a low-volume
       node by construction), so this reinforces rather than duplicates
       the FVG signal -- it's asking the same question about a wider
       window of history, not just the 3 candles that formed the gap.
       A high-volume node is where two-sided disagreement already
       happened -- price chops and rejects there far more often than it
       breaks cleanly through, which is exactly the kind of "human enters
       right into resistance" mistake this exists to catch.
    2. Has price already travelled unreasonably far from the Point of
       Control (POC), the single most-traded price in the window? POC
       acts as a magnet/fair-value anchor -- a candidate trade that's
       already `poc_distance_cap` value-area-widths away from it is
       chasing a move that's mostly already happened, not catching a
       fresh one. This is a plain distance check, not a forecast: it
       doesn't claim to know price will revert, only that entering this
       far from fair value is a worse-odds entry point than doing so
       closer to it.

    "fail" on either condition. "n/a" if the profile is degenerate (zero-
    width bins or value area -- see compute_volume_profile's docstring on
    the single-price edge case).
    """
    bin_width = profile.bin_high[0] - profile.bin_low[0]
    if bin_width <= 0:
        return "n/a"

    idx = min(max(int((price - profile.bin_low[0]) / bin_width), 0), len(profile.bin_volume) - 1)
    avg_volume = sum(profile.bin_volume) / len(profile.bin_volume)
    at_low_volume_node = profile.bin_volume[idx] < avg_volume

    value_area_width = profile.value_area_high - profile.value_area_low
    if value_area_width <= 0:
        return "n/a"
    poc_extended = abs(price - profile.poc_price) > poc_distance_cap * value_area_width

    return "pass" if (at_low_volume_node and not poc_extended) else "fail"
