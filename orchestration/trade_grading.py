"""Post-trade grading: a deterministic, fully-explainable score for each
CLOSED trade, split into two independent axes on purpose --

  PROCESS grade: was this a well-supported setup AT ENTRY (the confluence
  score computed before the outcome was known)? Bid/ask spread and DTE
  aren't part of this score, even though they're recorded alongside it --
  both are already HARD gates (risk.max_bid_ask_spread_pct,
  options.target_dte_min/max) that every opened trade already had to clear,
  so they don't vary trade-to-trade the way confluence score does. They're
  persisted for the record (see orchestration/trade_log.py), not graded.

  OUTCOME grade: how it actually turned out -- P&L, bucketed against this
  account's own stop_loss_pct/take_profit_pct so "a win" and "a loss" mean
  the same thing here that they mean in config/settings.yaml, not an
  arbitrary threshold picked in this file.

Why split them: "the trade lost" and "the process was bad" are NOT the
same claim. A well-supported trade can still lose -- that's what
brain/confluence.py's own honesty note means by "evidence, not a
guarantee." Grading only the outcome would reward good luck and punish bad
luck identically to good and bad decisions, which teaches nothing.

WHAT THIS MODULE IS NOT: an auto-tuner. It grades and aggregates
(aggregate_check_performance below) so a human can spot a real pattern --
e.g. "trades where RSI momentum failed lose more often than trades where
it passed" -- across enough trades to mean something. It does not rewrite
config/settings.yaml, reweight checks, or change any threshold on its own.
Turning an observed pattern into an actual config change should be a
deliberate decision made with real sample size behind it, the same way
min_confluence_score's 0.6 was chosen deliberately rather than discovered
by the system itself -- an autonomously self-adjusting live-money trading
system is a bad idea long before it's a good one, and that line doesn't
move just because this file exists now.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from orchestration.trade_log import ClosedTrade

# Below this many closed trades, aggregate check-performance stats are
# noise, not signal -- a 2-trade "0% win rate" and a 200-trade one look
# identical in a naive table but mean completely different things.
MIN_TRADES_FOR_AGGREGATE = 10

# A confluence score comfortably above the pass bar (not just barely
# clearing it) counts as a genuinely strong setup for grading purposes.
# Independent of config/settings.yaml's min_confluence_score -- that's the
# GATE (must clear this to trade at all); this is a stricter bar used only
# to label a trade's process quality after the fact.
STRONG_PROCESS_SCORE = 0.8


@dataclass(frozen=True)
class TradeGrade:
    letter: str              # "A" (best) through "F" (worst), or "N/A" if ungraded
    process_label: str
    outcome_label: str
    explanation: str


def grade_trade(trade: ClosedTrade, stop_loss_pct: float, take_profit_pct: float) -> Optional[TradeGrade]:
    """None only when there's nothing to grade on (pnl_pct unavailable --
    entry_notional was 0, shouldn't happen for a real trade but the field
    is Optional in trade_log.py). A trade opened before this session's
    confluence-snapshot logging existed grades on outcome alone, clearly
    labeled "no confluence data" rather than silently assuming a score.
    """
    if trade.pnl_pct is None:
        return None

    has_process_data = trade.confluence_score is not None
    process_strong = has_process_data and trade.confluence_score >= STRONG_PROCESS_SCORE

    if trade.pnl_pct >= take_profit_pct * 0.8:
        outcome_label = "clean win"
    elif trade.pnl_pct > 0:
        outcome_label = "small win"
    elif trade.pnl_pct > -stop_loss_pct * 0.5:
        outcome_label = "small loss"
    else:
        outcome_label = "full loss (near/at stop-loss)"
    is_win = trade.pnl_pct > 0

    if not has_process_data:
        process_label = "no confluence data (logged before this was tracked)"
        letter = "B" if is_win else "D"
        explanation = f"{outcome_label}, but this trade predates confluence-score logging -- process can't be graded."
        return TradeGrade(letter, process_label, outcome_label, explanation)

    process_label = "strong setup" if process_strong else "marginal setup (near the confluence floor)"

    if process_strong and outcome_label == "clean win":
        letter, explanation = "A", "Well-supported entry, worked as intended."
    elif process_strong and outcome_label == "small win":
        letter, explanation = "A", "Well-supported entry, modest but real gain."
    elif process_strong and outcome_label == "small loss":
        letter, explanation = "B", "Well-supported entry that still lost a little -- normal variance, not a process failure."
    elif process_strong and outcome_label == "full loss (near/at stop-loss)":
        letter, explanation = "C", "Well-supported entry that hit the stop anyway -- worth a closer look at what the checks missed, but not necessarily a bad process."
    elif not process_strong and outcome_label in ("clean win", "small win"):
        letter, explanation = "C", "Won, but the setup was only marginally above the confluence floor -- don't over-trust this pattern from one result."
    elif not process_strong and outcome_label == "small loss":
        letter, explanation = "D", "Marginal setup, and it lost -- the score being right at the floor may not have been enough support."
    else:
        letter, explanation = "F", "Marginal setup that hit the stop -- the kind of trade a stricter confluence bar would filter out."

    return TradeGrade(letter, process_label, outcome_label, explanation)


@dataclass(frozen=True)
class CheckPerformance:
    check_key: str
    pass_trades: int
    pass_win_rate: Optional[float]
    fail_trades: int
    fail_win_rate: Optional[float]


def aggregate_check_performance(trades: List[ClosedTrade], min_trades: int = MIN_TRADES_FOR_AGGREGATE) -> List[CheckPerformance]:
    """For each confluence check that was applicable (pass or fail, not
    n/a) across enough closed trades, compare the win rate of trades where
    it passed against trades where it failed. This is the "spot a pattern"
    surface the module docstring describes -- purely observational, read
    by a human (dashboard's Trade quality section), never written back
    into any config.

    Only trades with confluence_details populated (i.e. opened after this
    session's logging upgrade) count -- older trades silently excluded
    rather than skewing the stats with missing data treated as "n/a".
    """
    graded = [t for t in trades if t.confluence_details and t.pnl_pct is not None]
    if len(graded) < min_trades:
        return []

    by_check: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: {"pass": [], "fail": []})
    for t in graded:
        is_win = t.pnl_pct > 0
        for check_key, verdict in t.confluence_details.items():
            if verdict in ("pass", "fail"):
                by_check[check_key][verdict].append(is_win)

    def _win_rate(results: List[bool]) -> Optional[float]:
        return (sum(results) / len(results)) if results else None

    return [
        CheckPerformance(
            check_key=key,
            pass_trades=len(buckets["pass"]), pass_win_rate=_win_rate(buckets["pass"]),
            fail_trades=len(buckets["fail"]), fail_win_rate=_win_rate(buckets["fail"]),
        )
        for key, buckets in sorted(by_check.items())
    ]
