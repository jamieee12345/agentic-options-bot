# Trading bot — build log

Status: **config, data layer, and the full Brain (regime detector, regime
strategies, signal generator) are done and tested. End-to-end demo run
against synthetic data. Backtester foundation (portfolio math, walk-forward
windowing, forward-filter HMM algorithm, core metrics) is done and tested,
plus a full walk-forward run. Safety layer (circuit breakers, position
sizing) is done and tested. Orchestration layer (startup, main loop,
shutdown, error handling) is done and tested. Dashboard (regime, portfolio,
positions, recent signals) is done and tested.** Order validation,
correlation checks, portfolio-level limit enforcement, and wiring the real
Broker/RegimeProvider/DataFeed/Dashboard implementations into the live loop
are next, along with the rest of the backtester (regime/confidence tables,
benchmarks, Rich output).

## Layout

```
config/
  settings.yaml         All tunable parameters
  config_loader.py      Loads + validates settings.yaml (stdlib only, no deps)
data/
  fetchers.py            HistoricalDataFetcher (yfinance) + QuoteFetcher (Robinhood, live)
  features.py             Returns, volatility, volume, trend, mean-reversion, momentum, range (22 columns)
brain/
  regime_detector.py      Gaussian HMM volatility classifier, BIC model selection (3-7 states)
  regime_strategies.py    All three tiers (low/mid/high) + rank-to-tier mapping for any state count
  signal_generator.py     Uncertainty gate, rebalancing threshold, Strategy classes (+ regime-label aliases), StrategyOrchestrator
allocation/
  portfolio.py             PortfolioState, compute_rebalance, current_allocation_pct, should_rebalance —
                            shared by the backtester, dashboard, and (eventually) the live loop
demo/
  run_pipeline_demo.py    End-to-end demo on synthetic data (see below)
backtest/
  walk_forward.py          IS/OOS window generation + forward-only HMM filtering
  metrics.py                Sharpe, Sortino, Calmar, max drawdown (+duration), CAGR, trade stats, worst-case stats
  run_backtest_demo.py      Full walk-forward run on synthetic data (see below)
safety/
  circuit_breakers.py       Daily/weekly DD reduce+halt, peak-DD lock-file kill switch, structured event log
  position_sizing.py        Risk-based sizing, regime/portfolio caps, min position, overnight gap-risk adjustment
  order_validation.py        Pre-trade checks: buying power, tradability, bid-ask spread, duplicate orders
  correlation.py              Correlation-based size reduction/rejection against current holdings
  portfolio_limits.py         max_exposure / max_concurrent_positions / max_sector_exposure enforcement
  options_sizing.py           Premium-based sizing for long calls/puts (max_risk = premium paid, no stop distance)
broker/
  robinhood_broker.py         Real Broker implementation (robin_stocks), hard-restricted to ONE account number
                               -- equity orders AND options orders (get_option_positions/place_option_order)
data/
  options_data.py              Options chain lookup + ATM contract selection (robin_stocks, unofficial)
brain/
  fvg_indicators.py             Fair Value Gap detection + volume confirmation (see dedicated section below)
  options_strategy.py           FVG+volume trigger -> confluence gate -> long-call/long-put decision
  confluence.py                  Composes every module below into one pass/fail + score
  market_structure.py            Swing points, uptrend/downtrend/ranging classification, break of structure
  support_resistance.py          Levels clustered from swing points
  supply_demand.py               Demand/supply zones (the base candle before a displacement move)
  liquidity.py                   Equal-highs/lows pools + liquidity sweep detection
  volume_profile.py              Point of control + value area from binned volume
  elliott_wave.py                 3-rule impulse validity check (NOT a full subjective wave count -- see its docstring)
  trend_indicators.py             200-SMA trend filter
orchestration/
  retry.py                  Generic retry-with-exponential-backoff (Robinhood error policy)
  market_hours.py           Market hours check + HMM model-staleness check
  session_state.py          Session state save/load for crash recovery
  main_loop.py               Startup, per-bar loop, shutdown -- the error-handling policy from the spec
  order_execution.py          Signal -> validated order -> (dry-run log | broker.place_order)
  options_execution.py         Signal -> simple long call/put -> (dry-run log | broker.place_option_order)
dashboard/
  terminal_dashboard.py      Streamlined always-reviewable dashboard: regime, portfolio, positions, signals
  web_dashboard.py            Same data, rendered as self-contained HTML with auto-refresh for live monitoring
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For live trading later, create a `.env` file (and add it to `.gitignore` —
never commit credentials):

```
ROBINHOOD_USERNAME=you@example.com
ROBINHOOD_PASSWORD=...
ROBINHOOD_MFA_SECRET=...          # only if you use app-based 2FA
ROBINHOOD_ACCOUNT_NUMBER=...       # the ONE account broker/robinhood_broker.py is allowed to touch
```

`ROBINHOOD_ACCOUNT_NUMBER` is a hard restriction, not a default: every
account-scoped call in `broker/robinhood_broker.py` passes it explicitly to
`robin_stocks`, and `verify_account()` refuses to proceed if the account
that comes back doesn't match it exactly. Set it to whichever account you've
designated for bot access (e.g. Robinhood's own "agentic" account
permission) — never your primary/default account, unless that's genuinely
the one you mean to let this code trade.

## Design notes worth remembering

- **The HMM is a volatility classifier, not a direction predictor.** It's
  fit only on volatility-related features (`realized_vol_20`,
  `normalized_atr_14`, `vol_ratio_5_20`). Returns enter *only* in
  `label_states()`, after fitting, purely to name the states for
  readability — they never influence what the states are.
- **State indices aren't stable across HMM refits.** `StrategyOrchestrator.
  rebuild_mapping()` must be called after every retrain — confirmed by a
  test that simulates state 0 flipping from "calmest" to "most turbulent"
  between two fits.
- **Robinhood has no official equities API.** `RobinhoodQuoteFetcher` uses
  the unofficial `robin_stocks` library. It's your account/your risk:
  unsupported, can break on Robinhood's app changes, and outside their ToS
  for non-crypto accounts.
- **Backtest/live feature parity matters.** `features.py` is deliberately
  the same code path for both, so a backtest's indicator values are exactly
  what live trading would have seen.
- **Credentials never live in code.** Always environment variables, loaded
  from a git-ignored `.env`.

## Interpretation calls made along the way (flagged for review)

- "Volume ratio (5,20)" under Volatility → implemented as a **volatility**
  ratio (5-period realized vol ÷ 20-period), since trading volume already
  has its own section.
- RSI z-score window defaulted to 50 periods (not specified).
- "Slope of N-period SMA" implemented as an OLS regression slope over a
  trailing window equal to that same N.
- "Position <= .33 / >= .67" for the volatility-tier mapping implemented as
  exact thirds (1/3, 2/3) rather than literal two-decimal cutoffs, so tier
  boundaries don't drift inconsistently across different state counts.
- `is_flickering` (not specified precisely) uses a 5-observation window
  with a 2-transition threshold.
- The walk-forward spec's allocation formula `cash = delta * price` is
  implemented as `cash = cash - delta * price` instead — the literal
  version would discard the existing cash balance on every rebalance.
  Flagged prominently when built; confirmed by a test showing the two
  formulas diverge on identical inputs.
- "Run filtered HMM (forward algorithm)" implemented as the actual forward
  algorithm (alpha recursion) against the model's own fitted parameters,
  not `hmmlearn.predict_proba()` (which does forward-*backward* smoothing —
  fine only if never handed future bars, easy to misuse by accident).
- Added `max_sector_exposure: 0.30` to `settings.yaml`'s `risk` section (and
  `config_loader.py`'s `RiskConfig`) for "max correlated exposure: 30% in
  one sector" — config plumbing only; the actual sector-correlation
  enforcement is still pending.
- A trailing partial walk-forward window (not enough bars left for a full
  OOS period) is dropped rather than truncated.
- Circuit breakers measure daily/weekly drawdown from that period's
  start-of-period equity (not an intraday running peak), and once a reduce
  or halt triggers it's sticky for the rest of that day/week even if equity
  partially recovers before the period rolls over.

## What's been tested, and how

Everything below was actually run, not just written:

- **`config/config_loader.py`** — loaded against the real `settings.yaml`;
  validation checked both ways: accepts the current config (surfacing the
  `max_exposure` redundancy warning noticed by eye earlier), and correctly
  rejects a deliberately broken config (`daily_loss_reduce >= daily_loss_halt`).
- **`data/features.py`** — run against 500 days of synthetic OHLCV data
  (long enough to clear the 200-period SMA warmup). All 22 columns checked:
  NaNs appear only inside each feature's own warmup window, `rsi_14` stays
  within [0, 100], `normalized_atr_14` and `vol_ratio_5_20` stay strictly
  positive.
- **`data/fetchers.py`** — **not runnable in this sandbox** (no network
  access here). First thing worth running in your own environment:
  ```bash
  python3 -c "from data.fetchers import YFinanceHistoricalFetcher as F; print(F().get_bars('AAPL', '2024-01-01', '2024-06-01').tail())"
  ```
  `RobinhoodQuoteFetcher` additionally needs real credentials in `.env` and
  will prompt for 2FA on first login if `ROBINHOOD_MFA_SECRET` isn't set.
- **`brain/regime_detector.py`** — **not fully testable here** either (no
  `hmmlearn`, no network to install it). What *was* tested directly, without
  needing the library: `n_params()` against two hand-computed cases;
  `compute_bic()` confirmed to penalize extra states at equal
  log-likelihood; and the labeling logic, fed a fake decoded state sequence
  with known mean returns per state, correctly sorts and labels them
  (`bear`/`neutral`/`bull` in the right order) and correctly raises if a
  state is never assigned. First thing worth doing once `hmmlearn` is
  installed: fit on real SPY data and eyeball `RegimeModel.candidates` (the
  full BIC table) and `state_labels` for sanity.
- **`brain/regime_strategies.py`** — tested end to end: tier boundaries
  hand-verified for every state count 3 through 7 (including confirming
  n_states=4 genuinely never produces a "mid" tier — a real consequence of
  exact thirds, not a bug); volatility ranking tested against a fake
  model's `means_` array; both branches of the mid-vol price-vs-EMA
  conditional confirmed correct; high-vol's stop confirmed wider than
  mid-vol's at identical inputs; `resolve_strategy()` tested end-to-end for
  both the calmest and most turbulent states in a 5-state example.
- **`brain/signal_generator.py`** — tested thoroughly: `is_flickering`
  across five hand-picked state sequences; `apply_uncertainty` confirmed to
  trigger from *either* low confidence or flickering independently, leave
  values untouched when neither fires, and NOT trigger exactly at the 0.55
  threshold (spec says strictly `<`); `should_rebalance` confirmed to not
  fire at exactly 10% (spec says strictly `>`); each Strategy class tested
  for a valid signal and for correctly returning `None` on insufficient
  data; the orchestrator tested through a simulated HMM retrain where state
  0's meaning flips from "calmest" to "most turbulent" between fits; and
  the 5 backward-compatible aliases confirmed identical by both identity
  (`is`) and matching output.
- **`demo/run_pipeline_demo.py`** — full end-to-end run on synthetic
  calm/crash/recovery price data (see below).
- **`backtest/portfolio.py`** — every formula tested against hand-computed
  numbers, including the spec's own worked leverage example (1.25x
  allocation correctly produces negative cash / margin debt while equity
  still nets out correctly), slippage direction on both buys and sells, a
  no-op rebalance correctly charging zero commission, and — most
  importantly — a test confirming the corrected cash formula diverges from
  the literal (buggy) spec formula on identical inputs.
- **`backtest/walk_forward.py`** — window generation hand-verified (5
  windows from 1000 bars at IS=252/OOS=126/step=126, matching a manual
  calculation) and confirmed to yield zero windows rather than crash on
  insufficient data. The forward filter was checked against an independent
  brute-force linear-space re-implementation of the same recursion
  (matched to 1e-10), and — the test that matters most — confirmed that
  appending additional future bars to the observation sequence leaves every
  previously-computed filtered probability completely unchanged.
- **`backtest/metrics.py`** — every function tested against hand-computed
  values. One real bug caught by the test suite itself: the original
  `std == 0` check for a "flat" return series failed because floating-point
  standard deviation of a constant series is essentially never bit-exact
  zero (it came out to ~2e-19), which would have let Sharpe/Sortino divide
  by a near-zero number and return a wildly inflated ratio instead of the
  intended 0.0 — fixed with a tolerance check instead of exact equality.
  Also confirmed Sortino rewards upside-skewed volatility relative to
  Sharpe, and — deliberately — that `max_drawdown`'s own `underwater_bars`
  and `longest_underwater_streak` can (and in the test case, do) come from
  two *different* drawdown events.

## End-to-end demo

Runs the full chain — synthetic OHLCV → features → a quantile-based
stand-in for the HMM → regime strategies → signal generator → orchestrator
— and prints the resulting signal at sample points across a calm bull →
turbulent selloff → calm recovery synthetic price series.

```bash
PYTHONPATH=. python3 demo/run_pipeline_demo.py
```
(`PYTHONPATH=.` is needed since `demo/` isn't installed as a package.)

**What it validated:** the uncertainty gate firing exactly at a genuine
regime transition (confidence dropped, allocation halved, "UNCERTAINTY"
appeared in the reasoning trail); the state→tier mapping working correctly
end-to-end; and — unprompted but worth knowing — the recovery phase landing
in the mid-volatility tier rather than snapping straight back to low-vol,
because realized volatility takes time to fully decay after a crash. That's
the system behaving as designed, not a bug.

**What it did NOT validate:** the real HMM (still needs `hmmlearn` in a
networked environment), `should_rebalance()` filtering (the demo shows the
raw day-to-day signal, not gated trades), or anything from the still-unbuilt
risk manager.

## Full walk-forward backtest

`backtest/run_backtest_demo.py` runs an actual walk-forward backtest on ~3.4
years of synthetic multi-phase data (calm/crash/recovery/choppy/crash/calm),
4 windows at IS=252/OOS=126/step=126.

```bash
PYTHONPATH=. python3 backtest/run_backtest_demo.py
```

Unlike the earlier pipeline demo, this one is fit **only on each window's
IS data** (never sees OOS at fit time) and drives the real `forward_filter()`
bar-by-bar through OOS — not a simplified confidence heuristic. Everything
downstream of model-fitting (`rank_states_by_volatility`, `volatility_tier`,
`StrategyOrchestrator`, `compute_rebalance`, `should_rebalance`, every
function in `metrics.py`) is the actual production code.

**What's still a stand-in:** `FakeGaussianHMM` in that file estimates
Gaussian-HMM-shaped parameters (`startprob_`, `transmat_`, `means_`,
`covars_`) from IS data via empirical quantiles/moments plus Laplace
smoothing, rather than running hmmlearn's EM. Real numbers will shift once
run against `hmmlearn` in a networked environment — treat this run's
metrics as a pipeline validation, not a real performance estimate.

**Interpretation choice:** since the strategy continuously rebalances
rather than opening/closing discrete positions, "trade P&L" is defined as
the equity change between one rebalance and the next, not a classic
round-trip. Affects how to read win rate / profit factor.

**Bug caught mid-build:** the empirical transition matrix could have exact
zero entries for transitions never observed in a given IS window, producing
`log(0)` noise in the forward filter. Fixed with a small Laplace-smoothing
floor before normalizing — also more realistic, since a real EM-fit HMM
rarely produces literal zero probabilities either. This changed the trade
count in the demo run from 33 to 23, a reminder of how sensitive
walk-forward results are to model estimation details.

**Not yet built:** regime-specific and confidence-bucketed breakdown
tables, the three benchmark comparisons (buy-and-hold, 200-SMA trend,
random-entry Monte Carlo), and Rich-formatted terminal output (plain
`print()` for now).

## Safety layer

`safety/circuit_breakers.py` and `safety/position_sizing.py` — both fully
tested (18 test cases total across both files).

**Circuit breakers**, tested: normal operation; a daily reduce triggering
and staying sticky through a partial intra-day recovery; escalation from
reduce to halt within the same day; a new day correctly resetting the daily
halt while leaving an active weekly halt untouched (independent axes); the
peak-drawdown halt writing a lock file with the correct structured content
(breaker type, drawdown, equity, positions closed, regime at the time); the
halt confirmed to NOT self-clear even on a full equity recovery to new
highs; a simulated process restart correctly starting halted because the
lock file already exists, with no threshold breach needed; and a fresh
engine after manual lock-file removal resuming normally.

**Position sizing**, tested: each of the three caps (risk-based, regime,
portfolio) confirmed binding in turn depending on inputs; a position below
the $100 minimum correctly zeroed out; a missing stop loss correctly
refused rather than silently sized; and the overnight gap-risk adjustment
checked against hand-derived numbers — including confirming a real
structural property of these exact parameters: since the gap-risk
coefficient (2%/3x ≈ 0.67%) is always smaller than the base risk coefficient
(1%) for the same stop distance, overnight sizing will always reduce
further whenever risk-based sizing was already the binding daytime cap.
That's a consequence of the chosen numbers, not a bug — worth knowing if
overnight positions consistently come out smaller than expected.

**Not yet built:** order validation (buying power, tradeable status,
bid-ask spread, duplicate-order blocking) and the correlation check
(60-day rolling correlation, size reduction above 0.7, rejection above
0.85). The portfolio-level exposure/position-count/sector limits from this
spec are configured (`settings.yaml`) but not yet enforced by code.

## Orchestration layer

`orchestration/` — startup, the per-bar main loop, shutdown, and the
error-handling policy. Fully tested with fake Broker/RegimeProvider/
DataFeed/Dashboard implementations, since the real ones either need live
external resources (Robinhood session, hmmlearn, a live data feed) or don't
exist yet (Dashboard). 8 test cases, including:

- Startup correctly forces a retrain with no prior session state, skips
  retraining when the saved model is fresh, forces it again when the saved
  model is >7 days old, and recovers the prior regime label from a saved
  session.
- An HMM error **holds the current regime** rather than resetting or
  crashing — verified the regime label is literally unchanged after a
  simulated HMM failure.
- A data-feed drop **pauses new signals without crashing**, leaves the
  regime untouched, shows up in the dashboard snapshot, and correctly
  clears itself on the next successful bar.
- An unhandled exception (simulated as a broker call that fails past its
  retry budget) is logged with a full traceback, session state is saved
  anyway, a critical alert fires, and — deliberately — the exception still
  propagates rather than being silently swallowed and looped past.
- Shutdown confirmed to leave positions genuinely untouched (checked both
  the returned summary and the broker's own state), per "do NOT close
  positions."

Two ordering/scope notes surfaced while building this, flagged for
confirmation: "connect to Robinhood, verify account" and the later "log
onto Robinhood" in the spec are treated as the same step, done once, first.
And the loop's "5-min bars" default is implemented as the interval for
circuit-breaker/trailing-stop checks — regime/signal recomputation only
actually refreshes when new daily-bar data lands, since the strategy itself
is built on daily features.

**Not yet wired up:** the real `Broker`/`RegimeProvider`/`DataFeed`
implementations (would wrap the existing `broker` wrapper, `regime_detector`
+ `signal_generator`, and `data/fetchers.py` respectively), the Dashboard
itself, and actual trailing-stop updates / order placement inside the loop
body.

## Dashboard

`dashboard/terminal_dashboard.py` — stdlib-only text rendering (no `rich`
needed, though it's still a fine visual upgrade later), so this could
actually be rendered and tested here rather than written blind against an
unavailable dependency.

Tested: the flicker-count and stability helper metrics against hand-verified
sequences; a full render built from **real** `RegimeState`, `PortfolioState`
(reusing the actual `current_allocation_pct` from `backtest/portfolio.py`),
and `Signal` objects — not standalone fakes — confirming the adapter
functions genuinely integrate with what the rest of the system already
produces; and graceful (non-crashing) rendering with zero positions and zero
recent signals.

One naming/organization note surfaced while wiring this up, **now resolved**:
`PortfolioState`/`current_allocation_pct` moved from `backtest/portfolio.py`
to `allocation/portfolio.py` — the `allocation/` package had been sitting
empty since the very first turn of this project, reserved for exactly this
("takes the Brain's target signal and translates it into actual order sizes
given current positions and cash"). Both `backtest/run_backtest_demo.py`'s
import and this file's docstring reference were updated, and the full test
suite (portfolio math, the complete walk-forward backtest run, and the
dashboard adapter) was re-run afterward — identical results to before the
move, confirming it was purely organizational with zero behavior change.

Also: the dashboard's 20-bar "Flicker" window is a separate, display-only
metric from `is_flickering()`'s 5-bar uncertainty-trigger window — if the
live loop wants to feed both from one rolling buffer, that buffer needs to
hold at least 20 entries, not 5.

## Web dashboard

`dashboard/web_dashboard.py` — same `DashboardSnapshot` data model as the
terminal version, rendered as a self-contained HTML page with a
meta-refresh tag instead of monospace text.

**Important limitation, stated plainly:** this chat (or any Claude.ai
artifact) cannot connect live to a process running on your machine — there
is no bridge between a locally-running Python script and anything rendered
in the browser here. What this file is actually for: once the live loop is
wired up, it calls `render_html_dashboard()` and writes the result to a
fixed path after every iteration; a browser tab left open on that file
refreshes itself automatically (every 5 seconds by default, matching
`settings.yaml`'s `monitoring.dashboard_refresh_seconds`) because the
browser just re-reads the file from disk — no server needed. That's where
genuine continuous monitoring lives, not in this conversation.

Tested: the refresh tag defaults to 5 seconds and is confirmed configurable;
real data (built through the same `RegimeState`/`PortfolioState`/`Signal`
adapters as the terminal version) renders correctly; and empty
positions/signals render gracefully rather than breaking the page.

## Risk status section

Added a fourth dashboard section (between Portfolio and Positions in both
the terminal and HTML renderers): `Daily DD: %`.

Rather than recomputing daily drawdown separately for display, `BreakerStatus`
(in `safety/circuit_breakers.py`) now carries `daily_dd_pct` directly — the
exact number `CircuitBreakerEngine.evaluate()` itself is acting on — and
`build_risk_snapshot()` just reads it off. That was a real gap before this:
the field didn't exist on `BreakerStatus`, so displaying this would have
meant a second, parallel drawdown calculation that could silently drift
from what the actual breaker was doing.

Tested: the full pre-existing circuit-breaker test suite re-run and
confirmed passing after the change (purely additive, no regressions);
`daily_dd_pct` confirmed correct across the normal/reduce/halt states; and
— the case most likely to have been overlooked — confirmed it reports a
real, informative number (not a hardcoded `0.0`) in both the peak-drawdown
halt path and the already-halted early-return path. Both dashboards were
then re-tested end to end using a real `CircuitBreakerEngine.evaluate()`
call rather than a standalone fake, confirming the displayed value is
exactly what the breaker computed. The HTML dashboard's 5-second refresh
default is unchanged and reconfirmed.

## Order validation, correlation, portfolio limits, and live order placement

Closes out three of the four "Next up" items below: order validation,
correlation-based sizing, portfolio-level limits, and a real `Broker`
implementation wired into `main_loop.run_bar`'s order-placement integration
point. **Restricted, on request, to a single Robinhood account** — see
`ROBINHOOD_ACCOUNT_NUMBER` in Setup above and `broker/robinhood_broker.py`'s
module docstring for exactly how and how strongly that's enforced.

**Cannot be tested in this sandbox at all** — there's no Python runtime
here (not even python3), on top of the pre-existing no-network/no-hmmlearn
limitations noted throughout this file. Every module below is written and
reasoned through carefully, matching the hand-verified style used
elsewhere in this project, but literally none of it has executed. Treat it
as a draft that needs a real test pass (ideally against a paper account or
a single-share live order) before it's trusted with real size.

- **`safety/order_validation.py`** — buying power, tradability, bid-ask
  spread (skipped, not assumed-safe, when bid/ask aren't available — the
  current `RobinhoodQuoteFetcher.get_quote()` doesn't populate them), and
  an in-memory `DuplicateOrderGuard` cross-checked against the broker's own
  open orders so a crash-and-restart doesn't lose the "already have an
  order out" state.
- **`safety/correlation.py`** — the spec's "size reduction above 0.7,
  rejection above 0.85" implemented as a **linear ramp** from full size at
  0.70 down to zero at 0.85, not a step function (flagged as an
  interpretation call in the module docstring — a step would treat 0.71
  and 0.84 identically, which seemed wrong for a risk control). Correlation
  is checked against every current holding independently; the worst
  (highest) pairwise correlation binds.
- **`safety/portfolio_limits.py`** — `max_exposure`, `max_concurrent_positions`,
  and `max_sector_exposure`, gated only on buys (a sell that reduces
  exposure never needs permission). Sector exposure needs a symbol->sector
  map, which didn't exist anywhere in the codebase; added a small hand-built
  one (`SECTOR_MAP`) covering just the `core_watchlist` tickers. A symbol
  outside that map is exempted from the sector check (logged, not silently
  passed) rather than guessed at — extend the map before trading names
  outside the core watchlist.
- **`broker/robinhood_broker.py`** — the real `Broker` implementation
  `broker/__init__.py` never had. Every account-scoped `robin_stocks` call
  passes `account_number=` explicitly, and `verify_account()` fails closed
  if the account that comes back doesn't match `ROBINHOOD_ACCOUNT_NUMBER`
  exactly. The real caveat, stated in its docstring: `robin_stocks`'
  `account_number` support is unofficial and has varied across versions and
  functions — this was written against the documented pattern, but
  **confirm your installed version actually honors it on every order/
  account call before running this unattended**, e.g. by placing one
  single-share order and checking which account it landed in.
- **`orchestration/order_execution.py`** — `OrderExecutor`, the piece that
  turns a `brain.signal_generator.Signal` into a validated order and
  either logs it (dry run) or calls `broker.place_order`. Uses
  `safety/position_sizing.compute_position_size` (fixed-fractional risk
  sizing with per-position/portfolio caps) rather than
  `allocation/portfolio.compute_rebalance` for sizing — that module models
  a single instrument's cash+shares (the backtester's shape) and was never
  extended to a real multi-symbol live book; flagged in the module
  docstring as something to revisit if the allocation layer is ever
  generalized.
- **`orchestration/main_loop.py`** — `Broker` protocol gained
  `get_buying_power`, `get_open_orders`, `place_order`, `cancel_order`.
  `run_bar`/`run_forever` gained optional `orchestrator`/`order_executor`
  parameters (default `None`, so every existing caller keeps working
  unchanged) that, when both are supplied, generate signals and place
  orders each bar — skipped entirely while the circuit breaker reports
  `trading_halted`. **Known simplification, flagged in the docstring**:
  `RegimeProvider.current_regime()` returns one regime for the whole bar
  (its original, dashboard-oriented shape), so signal generation currently
  reuses that single regime state across every symbol in the watchlist
  rather than tracking one regime per symbol — fine if the watchlist is
  voting on one shared market regime, worth revisiting otherwise.
- **Dry-run by default** — `settings.yaml`: `broker.live_trading_enabled`
  (default `false`). With it false, every order that clears every safety
  gate is logged as `[DRY RUN] would place order: ...` and `broker.place_order`
  is never called. This wasn't explicitly requested but felt like the
  responsible default for a pipeline that has never executed against a real
  account — flip it deliberately, after reviewing dry-run output, not as
  part of getting this running at all.

**Still not built, so nothing here can actually run end-to-end yet**: a
real `RegimeProvider` (needs `hmmlearn` + a live fit, per the pre-existing
"not fully testable here" notes above) and a real `DataFeed`. Order
placement is wired to the point where `run_bar` will call it the moment
those two exist — it just has nothing feeding it signals yet without them.

## Fair Value Gap + volume momentum options strategy

**Superseded the earlier "reuse the equity regime signal" version of this
section entirely**, per request: the options pipeline no longer looks at
the HMM/regime brain at all. It now runs its own momentum signal off raw
price/volume action — Fair Value Gaps confirmed by volume — independent of
`brain/signal_generator.py`. One real upside of that independence: since
this only needs OHLCV bars, **the options pipeline no longer needs a real
`RegimeProvider` to exist before it can trade** — it only needs a real
`DataFeed`, which is a smaller remaining gap than the equity side has.

**Options approval is confirmed on the Agentic account** (this was the
blocker flagged previously — resolved). Same account restriction as
everything else in this project: every options call still goes through
`ROBINHOOD_ACCOUNT_NUMBER`.

**Hard rule, enforced twice, independently**: no 0DTE (same-day
expiration) trades, ever. `config/config_loader.py`'s `OptionsConfig`
rejects `options.target_dte_min < 1` at config-load time, and
`data/options_data.py`'s `RobinhoodOptionChainFetcher._pick_expiration()`
re-checks the same floor (`MIN_ALLOWED_DTE = 1`) at the point of contract
selection, regardless of what settings.yaml says. Two enforcement points
on purpose — a bypassed or hand-edited config still can't produce a 0DTE
pick through this code path. (Closing an existing position that's drifted
down to its last day, e.g. after the bot was offline for a few days, is
unaffected by this — the floor only blocks *opening* new 0DTE positions.)

- **`brain/fvg_indicators.py`** — the supplied `detect_fvg()` function,
  translated to this project's lowercase OHLCV column convention with the
  three-candle detection math otherwise UNCHANGED (same displacement-body-
  vs-average-body criteria, same defaults: `lookback_period=10`,
  `body_multiplier=1.5`). Volume confirmation is a separate, additive
  field on each detected gap (`volume_confirmed`: did the displacement
  candle's volume exceed its own trailing average by
  `options.fvg_volume_multiplier`, default 1.5x?) rather than fused into
  the geometry check — keeps "does a gap exist" and "is it strong enough to
  trade" as two independently-inspectable answers.
- **`brain/options_strategy.py`** — rewritten from the regime-based version.
  FVG+volume is now only the TRIGGER (a candidate direction), gated by a
  full confluence check before it becomes an actual decision — see next
  bullet. Looks only at the FVG on the MOST RECENT bar (trades the
  formation event itself, not some stale historical gap). No gap, an
  unconfirmed one, or a gap that fails confluence -> `hold` — explicitly
  NOT the same as closing an existing position; a quiet bar between
  momentum bursts shouldn't by itself exit a trade.
- **`brain/confluence.py`, plus five new indicator modules it composes**
  — added on request ("we should be able to understand price action,
  market structure, support and resistance, supply and demand, trend
  analysis, liquidity, volume profile" + Elliott Wave + "should meet 100%
  of these indicators to execute the trade"). Two tiers:
  - **Hard vetoes** (any one blocks the trade outright): trading against
    the `brain/trend_indicators.py` 200-SMA (`options.sma_period`), trading
    against a clear opposing `brain/market_structure.py` trend
    (higher-highs/higher-lows swing structure), or entering right as
    `brain/elliott_wave.py` says a same-direction impulse just completed
    (chasing exhaustion instead of the move).
  - **Soft confluence checks** (pass/fail/not-applicable, scored only over
    applicable ones): break of structure, support/resistance (blocks
    buying directly into a resistance/support wall), supply/demand zone
    alignment (`brain/supply_demand.py`), a liquidity sweep in the trade's
    favor (`brain/liquidity.py`), and trading outside the
    `brain/volume_profile.py` value area (a breakout, not chop).
    `options.min_confluence_score` is set to **1.0 — every single
    applicable check must agree, no exceptions** — among at least 3
    applicable checks; fewer than 3 applicable checks blocks the trade too
    (not enough evidence either way). This is deliberately the strictest
    possible setting: expect this to reject far more candidates than it
    approves, which is the intended effect of "if trades should not be
    taken they should not be taken."
  - **Read this regardless of the threshold chosen**: `confluence.py`'s
    score is a COUNT of how many independent, rule-based checks agree —
    it is explicitly **not** a backtested, calibrated probability of
    profit, even at 100%. "Every check I built agrees" is a real,
    meaningful selectivity filter; it is not the same claim as "this trade
    has a known, measured win rate," which would require actually
    backtesting this exact rule set against historical data (see the new
    "Backtesting the FVG/confluence options strategy" section below).
  - **`brain/elliott_wave.py`'s own honesty note**: genuine Elliott Wave
    analysis is famously subjective; this implements only the three
    OBJECTIVE validity rules (wave 2 doesn't retrace past wave 1's start,
    wave 3 isn't the shortest of 1/3/5, wave 4 doesn't overlap wave 1) as a
    heuristic proxy, not a full subjective wave count.
- **Stop-loss, new**: `safety/options_sizing.py` is unchanged (still sized
  by premium at risk), but `orchestration/options_execution.py` now checks
  every open position, every bar, before consulting any new signal: force-
  close inside `options.close_before_expiration_days` (expiration, checked
  first) -> force-close if current value has fallen `options.stop_loss_pct`
  (default 50%) below entry premium (stop-loss, checked second) -> flip on
  a fresh OPPOSING signal -> open on a fresh signal with nothing held ->
  hold otherwise. The stop-loss check fetches a fresh quote for the held
  contract every bar regardless of what the FVG/confluence signal says
  that bar — monitoring an open long option's value isn't something that
  can wait for a new signal to show up.
- **Wired into `main_loop.run_bar`** independently of the equity
  `orchestrator`/`order_executor` — verified this actually works
  independently while rewiring: `run_options` no longer requires
  `orchestrator is not None`, so options can trade with equities on, off,
  or never-built.
- **Still just as unverifiable as before, if not more so**: robin_stocks'
  options endpoints (`get_chains`, `find_tradable_options`,
  `get_option_market_data_by_id`, `get_open_option_positions`,
  `order_buy_option_limit`/`order_sell_option_limit`) have a less stable,
  less-documented history than its equity endpoints, and none of it could
  be checked against a live account or even a live install here (still no
  Python runtime in this sandbox at all). Field names in
  `build_open_positions_from_broker` (`chain_symbol`, `strike_price`,
  `expiration_date`, `type`, `average_price`) are this file's biggest
  unverified assumption — confirm them against one real
  `get_open_option_positions()` response before trusting anything that
  reads existing positions. The FVG math itself (`fvg_indicators.py`) has
  no such dependency and is straightforward to sanity-check by eye against
  a chart in your own environment before trusting it live.
- **What this hasn't been asked to handle, and doesn't**: intraday
  timeframes. `detect_fair_value_gaps()` is timeframe-agnostic — it works
  on whatever bars DataFrame it's given — but daily bars will produce far
  fewer, coarser signals than the 5m/15m charts FVG/momentum strategies are
  usually run on. Pointing this at intraday bars just means passing a
  different `timeframe` to whatever `DataFeed` eventually gets built; no
  code here assumes daily bars specifically.

## Backtesting the FVG/confluence options strategy

`backtest/options_strategy_backtest.py` — walk-forward, no lookahead
(bar *i* only ever sees `bars.iloc[:i+1]`, matching what the strategy would
actually see live), calling `brain.options_strategy.decide_options_action()`
directly — **the real production decision function**, not a
reimplementation of its logic, so a bug in the strategy shows up here
rather than being backtested-around by accident.

**Read the module's own docstring before trusting any number it prints** —
summarized: the DECISION logic is real and exact; the option PRICES are
not. There is no free historical options-pricing data source anywhere in
this project's toolset (yfinance: no historical option chains; robin_stocks:
live chains only), so `backtest/options_pricing.py` prices every simulated
trade with Black-Scholes — real historical spot prices, ATM strike, and a
trailing-realized-volatility proxy for IV, held constant for each trade's
life. No transaction costs or slippage either. This validates that the
*logic* works and gives a rough, clearly-approximate read on historical
edge — it is not a real historical P&L figure, and shouldn't be sized
capital off of.

Reuses `backtest/metrics.py` (`win_rate`, `profit_factor`, `avg_win_loss`,
`max_consecutive_losses`, `max_drawdown`) rather than reimplementing
performance stats a second time.

```bash
PYTHONPATH=. python3 backtest/options_strategy_backtest.py \
    --symbols SPY QQQ AAPL NVDA \
    --start 2022-01-01 --end 2024-12-31
```

Prints, per symbol: bars evaluated, how many FVG+volume triggers fired, how
many the confluence gate rejected (with `min_confluence_score` now at 1.0,
expect this rejection rate to be high — that's the intended effect, not a
bug), and trades opened/closed/still-open. Then pooled closed-trade stats
(win rate, avg win/loss, profit factor, max consecutive losses, max
drawdown) across every symbol's closed trades, sorted by exit date —
flagged in the code as a simplification too: it treats every trade as using
100% of one shared capital pool sequentially, not a real multi-symbol
concurrent-position allocation (that gap is the same one flagged under
`orchestration/order_execution.py`'s sizing note elsewhere in this file).

**Update: since this was written, a real Python environment was actually
set up and this backtest was run for real** — see git history / session
notes for the actual results (SPY/QQQ/AAPL/NVDA, 2022-2024: 1 closed trade
across all four symbols, at `min_confluence_score=1.0`). The caveats above
about theoretical option pricing still apply; only the "couldn't run it at
all" limitation is out of date.

## Live intraday data: Alpaca + Robinhood

`data/fetchers.py` gained two live intraday fetchers, both feeding
`orchestration/run_live.py` (see below) rather than the daily-bar-only
fetchers this project started with:

- **`AlpacaIntradayHistoricalFetcher`** — the default (`--data-source
  alpaca`). Real API contract (`alpaca-py`), not reverse-engineered.
  Free tier: real-time (not delayed) bars, IEX exchange only — meaning
  volume numbers will read much lower than what Robinhood's app or Yahoo
  Finance show (one exchange's volume, not the consolidated tape). This
  doesn't affect the FVG volume-confirmation logic, which compares a bar's
  volume against its own trailing average within the same feed, not
  against some absolute number. Needs a free account at alpaca.markets and
  an API key pair (`ALPACA_API_KEY_ID`/`ALPACA_API_SECRET_KEY` in `.env`)
  — no brokerage funding required, just the signup.
- **`RobinhoodIntradayHistoricalFetcher`** — available via `--data-source
  robinhood`, shares the broker's existing login (`RobinhoodBroker.rh_session`,
  no second auth). Field names (`begins_at`/`open_price`/etc.) were
  checked against one real live API response during development; the
  exact `robin_stocks` interval/span parameter pairing was not (that goes
  through the `robin_stocks` library itself, a different code path from
  the tool used to check field names) — only "5m" and "1h" are wired up,
  the two combinations most consistently documented for that library.

Daily bars (the 200-SMA filter only, see `brain/confluence.py`) stay on
`YFinanceHistoricalFetcher` regardless of which intraday source is
active — a day-old daily close doesn't meaningfully change a 200-day
average, so the freshness that matters intraday doesn't matter there.

**`.env` loading was a real gap, now fixed**: nothing in this project
called `load_dotenv()` anywhere before this — `orchestration/run_live.py`
and `dashboard/live_account_dashboard.py` now do, at the top of their
`if __name__ == "__main__":` blocks, before any credential is read.
`python-dotenv` added to `requirements.txt` (previously only present as an
incidental transitive dependency of `robin_stocks`).

## `orchestration/run_live.py` — the assembled live-trading entrypoint

Ties every already-built piece together into one process that actually
calls `broker.place_option_order` in a standing loop — nothing before this
file did. Polls every `--poll-seconds` (default 300s = 5min, matched to
`--interval`'s bar size) and re-evaluates the whole watchlist each cycle;
no freshness/dedup bookkeeping needed since re-evaluating an unchanged bar
is already idempotent (`OptionsOrderExecutor` won't add to a position it's
already holding; `DuplicateOrderGuard` independently blocks resubmitting).
Respects `options.live_trading_enabled` exactly like the backtester and
dashboard — dry-run (logged, not sent to the broker) until deliberately
flipped, safe to run continuously by default.

## `deploy/` — running this unattended on a server, on a market-hours schedule

Added for genuine hands-off operation, independent of any local machine or
Claude session being open. `deploy/README.md` is the full walkthrough
(connect, copy the project over, provision, create `.env` on the server,
enable the schedule); `deploy/setup.sh` installs Python/build tools,
creates a dedicated unprivileged `trading-bot` system user (so the process
holding live credentials isn't running as root or a personal login), and
installs the systemd units below. Provider-agnostic — written for Oracle
Cloud's Always Free tier (Ubuntu 22.04/24.04) but nothing in it is
Oracle-specific; works on any similar VPS. The account/server creation
itself is necessarily a you-not-Claude step (payment/identity
verification).

**Runs on a schedule, not 24/7**: `deploy/trading-bot.service` (the actual
`run_live.py` process — auto-restarts on crash, capped at 5 attempts per
10 minutes so a persistent failure like bad credentials doesn't hammer the
Robinhood/Alpaca APIs forever) is started and stopped by two `systemd`
timers, `deploy/trading-bot-start.timer`/`-stop.timer`, at 9:30am/4:00pm
America/New_York on weekdays — an explicit IANA timezone in the calendar
spec, so EST/EDT daylight saving is handled automatically regardless of
what timezone the server itself is set to. Only the *timers* get
`systemctl enable`d (so the schedule survives a reboot); the main service
is deliberately left un-enabled so it doesn't start immediately on every
boot regardless of time of day. One inherited gap: no holiday calendar
(same limitation `orchestration/market_hours.py` already had) — a market
holiday still triggers the start timer, harmlessly (it polls a market
that isn't moving and does nothing).

## Next up

1. **Real `RegimeProvider` and `DataFeed`** — the two integration points
   order placement is now waiting on; everything downstream of a Signal is
   wired (see above).
2. **The Dashboard's `Dashboard` protocol implementation** — `refresh()`/
   `alert()` wrapping `dashboard/terminal_dashboard.py` or
   `dashboard/web_dashboard.py`, plus surfacing the new `order_records` key
   `run_bar` now passes into `dashboard.refresh()`'s snapshot.
3. **Trailing-stop updates inside the loop** — order placement handles
   entries/rebalances; nothing yet manages an existing stop as price moves.
4. **Rest of the backtester** — regime-specific and confidence-bucketed
   breakdown tables, benchmark comparisons (buy-and-hold, 200-SMA trend,
   random-entry Monte Carlo), and Rich-formatted terminal output.
