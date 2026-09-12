# Indicator attribution — 2026-07-15 .. 2026-09-11

Symbols: SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, NVDA, META, TSLA, AMD. Policy: hard = ['trend_1h', 'trend_4h', 'market_structure'], soft = ['volume_profile', 'trend_200sma', 'support_resistance'], ranging-structure veto = True, min applicable = 2.

## Headline (current config)
- Triggers (FVG + volume): **744** · rejected by confluence: 692 · trades opened: 39 · closed: 39
- Win rate **41.0%** · avg win +110.3% · avg loss -31.5% · **profit factor 2.436** · sum of trade P&L +1040.1%

| exit reason | n | win rate | avg P&L | total |
|---|---|---|---|---|
| trend_invalidated | 17 | 76% | +61.7% | +1048.7% |
| fvg_invalidated | 16 | 0% | -24.7% | -395.0% |
| max_hold | 6 | 50% | +64.4% | +386.5% |

## What is doing the filtering
Of 744 triggers, the named rejection reasons were:
- trend_1h: 332
- trend_4h: 153
- market_structure (ranging): 96
- soft score below minimum: 61
- market_structure (opposing): 50

## How each check read at triggers
| check | role | pass | fail | n/a |
|---|---|---|---|---|
| trend_1h | HARD | 412 (55%) | 332 (45%) | 0 (0%) |
| trend_4h | HARD | 357 (48%) | 387 (52%) | 0 (0%) |
| market_structure | HARD | 270 (36%) | 209 (28%) | 265 (36%) |
| elliott_wave | unused | 0 (0%) | 27 (4%) | 717 (96%) |
| break_of_structure | unused | 592 (80%) | 152 (20%) | 0 (0%) |
| support_resistance | soft | 325 (44%) | 165 (22%) | 254 (34%) |
| supply_demand | unused | 359 (48%) | 385 (52%) | 0 (0%) |
| liquidity_sweep | unused | 64 (9%) | 85 (11%) | 595 (80%) |
| volume_profile | soft | 363 (49%) | 381 (51%) | 0 (0%) |
| rsi_momentum | unused | 600 (81%) | 144 (19%) | 0 (0%) |
| volatility_expansion | unused | 559 (75%) | 185 (25%) | 0 (0%) |
| vpvr_node_quality | unused | 397 (53%) | 347 (47%) | 0 (0%) |
| trend_200sma | soft | 351 (47%) | 393 (53%) | 0 (0%) |

## Outcome by each check's reading at entry
A check is informative when its PASS trades beat its FAIL trades; equal = noise; inverted = pulling against the others. Splits with fewer than 10 trades on a side are marked ⚠ thin.

| check | role | pass n / win / avg | fail n / win / avg | n/a n / win / avg | edge (pass−fail) |
|---|---|---|---|---|---|
| trend_1h | HARD | 39 / 41% / +26.7% | — | — | — |
| trend_4h | HARD | 39 / 41% / +26.7% | — | — | — |
| market_structure | HARD | 39 / 41% / +26.7% | — | — | — |
| elliott_wave | unused | — | 6 / 50% / +89.0% | 33 / 39% / +15.3% | — |
| break_of_structure | unused | 34 / 41% / +33.0% | 5 / 40% / -16.6% | — | +49.7% ⚠ thin |
| support_resistance | soft | 13 / 46% / +9.9% | 1 / 0% / -20.8% | 25 / 40% / +37.3% | +30.8% ⚠ thin |
| supply_demand | unused | 11 / 36% / +5.3% | 28 / 43% / +35.1% | — | -29.8% |
| liquidity_sweep | unused | — | 4 / 25% / -1.0% | 35 / 43% / +29.8% | — |
| volume_profile | soft | 36 / 44% / +29.6% | 3 / 0% / -9.0% | — | +38.7% ⚠ thin |
| rsi_momentum | unused | 29 / 45% / +20.2% | 10 / 30% / +45.4% | — | -25.2% |
| volatility_expansion | unused | 27 / 41% / +11.6% | 12 / 42% / +60.6% | — | -49.1% |
| vpvr_node_quality | unused | 24 / 42% / +27.2% | 15 / 40% / +25.8% | — | +1.5% |
| trend_200sma | soft | 33 / 39% / +31.0% | 6 / 50% / +2.8% | — | +28.2% ⚠ thin |

## Ranked
Readable splits, most harmful/uninformative first:
- volatility_expansion: edge -49.1% — harmful (inverted)
- supply_demand: edge -29.8% — harmful (inverted)
- rsi_momentum: edge -25.2% — harmful (inverted)
- vpvr_node_quality: edge +1.5% — noise

Too thin to call (one side under 10 trades): break_of_structure (+49.7%), support_resistance (+30.8%), volume_profile (+38.7%), trend_200sma (+28.2%)

No pass-vs-fail split available (hard vetoes never enter on a fail; or a check never read fail on an entered trade): trend_1h, trend_4h, market_structure, elliott_wave, liquidity_sweep